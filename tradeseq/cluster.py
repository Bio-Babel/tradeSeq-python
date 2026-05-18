"""Cluster gene expression patterns (R source: tradeSeq/R/clusterExpressionPatterns.R).

Implements :func:`cluster_expression_patterns`, the Python port of
``clusterExpressionPatterns``.

Tier-3 deviation
----------------
The R implementation uses ``clusterExperiment::RSEC``, a consensus-clustering
pipeline (PCA reduction -> many clusterings -> consensus merging) that is not
available in the Python ecosystem and whose output is rarely reproducible
without the upstream Bioconductor stack. We **delegate to**
``sklearn.cluster.AgglomerativeClustering`` (Ward linkage) on the same
row-standardised mean-expression profiles that R clusters; when ``n_clusters``
is not given we sweep a range of candidate cluster counts and pick the one
with the largest ``sklearn.metrics.silhouette_score``. Shape and tidy
long-form output are preserved; cluster labels are not directly comparable to
RSEC labels.

Return-structure parity
-----------------------
R's ``clusterExpressionPatterns`` returns ``list(rsec=<ClusterExperiment>,
yhatScaled=<matrix>)`` (clusterExpressionPatterns.R:64, 133). We return a
:class:`ClusterResult` :func:`typing.NamedTuple` exposing:

* ``cluster_labels`` -- a :class:`pandas.Series` indexed by gene with the
  integer cluster id (analogous to ``clusterExperiment::primaryCluster(rsec)``).
* ``yhat_scaled`` -- a :class:`pandas.DataFrame` ``(n_genes, n_lineages *
  n_points)`` of row-standardised profiles, with R-style column names
  (``l{ii}:t{t}`` without conditions, ``l{ii}_{kk}:t{t}`` with conditions) and
  gene rownames.
* ``long_df`` -- the original long-form DataFrame ``(gene, cluster, profile,
  time_point)`` retained for notebook-style usage.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ._design import get_predict_range_df
from ._predict import predict_gam
from ._slot_io import read_beta, read_conditions, read_design_matrix
from ._test_common import (
    lpmatrix_dataframe,
    n_curves_from_dm,
    pseudotime_matrix,
)
from .fit_gam import FittedGam

__all__ = ["cluster_expression_patterns", "ClusterResult"]


class ClusterResult(NamedTuple):
    """Return value of :func:`cluster_expression_patterns`.

    Mirrors R's ``list(rsec=<ClusterExperiment>, yhatScaled=<matrix>)``
    (clusterExpressionPatterns.R:64, 133), augmented with the long-form
    profile DataFrame used by notebooks.

    Attributes
    ----------
    cluster_labels : pandas.Series
        Integer cluster id per gene, indexed by gene label. Analogous to
        ``clusterExperiment::primaryCluster(rsec)`` over
        ``rownames(yhatScaled)``.
    yhat_scaled : pandas.DataFrame
        Row-standardised per-gene profiles, shape ``(n_genes, n_lineages *
        n_points)``. Column names follow R's convention -- ``l{ii}:t{t}``
        without conditions, ``l{ii}_{kk}:t{t}`` with conditions.
        Index is gene labels.
    long_df : pandas.DataFrame
        Long-form view with columns ``gene, cluster, profile, time_point``.
        Length ``n_genes * (n_lineages * n_points)``.
    """

    cluster_labels: pd.Series
    yhat_scaled: pd.DataFrame
    long_df: pd.DataFrame


def _first_valid_fitted_gam(
    models: dict[str, FittedGam | None], *, context: str
) -> FittedGam:
    """Return the first successful list-mode fit, mirroring R's reference scan."""
    if not models:
        raise ValueError(f"{context}: empty models dict.")
    for fg in models.values():
        if fg is not None:
            return fg
    raise ValueError(f"{context}: all models are failed fits.")


def _resolve_gene_ids(
    var_names: pd.Index, genes: Union[Sequence[str], Sequence[int]]
) -> tuple[list[int], list[str]]:
    """Resolve a heterogeneous ``genes`` argument into row indices and labels.

    Mirrors R's ``if (is(genes, "character")) ... else id <- genes``.

    Parameters
    ----------
    var_names : pandas.Index
        ``adata.var_names`` (gene labels in row order).
    genes : sequence of str or sequence of int
        Either gene names (strings) or 0-based row indices (integers).

    Returns
    -------
    indices : list of int
        0-based row indices into ``var_names``.
    labels : list of str
        Gene labels in the same order as ``indices``.
    """
    genes_list = list(genes)
    if len(genes_list) == 0:
        raise ValueError("'genes' must contain at least one entry.")
    first = genes_list[0]
    if isinstance(first, str):
        missing = [g for g in genes_list if g not in var_names]
        if missing:
            raise ValueError(
                f"Not all gene IDs are present in the models object: {missing}"
            )
        indices = [int(var_names.get_loc(g)) for g in genes_list]
        labels = [str(g) for g in genes_list]
    elif isinstance(first, (int, np.integer)):
        indices = [int(i) for i in genes_list]
        if any(i < 0 or i >= len(var_names) for i in indices):
            raise ValueError(
                f"Integer gene indices must be in [0, {len(var_names)}); got {indices}"
            )
        labels = [str(var_names[i]) for i in indices]
    else:
        raise TypeError(
            f"'genes' entries must be all str or all int; got first element of "
            f"type {type(first).__name__}"
        )
    return indices, labels


def _resolve_gene_ids_list(
    keys: list[str], genes: Union[Sequence[str], Sequence[int]]
) -> tuple[list[str], list[str]]:
    """Resolve ``genes`` against a list-mode dict's keys.

    Mirrors clusterExpressionPatterns.R:10-15: when ``genes`` is character,
    every entry must appear in ``names(models)``; otherwise treat as integer
    1-based indices into ``names(models)``.

    Returns
    -------
    selected_keys : list of str
        Dict keys to look up in ``models``.
    labels : list of str
        Gene labels corresponding to ``selected_keys`` (identical to the keys).
    """
    genes_list = list(genes)
    if len(genes_list) == 0:
        raise ValueError("'genes' must contain at least one entry.")
    first = genes_list[0]
    if isinstance(first, str):
        missing = [g for g in genes_list if g not in keys]
        if missing:
            raise ValueError(
                f"Not all gene IDs are present in the models object: {missing}"
            )
        selected = [str(g) for g in genes_list]
    elif isinstance(first, (int, np.integer)):
        idx_list = [int(i) for i in genes_list]
        if any(i < 0 or i >= len(keys) for i in idx_list):
            raise ValueError(
                f"Integer gene indices must be in [0, {len(keys)}); got {idx_list}"
            )
        selected = [keys[i] for i in idx_list]
    else:
        raise TypeError(
            f"'genes' entries must be all str or all int; got first element of "
            f"type {type(first).__name__}"
        )
    return selected, list(selected)


def _build_yhat_matrix_sce_no_cond(
    adata: ad.AnnData,
    gene_indices: Sequence[int],
    n_points: int,
    key: str,
) -> tuple[np.ndarray, list[str]]:
    """AnnData no-conditions branch (clusterExpressionPatterns.R:33-52).

    Mirrors lines 33-52 of ``.clusterExpressionPatterns``: for each lineage
    build the prediction grid, push it through :func:`predict_gam`, multiply
    by the per-gene beta, add ``df$offset``, and concatenate blocks
    horizontally. Column labels follow R's ``paste0("l", ii, ":t", df$tII)``.
    """
    dm = read_design_matrix(adata, key=key)
    lpmatrix = lpmatrix_dataframe(adata, key=key)
    pseudotime = pseudotime_matrix(adata, key=key)
    beta_all = read_beta(adata, key=key)
    beta = beta_all[list(gene_indices), :]  # (n_genes, n_coefs)

    n_curves = n_curves_from_dm(dm)
    yhat_blocks: list[np.ndarray] = []
    col_labels: list[str] = []
    for ii in range(1, n_curves + 1):
        df = get_predict_range_df(dm, ii, n_points=n_points)
        xdf = predict_gam(
            lpmatrix=lpmatrix,
            df=df,
            pseudotime=pseudotime,
            conditions=None,
        ).to_numpy()  # (n_points, n_coefs)
        offset_col = df["offset"].to_numpy()
        # R: y <- t(Xdf %*% t(beta)) + df$offset -> per-row offset broadcast
        y = (xdf @ beta.T).T + offset_col[np.newaxis, :]
        yhat_blocks.append(y)
        time_vals = df[f"t{ii}"].to_numpy()
        col_labels.extend(f"l{ii}:t{t}" for t in time_vals)

    yhat = np.concatenate(yhat_blocks, axis=1)
    return yhat, col_labels


def _build_yhat_matrix_sce_conditions(
    adata: ad.AnnData,
    gene_indices: Sequence[int],
    n_points: int,
    key: str,
) -> tuple[np.ndarray, list[str]]:
    """AnnData conditions branch (clusterExpressionPatterns.R:97-122).

    For each ``(lineage, condition)`` pair, builds the prediction grid via
    :func:`get_predict_range_df` (with ``condition_id``), evaluates
    :func:`predict_gam` with the per-cell conditions vector, multiplies by
    beta, and concatenates. Column labels follow R's
    ``paste0("l", ii, "_", kk, ":t", df$tII)`` (line 118).
    """
    dm = read_design_matrix(adata, key=key)
    lpmatrix = lpmatrix_dataframe(adata, key=key)
    pseudotime = pseudotime_matrix(adata, key=key)
    conditions = read_conditions(adata, key=key)
    beta_all = read_beta(adata, key=key)
    beta = beta_all[list(gene_indices), :]

    n_curves = n_curves_from_dm(dm)
    n_conditions = len(conditions.categories)

    yhat_blocks: list[np.ndarray] = []
    col_labels: list[str] = []
    for ii in range(1, n_curves + 1):
        for kk in range(1, n_conditions + 1):
            df = get_predict_range_df(
                dm, lineage_id=ii, condition_id=kk, n_points=n_points
            )
            xdf = predict_gam(
                lpmatrix=lpmatrix,
                df=df,
                pseudotime=pseudotime,
                conditions=conditions,
            ).to_numpy()
            offset_col = df["offset"].to_numpy()
            y = (xdf @ beta.T).T + offset_col[np.newaxis, :]
            yhat_blocks.append(y)
            time_vals = df[f"t{ii}"].to_numpy()
            col_labels.extend(f"l{ii}_{kk}:t{t}" for t in time_vals)

    yhat = np.concatenate(yhat_blocks, axis=1)
    return yhat, col_labels


def _build_yhat_matrix_list(
    models: dict[str, FittedGam],
    selected_keys: Sequence[str],
    n_points: int,
) -> tuple[np.ndarray, list[str]]:
    """List-mode branch (clusterExpressionPatterns.R:23-32).

    R code calls ``predict.gam(m, newdata=df, type="link")`` on each gam in
    the list. The equivalent Python operation reads the shared design /
    lpmatrix from any :class:`FittedGam`, builds the prediction grid via
    :func:`get_predict_range_df`, reconstructs the linear-predictor matrix
    via :func:`predict_gam`, and computes ``Xdf @ beta + df$offset`` per
    gene.

    Column labels mirror the AnnData no-conditions branch (R's helper does not
    set ``colnames`` in the list path, so we use the AnnData convention
    ``l{ii}:t{t}`` for parity with downstream consumers).
    """
    # Reference model: any FittedGam supplies the shared design / lpmatrix.
    ref = _first_valid_fitted_gam(
        models, context="cluster_expression_patterns"
    )
    dm = ref.dm_
    lpmatrix = ref.lpmatrix_
    n_curves = n_curves_from_dm(dm)
    pseudotime = np.column_stack(
        [dm[f"t{j+1}"].to_numpy(dtype=float) for j in range(n_curves)]
    )

    n_genes = len(selected_keys)
    n_coefs = lpmatrix.shape[1]
    beta = np.full((n_genes, n_coefs), np.nan, dtype=np.float64)
    for i, k in enumerate(selected_keys):
        fg = models[k]
        if fg is None or fg.coef_ is None or np.any(np.isnan(fg.coef_)):
            continue
        beta[i, :] = np.asarray(fg.coef_, dtype=np.float64)

    yhat_blocks: list[np.ndarray] = []
    col_labels: list[str] = []
    for ii in range(1, n_curves + 1):
        df = get_predict_range_df(dm, ii, n_points=n_points)
        xdf = predict_gam(
            lpmatrix=lpmatrix,
            df=df,
            pseudotime=pseudotime,
            conditions=None,
        ).to_numpy()
        offset_col = df["offset"].to_numpy()
        y = (xdf @ beta.T).T + offset_col[np.newaxis, :]
        yhat_blocks.append(y)
        time_vals = df[f"t{ii}"].to_numpy()
        col_labels.extend(f"l{ii}:t{t}" for t in time_vals)

    yhat = np.concatenate(yhat_blocks, axis=1)
    return yhat, col_labels


def _row_standardise(mat: np.ndarray) -> np.ndarray:
    """Row-wise z-score, matching R's ``t(scale(t(mat)))`` (mean 0, sd 1 per row).

    R's ``scale`` uses the sample SD (``n-1`` denominator). Genes whose profile
    has zero variance receive a row of zeros (R would emit ``NaN`` divided by
    zero; we surface this as a deterministic zero row so downstream clustering
    does not silently NaN-poison its inputs).
    """
    mean = mat.mean(axis=1, keepdims=True)
    centered = mat - mean
    # Sample SD with n-1 denominator (R default).
    sd = centered.std(axis=1, ddof=1, keepdims=True)
    out = np.zeros_like(mat)
    nonzero = sd[:, 0] > 0
    out[nonzero] = centered[nonzero] / sd[nonzero]
    return out


def _select_n_clusters(
    profiles: np.ndarray,
    min_size: int,
    random_state: int,
) -> int:
    """Pick the cluster count maximising silhouette over feasible ``k``.

    The candidate range is ``[2, n_genes // min_size]`` clipped to at most
    ``n_genes - 1`` (silhouette is undefined for the trivial partition into
    singletons). If the upper bound is below 2 we fall back to ``k = 2``.

    Parameters
    ----------
    profiles : numpy.ndarray
        ``(n_genes, n_features)`` standardised gene profiles.
    min_size : int
        Minimum cluster size, used to cap the candidate ``k`` range --
        a clustering with ``k`` clusters needs at least ``k * min_size`` genes.
    random_state : int
        Reproducibility seed.
    """
    n_genes = profiles.shape[0]
    k_upper = max(2, n_genes // max(1, min_size))
    k_upper = min(k_upper, n_genes - 1)
    if k_upper < 2:
        return 2

    rng = np.random.default_rng(random_state)
    best_k = 2
    best_score = -np.inf
    for k in range(2, k_upper + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(profiles)
        # silhouette_score is undefined when only one effective cluster.
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(profiles, labels)
        # Deterministic tie-breaking via a tiny RNG perturbation: same k order,
        # rng draws are consumed identically across calls with same seed.
        jitter = rng.random() * 1e-12
        if score + jitter > best_score:
            best_score = score + jitter
            best_k = k
    return best_k


def cluster_expression_patterns(
    models: Union[ad.AnnData, dict[str, FittedGam]],
    *,
    n_points: int,
    genes: Union[Sequence[str], Sequence[int]],
    n_clusters: Optional[int] = None,
    key: str = "tradeseq",
    random_state: int = 176201,
) -> ClusterResult:
    """Cluster gene expression patterns along a trajectory.

    Port of ``tradeSeq::clusterExpressionPatterns``
    (``tradeSeq/R/clusterExpressionPatterns.R``). The R implementation reduces
    the standardised per-gene profile matrix via PCA and runs the
    ``clusterExperiment::RSEC`` consensus-clustering pipeline. We **delegate to
    Ward-linkage agglomerative clustering** (``sklearn.cluster.
    AgglomerativeClustering``) on the same row-standardised profiles. When
    ``n_clusters`` is not supplied we search for the value maximising
    silhouette score over ``[2, n_genes // min_size]``.

    Tier-3 deviation: the cluster labels are not directly comparable to
    R's RSEC labels -- only the input matrix shape and the long-form
    output structure are preserved (see module docstring).

    Parameters
    ----------
    models : anndata.AnnData or dict[str, FittedGam]
        AnnData populated by :func:`tradeseq.fit_gam` or the list-mode dict
        from ``fit_gam(sce=False)``. Both branches dispatch on
        ``isinstance(models, dict)`` while mirroring R's list and
        SingleCellExperiment methods (clusterExpressionPatterns.R:181-251).
    n_points : int
        Number of evaluation points per lineage on the prediction grid.
        Mirrors R's ``nPoints``.
    genes : sequence of str or sequence of int
        Genes to cluster, given either as gene labels matching
        ``adata.var_names`` (AnnData) / ``models`` keys (list) or as 0-based
        row indices. Mirrors R's ``genes``.
    n_clusters : int, optional
        Number of clusters to form. When ``None`` (the default), a silhouette
        sweep selects ``k`` in ``[2, n_genes // min_size]`` where
        ``min_size = max(2, n_genes // 10)``.
    key : str, default ``"tradeseq"``
        ``adata.uns`` namespace prefix used by ``fit_gam`` (AnnData branch only).
    random_state : int, default ``176201``
        Reproducibility seed. The default matches R's ``random.seed = 176201``.

    Returns
    -------
    ClusterResult
        Named tuple with three fields, mirroring R's ``list(rsec, yhatScaled)``
        return contract (clusterExpressionPatterns.R:64, 133) augmented with a
        long-form view:

        * ``cluster_labels`` : :class:`pandas.Series` indexed by gene with the
          integer cluster id.
        * ``yhat_scaled`` : :class:`pandas.DataFrame` ``(n_genes, n_lineages *
          n_points)`` of row-standardised profiles. Column names follow R's
          ``l{ii}:t{t}`` (no conditions) or ``l{ii}_{kk}:t{t}`` (conditions).
        * ``long_df`` : :class:`pandas.DataFrame` long-form view, columns
          ``gene, cluster, profile, time_point``.

    Examples
    --------
    >>> import tradeseq
    >>> adata = tradeseq.load_paul15()
    >>> fitted = tradeseq.fit_gam(adata, genes=adata.var_names[:20],
    ...                           n_knots=6, copy=True)
    >>> res = tradeseq.cluster_expression_patterns(
    ...     fitted, n_points=20, genes=list(adata.var_names[:20]))
    >>> res.cluster_labels.head()
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2; got {n_points}")

    if isinstance(models, ad.AnnData):
        var_names = models.var_names
        gene_indices, gene_labels = _resolve_gene_ids(var_names, genes)
        n_genes = len(gene_indices)
        if n_genes < 2:
            raise ValueError(
                f"Clustering requires at least 2 genes; got {n_genes}."
            )
        conditions = read_conditions(models, key=key)
        if conditions is None:
            yhat, col_labels = _build_yhat_matrix_sce_no_cond(
                models, gene_indices, n_points, key
            )
        else:
            yhat, col_labels = _build_yhat_matrix_sce_conditions(
                models, gene_indices, n_points, key
            )
    elif isinstance(models, dict):
        keys = list(models.keys())
        selected_keys, gene_labels = _resolve_gene_ids_list(keys, genes)
        n_genes = len(selected_keys)
        if n_genes < 2:
            raise ValueError(
                f"Clustering requires at least 2 genes; got {n_genes}."
            )
        yhat, col_labels = _build_yhat_matrix_list(
            models, selected_keys, n_points
        )
    else:
        raise TypeError(
            f"cluster_expression_patterns: unsupported models type "
            f"{type(models).__name__!r}; expected anndata.AnnData or "
            "dict[str, FittedGam]."
        )

    nonfinite = ~np.isfinite(yhat).all(axis=1)
    if np.any(nonfinite):
        bad_genes = [str(gene_labels[i]) for i in np.where(nonfinite)[0]]
        raise ValueError(
            "Cannot cluster genes with non-finite fitted expression profiles: "
            f"{bad_genes}"
        )

    profiles = _row_standardise(yhat)

    # R's RSEC default minSizes=6 is hardcoded for ~thousands of genes; for the
    # ~10s-of-genes gene panels typical of tradeSeq we rescale to a fraction.
    min_size = max(2, n_genes // 10)

    if n_clusters is None:
        k = _select_n_clusters(
            profiles, min_size=min_size, random_state=random_state
        )
    else:
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2; got {n_clusters}")
        if n_clusters > n_genes:
            raise ValueError(
                f"n_clusters ({n_clusters}) cannot exceed n_genes ({n_genes})."
            )
        k = int(n_clusters)

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    cluster_labels_arr = model.fit_predict(profiles).astype(int)

    cluster_labels = pd.Series(
        cluster_labels_arr, index=pd.Index(gene_labels, name="gene"),
        name="cluster",
    )
    yhat_scaled = pd.DataFrame(
        profiles, index=pd.Index(gene_labels, name="gene"), columns=col_labels
    )

    n_cols = profiles.shape[1]
    long_df = pd.DataFrame(
        {
            "gene": np.repeat(gene_labels, n_cols),
            "cluster": np.repeat(cluster_labels_arr, n_cols),
            "profile": profiles.reshape(-1),
            "time_point": np.tile(col_labels, n_genes),
        }
    )
    return ClusterResult(
        cluster_labels=cluster_labels,
        yhat_scaled=yhat_scaled,
        long_df=long_df,
    )
