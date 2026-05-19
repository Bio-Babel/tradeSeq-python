"""Expression-peak cascades (R source: tradeSeq/R/cascade.R).

The R export ``cascade`` mixes two responsibilities: it computes genes whose
fitted expression is significantly increasing along a lineage and, by default,
draws a heatmap of those genes.  The Python port splits that behavior into:

* :func:`cascade` -- computation only.
* :func:`plot_cascade` -- heatmap rendering from the computed result.

This intentionally fixes the upstream R implementation bugs documented in
``tradeseq_porting_essential_suggestions.md``: the R method body references an
unbound source variable instead of its ``models`` argument, uses an out-of-scope
``nPoints`` name in the NaN-coefficient branch, and tries to return ``yHat``
even when ``plotHeatmap=FALSE``.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scales
from pheatmap import PHeatmap, pheatmap
from scipy.stats import norm

from ._design import get_predict_range_df
from ._predict import predict_gam
from ._slot_io import read_beta, read_conditions, read_design_matrix, read_sigma
from ._test_common import lpmatrix_dataframe, n_curves_from_dm, pseudotime_matrix
from .predict_smooth import predict_smooth

__all__ = ["CascadeResult", "cascade", "plot_cascade"]


_ZISSOU1_5_STOPS = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
_YHAT_POINT_RE = re.compile(r"^lineage(?P<lineage>\d+)_(?P<point>\d+)$")
_YHAT_CONDITION_POINT_RE = re.compile(
    r"^lineage(?P<lineage>\d+)_condition(?P<condition>.+)_point(?P<point>\d+)$"
)


class CascadeResult(NamedTuple):
    """Return value of :func:`cascade`.

    Attributes
    ----------
    peak_time : pandas.Series
        Peak pseudotime per significantly increasing gene, indexed by gene.
        Mirrors R's ``peakTime`` vector.
    yhat : pandas.DataFrame
        Wide fitted-expression matrix for ``peak_time.index`` on the requested
        lineage, with R-style ``lineage{lineage}_{point}`` columns. Mirrors R's
        ``yHat`` matrix, but is computed regardless of plotting.
    derivatives : pandas.DataFrame
        First derivative estimates for every requested gene and grid point.
    sd_derivatives : pandas.DataFrame
        Standard errors of the derivative estimates.
    test_statistics : pandas.DataFrame
        Pointwise threshold test statistics.
    p_values : pandas.DataFrame
        Pointwise p-values after R's one-sided positive-derivative filter.
    lineage : int
        1-based lineage id used for the cascade.
    grid : pandas.DataFrame
        Prediction grid returned by ``get_predict_range_df``.
    """

    peak_time: pd.Series
    yhat: pd.DataFrame
    derivatives: pd.DataFrame
    sd_derivatives: pd.DataFrame
    test_statistics: pd.DataFrame
    p_values: pd.DataFrame
    lineage: int
    grid: pd.DataFrame


def _resolve_genes(
    adata: ad.AnnData, genes: Sequence[Union[str, int]] | None
) -> tuple[list[int], list[str]]:
    """Resolve ``genes`` into AnnData column indices and labels."""
    if genes is None:
        return list(range(adata.n_vars)), [str(g) for g in adata.var_names]
    genes_list = list(genes)
    if len(genes_list) == 0:
        raise ValueError("'genes' must contain at least one entry.")
    first = genes_list[0]
    if isinstance(first, str):
        missing = [g for g in genes_list if g not in adata.var_names]
        if missing:
            raise ValueError(
                f"Not all gene IDs are present in the models object: {missing}"
            )
        return [int(adata.var_names.get_loc(g)) for g in genes_list], [
            str(g) for g in genes_list
        ]
    if isinstance(first, (int, np.integer)):
        idx = [int(g) for g in genes_list]
        if any(i < 0 or i >= adata.n_vars for i in idx):
            raise ValueError(
                f"Integer gene indices must be in [0, {adata.n_vars}); got {idx}"
            )
        return idx, [str(adata.var_names[i]) for i in idx]
    raise TypeError(
        f"'genes' entries must be all str or all int; got first element of "
        f"type {type(first).__name__}"
    )


def _calculate_derivative_one_gene(
    *,
    beta: np.ndarray,
    sigma: np.ndarray,
    lpmatrix: pd.DataFrame,
    pseudotime: np.ndarray,
    grid: pd.DataFrame,
    lineage: int,
    epsilon: float,
    conditions: pd.Categorical | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Port ``.calculateDerivativeOneGene`` using finite differencing."""
    n_points = grid.shape[0]
    if np.isnan(beta).any():
        return (
            np.full(n_points, np.nan, dtype=float),
            np.full(n_points, np.nan, dtype=float),
        )

    newd1 = grid.copy()
    newd2 = grid.copy()
    t_col = f"t{lineage}"
    newd1[t_col] = newd1[t_col].astype(float) - epsilon
    newd2[t_col] = newd2[t_col].astype(float) + epsilon
    x0 = predict_gam(
        lpmatrix=lpmatrix,
        df=newd1,
        pseudotime=pseudotime,
        conditions=conditions,
    ).to_numpy(dtype=float)
    x1 = predict_gam(
        lpmatrix=lpmatrix,
        df=newd2,
        pseudotime=pseudotime,
        conditions=conditions,
    ).to_numpy(dtype=float)
    xp = (x1 - x0) / (2.0 * epsilon)
    der = xp @ beta.reshape(-1)
    var_der = np.einsum("ij,jk,ik->i", xp, sigma, xp)
    sd_der = np.sqrt(var_der)
    return der, sd_der


def _lineage_yhat_columns(columns: Sequence[str], lineage: int) -> list[str]:
    """Mirror R's ``grep(paste0('lineage', lineage), colnames(yHat))``."""
    pattern = f"lineage{lineage}"
    return [c for c in columns if pattern in str(c)]


def cascade(
    adata: ad.AnnData,
    *,
    lineage: int = 1,
    genes: Sequence[Union[str, int]] | None = None,
    epsilon: float = 1e-6,
    n_points: int = 100,
    derivative_threshold: float = 0.1,
    der_pval_threshold: float | None = None,
    key: str = "tradeseq",
) -> CascadeResult:
    """Find and order genes peaking in expression along one lineage.

    Port of the computational part of ``tradeSeq::cascade``. First derivatives
    are estimated pointwise by central finite differencing of the reconstructed
    linear-predictor matrix. Genes are retained when at least one grid point has
    a positive derivative significantly above ``derivative_threshold``. Each
    retained gene is ordered by the first post-increase derivative crossing from
    positive to negative; if no crossing exists, the lineage endpoint is used.

    Parameters
    ----------
    adata : anndata.AnnData
        Container populated by :func:`tradeseq.fit_gam`.
    lineage : int, default 1
        1-based lineage id to construct the cascade for.
    genes : sequence of str/int or None
        Genes to inspect. ``None`` uses all genes in ``adata``.
    epsilon : float, default 1e-6
        Finite-difference distortion.
    n_points : int, default 100
        Number of grid points on the lineage.
    derivative_threshold : float, default 0.1
        Positive derivative threshold tested at every grid point.
    der_pval_threshold : float or None
        P-value threshold for calling a gene significantly increasing.
        ``None`` mirrors R's default ``0.05 / nPoints``.
    key : str, default "tradeseq"
        Namespace prefix used by :func:`tradeseq.fit_gam`.

    Returns
    -------
    CascadeResult
        Computed cascade result. Use :func:`plot_cascade` to draw the heatmap.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("cascade expects an anndata.AnnData object.")
    lineage = int(lineage)
    if lineage < 1:
        raise ValueError("'lineage' must be a positive 1-based integer.")
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("'n_points' must be at least 2.")
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("'epsilon' must be positive.")
    derivative_threshold = float(derivative_threshold)
    if der_pval_threshold is None:
        der_pval_threshold = 0.05 / n_points
    der_pval_threshold = float(der_pval_threshold)

    conditions = read_conditions(adata, key=key)
    dm = read_design_matrix(adata, key=key)
    n_curves = n_curves_from_dm(dm)
    if lineage > n_curves:
        raise ValueError(
            f"'lineage' must be <= number of lineages ({n_curves}); got {lineage}."
        )
    grid = get_predict_range_df(dm, lineage_id=lineage, n_points=n_points)
    lpmatrix = lpmatrix_dataframe(adata, key=key)
    pseudotime = pseudotime_matrix(adata, key=key)
    beta_all = read_beta(adata, key=key)
    sigma_all = read_sigma(adata, key=key)
    gene_idx, gene_labels = _resolve_genes(adata, genes)

    der_mat = np.full((len(gene_labels), n_points), np.nan, dtype=float)
    sd_mat = np.full_like(der_mat, np.nan)
    for row, idx in enumerate(gene_idx):
        der, sd_der = _calculate_derivative_one_gene(
            beta=np.asarray(beta_all[idx], dtype=float),
            sigma=np.asarray(sigma_all[idx], dtype=float),
            lpmatrix=lpmatrix,
            pseudotime=pseudotime,
            grid=grid,
            lineage=lineage,
            epsilon=epsilon,
            conditions=conditions,
        )
        der_mat[row, :] = der
        sd_mat[row, :] = sd_der

    time = grid[f"t{lineage}"].to_numpy(dtype=float)
    point_cols = [f"point{j}" for j in range(1, n_points + 1)]
    derivatives = pd.DataFrame(der_mat, index=gene_labels, columns=point_cols)
    sd_derivatives = pd.DataFrame(sd_mat, index=gene_labels, columns=point_cols)

    test_mat = (np.abs(der_mat) - derivative_threshold) / sd_mat
    pval_mat = np.minimum(1.0, norm.sf(test_mat))
    pval_mat[der_mat < derivative_threshold] = 1.0
    test_statistics = pd.DataFrame(test_mat, index=gene_labels, columns=point_cols)
    p_values = pd.DataFrame(pval_mat, index=gene_labels, columns=point_cols)

    finite_pval_mat = np.where(np.isfinite(pval_mat), pval_mat, np.inf)
    sig_mask = finite_pval_mat.min(axis=1) <= der_pval_threshold
    peak_genes = [g for g, keep in zip(gene_labels, sig_mask) if bool(keep)]
    first_peak = pd.Series(dtype=float, name="peakTime")
    if peak_genes:
        peak_values: list[float] = []
        for gene in peak_genes:
            row_idx = gene_labels.index(gene)
            pvals = finite_pval_mat[row_idx, :]
            min_idx = int(np.argmin(pvals))
            t_max = time[min_idx]
            der_after = der_mat[row_idx, time > t_max]
            time_after = time[time > t_max]
            crossing_pos = np.where(np.diff(np.sign(der_after)) == -2)[0]
            if crossing_pos.size == 0:
                peak_values.append(float(np.nanmax(time)))
            else:
                # R indexes ``t1[which(diff(sign(d1)) == -2)]``; ``which`` is
                # based on the shorter diff vector, so this selects the time
                # point immediately before the sign-crossing pair.
                peak_values.append(float(np.nanmin(time_after[crossing_pos])))
        first_peak = pd.Series(peak_values, index=peak_genes, name="peakTime")

    if peak_genes:
        yhat_all = predict_smooth(
            adata, gene=peak_genes, n_points=n_points, tidy=False, key=key
        )
        cols = _lineage_yhat_columns(list(yhat_all.columns), lineage)
        yhat = yhat_all.loc[peak_genes, cols].copy()
    else:
        yhat_all = predict_smooth(
            adata, gene=[gene_labels[0]], n_points=n_points, tidy=False, key=key
        )
        cols = _lineage_yhat_columns(list(yhat_all.columns), lineage)
        yhat = pd.DataFrame(index=pd.Index([], dtype=object), columns=cols, dtype=float)

    return CascadeResult(
        peak_time=first_peak,
        yhat=yhat,
        derivatives=derivatives,
        sd_derivatives=sd_derivatives,
        test_statistics=test_statistics,
        p_values=p_values,
        lineage=lineage,
        grid=grid,
    )


def _scale_rows(mat: pd.DataFrame) -> pd.DataFrame:
    """Return row-wise z-scores, mirroring R ``t(scale(t(mat)))``."""
    arr = mat.to_numpy(dtype=float)
    center = np.nanmean(arr, axis=1, keepdims=True)
    scale = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
    scaled = (arr - center) / scale
    return pd.DataFrame(scaled, index=mat.index, columns=mat.columns)


def _zissou1_palette(n: int = 12) -> list[str]:
    """Return the R ``wesanderson::wes_palette('Zissou1', type='continuous')`` ramp."""
    pal = scales.colour_ramp(_ZISSOU1_5_STOPS)
    return list(pal(np.linspace(0.0, 1.0, int(n))))


def _merge_annotation(
    auto: pd.DataFrame | None,
    user: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Merge auto annotations with user-supplied pheatmap annotations."""
    if auto is None:
        return user
    if user is None:
        return auto
    user = pd.DataFrame(user).reindex(auto.index)
    auto = auto.drop(columns=[c for c in auto.columns if c in user.columns])
    if auto.shape[1] == 0:
        return user
    return pd.concat([auto, user], axis=1)


def _cascade_row_annotation(
    result: CascadeResult, ordered_genes: Sequence[str]
) -> pd.DataFrame:
    """Peak pseudotime annotation aligned to plotted gene order."""
    return pd.DataFrame(
        {"peak_time": result.peak_time.reindex(ordered_genes).astype(float)},
        index=pd.Index(ordered_genes),
    )


def _cascade_col_annotation(result: CascadeResult) -> pd.DataFrame:
    """Pseudotime annotation aligned to plotted yhat columns."""
    time_col = f"t{result.lineage}"
    if time_col in result.grid:
        time_values = result.grid[time_col].to_numpy(dtype=float)
    else:
        time_values = np.arange(1, result.yhat.shape[1] + 1, dtype=float)

    records: list[dict[str, object]] = []
    has_condition = False
    for pos, col in enumerate(result.yhat.columns):
        col_str = str(col)
        match = _YHAT_CONDITION_POINT_RE.match(col_str) or _YHAT_POINT_RE.match(
            col_str
        )
        point = pos + 1
        condition: str | None = None
        if match is not None and int(match.group("lineage")) == result.lineage:
            point = int(match.group("point"))
            condition = match.groupdict().get("condition")
        if condition is not None:
            has_condition = True
        pseudotime = (
            time_values[point - 1] if 1 <= point <= len(time_values) else np.nan
        )
        records.append({"pseudotime": pseudotime, "condition": condition})

    annotation = pd.DataFrame(records, index=result.yhat.columns)
    if not has_condition:
        annotation = annotation.drop(columns=["condition"])
    return annotation


def plot_cascade(
    result: CascadeResult,
    *,
    show_gene_names: bool = True,
    show_peak_time: bool = True,
    show_time_points: bool = True,
    color: Sequence[str] | None = None,
    silent: bool = True,
    **kwargs,
) -> PHeatmap:
    """Draw the cascade heatmap from :func:`cascade` output.

    Parameters
    ----------
    result : CascadeResult
        Output from :func:`cascade`.
    show_gene_names : bool, default True
        Whether to show gene names as row labels. Set ``False`` for the sparse
        R visual default.
    show_peak_time : bool, default True
        Whether to add row annotation with each gene's peak pseudotime. Set
        ``False`` for the sparse R visual default.
    show_time_points : bool, default True
        Whether to add column annotation with pseudotime values. Time is shown
        as an annotation rather than dense column labels.
    color : sequence of str or None
        Heatmap palette. ``None`` uses the 12-colour continuous Zissou1 ramp
        used by the R source.
    silent : bool, default True
        Passed to :func:`pheatmap.pheatmap`; ``True`` builds and returns the
        heatmap object without drawing to the active grid device.
    **kwargs
        Additional keyword arguments forwarded to :func:`pheatmap.pheatmap`.

    Returns
    -------
    pheatmap.PHeatmap
        Heatmap object with ``gtable`` and optional row tree.

    Notes
    -----
    The Python defaults keep R's row ordering, column ordering, row scaling,
    border-free heatmap, and Zissou1 palette, but add labels/annotations so the
    heatmap carries more information. The sparse R visual default is:
    ``plot_cascade(result, show_gene_names=False, show_peak_time=False,
    show_time_points=False)``.
    """
    if not isinstance(result, CascadeResult):
        raise TypeError("plot_cascade expects a CascadeResult from cascade().")
    if result.yhat.empty:
        raise ValueError("Cannot plot an empty cascade result with no peak genes.")
    yhat_scaled = _scale_rows(result.yhat)
    ordered_genes = list(result.peak_time.sort_values(ascending=True).index)
    yhat_scaled = yhat_scaled.loc[ordered_genes, :]
    if color is None:
        color = _zissou1_palette(12)

    user_annotation_row = kwargs.pop("annotation_row", None)
    user_annotation_col = kwargs.pop("annotation_col", None)
    annotation_row = (
        _cascade_row_annotation(result, ordered_genes) if show_peak_time else None
    )
    annotation_col = _cascade_col_annotation(result) if show_time_points else None
    annotation_row = _merge_annotation(annotation_row, user_annotation_row)
    annotation_col = _merge_annotation(annotation_col, user_annotation_col)

    return pheatmap(
        yhat_scaled,
        color=list(color),
        cluster_cols=kwargs.pop("cluster_cols", False),
        cluster_rows=kwargs.pop("cluster_rows", False),
        border_color=kwargs.pop("border_color", None),
        show_rownames=kwargs.pop("show_rownames", show_gene_names),
        show_colnames=kwargs.pop("show_colnames", False),
        annotation_row=annotation_row,
        annotation_col=annotation_col,
        annotation_names_row=kwargs.pop("annotation_names_row", True),
        annotation_names_col=kwargs.pop("annotation_names_col", True),
        silent=silent,
        **kwargs,
    )
