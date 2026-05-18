"""Shared helpers for the Wald-test battery (R source: tradeSeq/R/utils.R + associationTest.R).

Three categories of helpers:

1.  :func:`get_fold_changes` — port of ``.getFoldChanges`` (utils.R:426-428).
2.  :func:`build_assoc_contrast_matrix` — extracts the per-lineage contrast block
    used by :func:`tradeseq.association_test`, mirroring the ``L1, L2, ...``
    construction inside ``.associationTest`` (associationTest.R:42-121).
3.  :func:`run_wald_table` — drives the per-gene Wald-test loop used by all
    three Tier-T2 exports.

All helpers operate on plain NumPy / pandas inputs so they can be tested
independently of the AnnData slot machinery.
"""

from __future__ import annotations

from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._design import (
    get_predict_custom_point_df,
    get_predict_end_point_df,
    get_predict_start_point_df,
)
from ._predict import predict_gam
from ._slot_io import (
    _require_uns,
    read_beta,
    read_conditions,
    read_design_matrix,
    read_sigma,
)
from ._wald import wald_test_fc

__all__ = [
    "get_fold_changes",
    "mean_log_fc",
    "build_assoc_per_lineage_contrast",
    "build_sve_contrast",
    "build_diffend_pair_contrasts",
    "lpmatrix_dataframe",
    "pseudotime_matrix",
    "max_pseudotime_per_lineage",
    "n_curves_from_dm",
    "run_wald_table",
    "fetch_test_arrays",
    "INVERSE_DEFAULT_L2FC0",
    "INVERSE_DEFAULT_L2FC_NONZERO",
]


INVERSE_DEFAULT_L2FC0 = "Chol"
"""Default ``inverse`` strategy used by ``associationTest`` when ``l2fc == 0``.

R source: ``associationTest.R:483`` — ``ifelse(l2fc==0, "Chol", "eigen")``.
"""

INVERSE_DEFAULT_L2FC_NONZERO = "eigen"
"""Default ``inverse`` strategy used by ``associationTest`` when ``l2fc != 0``."""


# ---------------------------------------------------------------------------
# Fold-change helpers
# ---------------------------------------------------------------------------


def get_fold_changes(beta: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return the per-contrast fold change ``Lᵀ β`` for a single gene.

    Port of ``.getFoldChanges`` (utils.R:426-428):
    ``apply(L, 2, function(contrast) contrast %*% beta)``.

    Parameters
    ----------
    beta : numpy.ndarray
        Coefficient vector of length ``p``.
    L : numpy.ndarray
        ``(p, k)`` contrast matrix.

    Returns
    -------
    numpy.ndarray
        Length-``k`` vector of contrast estimates.
    """
    beta = np.asarray(beta, dtype=float).reshape(-1)
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    return L.T @ beta


def mean_log_fc(beta_all: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return per-gene mean absolute log fold change, mirroring R's pipeline.

    Implements the R idiom of ``associationTest.R:170-185``:

    * For each gene with no NaN coefficient, build ``fc_g = Lᵀ β_g``.
    * Stack across genes into ``(n_genes, k)``.
    * Reduce to a length-``n_genes`` vector of ``rowMeans(abs(.))``.
    * Genes with any NaN coefficient receive ``NaN``.

    Parameters
    ----------
    beta_all : numpy.ndarray
        ``(n_genes, p)`` coefficient matrix.
    L : numpy.ndarray
        ``(p, k)`` contrast matrix.

    Returns
    -------
    numpy.ndarray
        Length-``n_genes`` mean |log FC| vector.
    """
    beta_all = np.asarray(beta_all, dtype=float)
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    n_genes = beta_all.shape[0]
    # (n_genes, k) — vectorised "Lᵀ β" across genes.
    fc = beta_all @ L  # (n_genes, p) @ (p, k) = (n_genes, k)
    # Mask NaN rows -> NaN mean.
    nan_mask = np.isnan(beta_all).any(axis=1)
    out = np.mean(np.abs(fc), axis=1)
    out[nan_mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Slot-extraction helpers
# ---------------------------------------------------------------------------


def lpmatrix_dataframe(adata: ad.AnnData, *, key: str = "tradeseq") -> pd.DataFrame:
    """Return the lpmatrix wrapped as a DataFrame with its R-faithful column names.

    The Python ``write_fit_results`` stores the lpmatrix as a 2-D ``ndarray``
    plus a separate ``uns[key]["lpmatrix_columns"]`` list (see ``fit_gam.py:408``).
    This helper reattaches them so the matrix can be fed to :func:`predict_gam`.
    """
    uns = _require_uns(adata, key)
    if "lpmatrix_columns" not in uns:
        raise KeyError(
            f"adata.uns[{key!r}] missing 'lpmatrix_columns'; "
            "fit_gam must be re-run with the slice0 build."
        )
    return pd.DataFrame(
        np.asarray(uns["lpmatrix"], dtype=float),
        columns=list(uns["lpmatrix_columns"]),
    )


def pseudotime_matrix(adata: ad.AnnData, *, key: str = "tradeseq") -> np.ndarray:
    """Return the ``(n_cells, n_lineages)`` pseudotime matrix.

    R's ``.associationTest`` reads pseudotime via:

    .. code-block:: R

        slingshotColData <- colData(models)$crv
        pseudotime <- slingshotColData[, grep("pseudotime")]

    The Python container puts the same matrix at ``adata.obsm["pseudotime"]``;
    we mirror the R idiom here.
    """
    if "pseudotime" not in adata.obsm:
        raise KeyError(
            "adata.obsm has no key 'pseudotime'; this is required by the Wald tests."
        )
    arr = np.asarray(adata.obsm["pseudotime"], dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def n_curves_from_dm(dm: pd.DataFrame) -> int:
    """Return the number of trajectory lineages, mirroring R's ``grep('t[1-9]')``."""
    import re

    rx = re.compile(r"t[1-9]")
    return sum(1 for c in dm.columns if rx.search(c) is not None)


def max_pseudotime_per_lineage(dm: pd.DataFrame, n_curves: int) -> np.ndarray:
    """Return the per-lineage maximum pseudotime, mirroring ``associationTest.R:33-36``.

    R idiom:

    .. code-block:: R

        sapply(seq_len(nCurves), function(cc){
          max(dm[, paste0("t", cc)][dm[, paste0("l", cc)] == 1])
        })

    Note this uses the *single* ``l<cc>`` column (no condition split), so the
    helper is correct only for the non-conditions branch — which is the only
    branch this slice is responsible for.
    """
    out = np.zeros(n_curves, dtype=float)
    for cc in range(1, n_curves + 1):
        t_col = dm[f"t{cc}"].to_numpy(dtype=float)
        l_col = dm[f"l{cc}"].to_numpy(dtype=float)
        out[cc - 1] = float(t_col[l_col == 1].max())
    return out


# ---------------------------------------------------------------------------
# Contrast-matrix builders
# ---------------------------------------------------------------------------


def build_assoc_per_lineage_contrast(
    lpmatrix: pd.DataFrame,
    dm: pd.DataFrame,
    pseudotime: np.ndarray,
    conditions: Optional[pd.Categorical],
    lineage_id: int,
    max_t: float,
    n_points: int,
    contrast_type: str,
) -> np.ndarray:
    """Build one lineage's contrast block ``L_jj`` for :func:`association_test`.

    Mirrors ``associationTest.R:79-119`` for ``nCurves > 1`` (and ``42-76`` for the
    single-curve case — the matrix construction is identical, only the variable
    name changes).

    Parameters
    ----------
    lpmatrix : pandas.DataFrame
        Linear-predictor matrix of the fitted GAM, with R-faithful column names.
    dm : pandas.DataFrame
        Per-cell design matrix.
    pseudotime : numpy.ndarray
        ``(n_cells, n_lineages)`` per-cell pseudotime matrix.
    conditions : pandas.Categorical or None
        Per-cell condition labels, or ``None`` if no conditions were fitted.
    lineage_id : int
        1-based lineage index.
    max_t : float
        Maximum observed pseudotime on this lineage.
    n_points : int
        Number of contrast pseudotime points (``nPoints`` in R).
    contrast_type : {"start", "end", "consecutive"}
        Which difference pattern to use.

    Returns
    -------
    numpy.ndarray
        ``(p, n_points - 1)`` contrast matrix where ``p`` is the number of GAM
        coefficients.
    """
    n_coefs = lpmatrix.shape[1]
    contrast_points = np.linspace(0.0, max_t, n_points)
    # Stack n_points custom-point df rows for predict_gam.
    df_points = pd.concat(
        [
            get_predict_custom_point_df(dm, lineage_id, float(tp))
            for tp in contrast_points
        ],
        axis=0,
        ignore_index=True,
    )
    x_points = predict_gam(
        lpmatrix=lpmatrix,
        df=df_points,
        pseudotime=pseudotime,
        conditions=conditions,
    ).to_numpy()  # (n_points, p)

    C = np.zeros((n_coefs, n_points - 1), dtype=float)
    if contrast_type == "start":
        for pp in range(1, n_points):  # R loop: seq_len(nPoints)[-1] = 2..nPoints
            # column index is pp - 1 (R: L[,pp-1] <- XPoints[pp,] - XPoints[1,])
            C[:, pp - 1] = x_points[pp, :] - x_points[0, :]
    elif contrast_type == "end":
        for pp in range(1, n_points):  # R: seq_len(nPoints)[-nPoints] = 1..(nPoints-1)
            # R indices: pp goes from 1 to nPoints-1 → Python idx pp-1
            # but R fills C[, pp] = XPoints[pp,] - XPoints[nPoints,]
            # → 0-based: C[:, pp-1] = x_points[pp-1, :] - x_points[-1, :]
            C[:, pp - 1] = x_points[pp - 1, :] - x_points[-1, :]
    elif contrast_type == "consecutive":
        for pp in range(1, n_points):
            # R: C[, pp] = XPoints[pp+1,] - XPoints[pp,]
            # → 0-based: C[:, pp-1] = x_points[pp, :] - x_points[pp-1, :]
            C[:, pp - 1] = x_points[pp, :] - x_points[pp - 1, :]
    else:
        raise ValueError(
            f"contrast_type must be one of 'start', 'end', 'consecutive'; "
            f"got {contrast_type!r}"
        )
    return C


def build_sve_contrast(
    lpmatrix: pd.DataFrame,
    dm: pd.DataFrame,
    pseudotime: np.ndarray,
    conditions: Optional[pd.Categorical],
    n_lineages: int,
    pseudotime_values: Optional[Sequence[float]],
) -> np.ndarray:
    """Build the start-vs-end contrast matrix ``L`` (startVsEndTest.R:66-103).

    For each lineage ``jj``:

    * If ``pseudotime_values is None`` → ``L[, jj] = XEnd - XStart`` with the
      lineage's natural endpoints.
    * Else → ``L[, jj] = XPoint(t1) - XPoint(t0)`` where ``(t0, t1) = pseudotime_values``.

    Parameters
    ----------
    lpmatrix : pandas.DataFrame
        Linear-predictor matrix.
    dm : pandas.DataFrame
        Per-cell design matrix.
    pseudotime : numpy.ndarray
        ``(n_cells, n_lineages)`` per-cell pseudotime matrix.
    conditions : pandas.Categorical or None
        Per-cell condition labels.
    n_lineages : int
        Number of lineages.
    pseudotime_values : sequence of float or None
        Optional ``(t0, t1)`` pair to use instead of natural start/end.

    Returns
    -------
    numpy.ndarray
        ``(p, n_lineages)`` contrast matrix.
    """
    if pseudotime_values is not None:
        max_t = float(pseudotime.max())
        if any(pv > max_t for pv in pseudotime_values):
            raise ValueError(
                "Pseudotime values provided are larger than the maximum "
                "pseudotime in the trajectory."
            )

    n_coefs = lpmatrix.shape[1]
    L = np.zeros((n_coefs, n_lineages), dtype=float)
    for jj in range(1, n_lineages + 1):
        if pseudotime_values is None:
            df_end = get_predict_end_point_df(dm, jj)
            df_start = get_predict_start_point_df(dm, jj)
        else:
            df_end = get_predict_custom_point_df(
                dm, jj, pseudotime=float(pseudotime_values[1])
            )
            df_start = get_predict_custom_point_df(
                dm, jj, pseudotime=float(pseudotime_values[0])
            )
        x_end = predict_gam(
            lpmatrix=lpmatrix,
            df=df_end,
            pseudotime=pseudotime,
            conditions=conditions,
        ).to_numpy().reshape(-1)
        x_start = predict_gam(
            lpmatrix=lpmatrix,
            df=df_start,
            pseudotime=pseudotime,
            conditions=conditions,
        ).to_numpy().reshape(-1)
        L[:, jj - 1] = x_end - x_start
    return L


def build_diffend_pair_contrasts(
    lpmatrix: pd.DataFrame,
    dm: pd.DataFrame,
    pseudotime: np.ndarray,
    conditions: Optional[pd.Categorical],
    n_lineages: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build pairwise end-point contrasts for :func:`diff_end_test`.

    Mirrors ``diffEndTest.R:43-72``.

    Returns
    -------
    L : numpy.ndarray
        ``(p, n_pairs)`` contrast matrix.
    pairs : list of tuple
        ``utils::combn(nLineages, 2)`` pairs as 1-based ``(i, j)`` tuples,
        column-aligned with ``L``.
    """
    # Build end-point lpmatrix rows for every lineage.
    x_end = np.zeros((n_lineages, lpmatrix.shape[1]), dtype=float)
    for jj in range(1, n_lineages + 1):
        df = get_predict_end_point_df(dm, jj)
        x_end[jj - 1, :] = (
            predict_gam(
                lpmatrix=lpmatrix,
                df=df,
                pseudotime=pseudotime,
                conditions=conditions,
            )
            .to_numpy()
            .reshape(-1)
        )

    # R's combn(n, 2) emits column-major all-pair indices.
    pairs: list[tuple[int, int]] = []
    for i in range(1, n_lineages + 1):
        for j in range(i + 1, n_lineages + 1):
            pairs.append((i, j))
    n_pairs = len(pairs)
    L = np.zeros((lpmatrix.shape[1], n_pairs), dtype=float)
    for col, (i, j) in enumerate(pairs):
        L[:, col] = x_end[i - 1, :] - x_end[j - 1, :]
    return L, pairs


# ---------------------------------------------------------------------------
# Wald-test driver
# ---------------------------------------------------------------------------


def run_wald_table(
    beta_all: np.ndarray,
    sigma_all: np.ndarray,
    L: np.ndarray,
    l2fc: float,
    inverse: str,
    eigen_thresh: float = 1e-2,
) -> np.ndarray:
    """Run :func:`wald_test_fc` for every gene against the contrast ``L``.

    Mirrors the per-gene ``lapply`` in ``associationTest.R:128-138`` (and the
    sibling SVE / DiffEnd drivers): genes with any NaN coefficient receive
    ``(NaN, NaN, NaN)``.

    Parameters
    ----------
    beta_all : numpy.ndarray
        ``(n_genes, p)`` coefficient matrix.
    sigma_all : numpy.ndarray
        Length-``n_genes`` object array of ``(p, p)`` covariance matrices.
    L : numpy.ndarray
        ``(p, k)`` contrast matrix.
    l2fc : float
        TREAT-style log2 fold-change threshold.
    inverse : str
        Inverse strategy for :func:`wald_test_fc`.
    eigen_thresh : float, default 1e-2
        Relative eigenvalue cutoff forwarded to the eigen inverse branch.

    Returns
    -------
    numpy.ndarray
        ``(n_genes, 3)`` matrix of ``(stat, df, pvalue)``.
    """
    n_genes = beta_all.shape[0]
    out = np.empty((n_genes, 3), dtype=float)
    for ii in range(n_genes):
        beta = beta_all[ii, :]
        if np.isnan(beta).any():
            out[ii, :] = np.nan
            continue
        sigma = sigma_all[ii]
        stat, df, pval = wald_test_fc(
            beta,
            sigma,
            L,
            l2fc=l2fc,
            inverse=inverse,
            eigen_thresh=eigen_thresh,
        )
        out[ii, 0] = stat
        out[ii, 1] = df
        out[ii, 2] = pval
    return out


def fetch_test_arrays(
    adata: ad.AnnData, key: str
) -> tuple[
    np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, Optional[pd.Categorical]
]:
    """One-stop slot reader returning every input needed by the three exports."""
    beta_all = read_beta(adata, key=key)
    sigma_all = read_sigma(adata, key=key)
    dm = read_design_matrix(adata, key=key)
    lpmatrix = lpmatrix_dataframe(adata, key=key)
    pseudotime = pseudotime_matrix(adata, key=key)
    conditions = read_conditions(adata, key=key)
    return beta_all, sigma_all, dm, lpmatrix, pseudotime, conditions
