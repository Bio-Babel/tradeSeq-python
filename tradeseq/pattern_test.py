"""Pattern / earlyDE test (R source: tradeSeq/R/patternTest.R + earlyDETest.R).

Implements the eigen-decomposition Wald battery for differential expression
between lineages along pseudotime. Both exports share a single implementation:
``patternTest`` is exactly ``earlyDETest(knots=NULL)`` (see the comment in
``patternTest.R:43-50``).

Exports
-------
:func:`pattern_test`
    Differential expression patterns across the full pseudotime range.
:func:`early_de_test`
    Same machinery, with an optional ``knots`` argument selecting a sub-range
    of pseudotime to contrast (``(k1, k2)`` is 1-indexed, matching R).
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._design import pattern_df, pattern_df_pairwise
from ._predict import predict_gam
from ._slot_io import (
    read_beta,
    read_conditions,
    read_design_matrix,
    read_knots,
    read_sigma,
)
from ._wald import get_eigen_stat_gam_fc

__all__ = ["pattern_test", "early_de_test"]


def _get_eigen_stat_gam(
    beta: np.ndarray, Sigma: np.ndarray, L: np.ndarray
) -> tuple[float, float]:
    """Port of ``getEigenStatGAM`` (utils.R:394-407).

    The non-FC sibling of :func:`get_eigen_stat_gam_fc`: no TREAT-style tube
    (``l2fc`` always 0), fixed eigen-value cutoff of ``1e-8``, and (crucially)
    does NOT return ``NaN`` for ``r == 1`` -- nor for the all-zero
    contrast-covariance case (``r == 0``). R's ``getEigenStatGAM`` has no
    sentinel at all; it errors out with ``seq_len(NA)`` when the top
    eigenvalue is exactly 0. The AnnData pairwise branch of ``earlyDETest``
    (earlyDETest.R:173) calls this helper, so to keep downstream Wald
    tables finite we mirror "R has no NaN sentinel" by returning
    ``(stat=0, r=0)`` for the degenerate case (the projection of ``L' beta``
    onto an empty rank-0 subspace is zero).
    """
    beta = np.asarray(beta, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(Sigma, dtype=float)
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    est = L.T @ beta
    sigma = L.T @ Sigma @ L
    sigma_sym = 0.5 * (sigma + sigma.T)
    eig_vals_asc, eig_vecs_asc = np.linalg.eigh(sigma_sym)
    eig_vals = eig_vals_asc[::-1]
    eig_vecs = eig_vecs_asc[:, ::-1]
    # All-zero / empty L'ΣL: in R, ``sum(NaN > 1e-8)`` is ``NA`` and the
    # downstream ``seq_len(NA)`` errors out. Replace that with a
    # rank-0 graceful return (the half-cov projection onto a zero-dim
    # subspace is zero).
    if eig_vals.size == 0 or eig_vals[0] == 0:
        return (0.0, 0.0)
    r = int(np.sum(eig_vals / eig_vals[0] > 1e-8))
    if r == 0:
        return (0.0, 0.0)
    V = eig_vecs[:, :r]
    D_inv_half = np.diag(1.0 / np.sqrt(eig_vals[:r]))
    half_cov_inv = V @ D_inv_half
    half_stat = est.T @ half_cov_inv
    stat = float((half_stat @ half_stat.T).reshape(()))
    return (stat, float(r))


def _lpmatrix_dataframe(adata: ad.AnnData, key: str) -> pd.DataFrame:
    """Reconstruct the lpmatrix as a DataFrame from the stored ndarray + columns.

    R stores the lpmatrix with named columns; the Python port
    splits these into ``uns[key]["lpmatrix"]`` (ndarray) and
    ``uns[key]["lpmatrix_columns"]`` (list of names).
    """
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns has no key {key!r}; call fit_gam first"
        )
    uns = adata.uns[key]
    if "lpmatrix" not in uns or "lpmatrix_columns" not in uns:
        raise KeyError(
            f"adata.uns[{key!r}] is missing 'lpmatrix' or 'lpmatrix_columns'; "
            "this slot is written by fit_gam"
        )
    lp_arr = np.asarray(uns["lpmatrix"])
    lp_cols = list(uns["lpmatrix_columns"])
    return pd.DataFrame(lp_arr, columns=lp_cols)


def _pseudotime_matrix(adata: ad.AnnData, pseudotime_key: str) -> np.ndarray:
    """Return the per-cell pseudotime matrix from ``adata.obsm[pseudotime_key]``."""
    if pseudotime_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {pseudotime_key!r}; populate it before "
            "pattern_test / early_de_test / condition_test"
        )
    pt = np.asarray(adata.obsm[pseudotime_key], dtype=np.float64)
    if pt.ndim == 1:
        pt = pt[:, None]
    return pt


def _count_lineages(dm_columns: Sequence[str]) -> int:
    """Number of distinct lineages in the design matrix (count ``t1, t2, ...``)."""
    import re

    rx = re.compile(r"t[1-9]")
    return sum(1 for c in dm_columns if rx.search(c) is not None)


def _build_global_L(
    dm: pd.DataFrame,
    lpmatrix_df: pd.DataFrame,
    pseudotime: np.ndarray,
    knots: Optional[tuple[int, int]],
    knot_points: np.ndarray,
    n_points: int,
    n_lineages: int,
    conditions: Optional[pd.Categorical],
) -> np.ndarray:
    """Build the omnibus contrast matrix L (shape ``(n_coefs, k)``).

    R reference: ``earlyDETest.R:63-94``. For each pair of lineages we form the
    block ``X_a - X_b`` (predicted lp-matrix at the n_points grid) and stack
    vertically. The final ``t(...)`` transposes to put contrasts in columns.
    """
    df_list = pattern_df(dm, knots=knots, knot_points=knot_points, n_points=n_points)
    X_per_lineage: list[np.ndarray] = []
    for jj in range(n_lineages):
        Xj = predict_gam(
            lpmatrix=lpmatrix_df,
            df=df_list[jj],
            pseudotime=pseudotime,
            conditions=conditions,
        )
        X_per_lineage.append(Xj.to_numpy())
    combs = list(combinations(range(1, n_lineages + 1), 2))
    blocks: list[np.ndarray] = []
    for (a, b) in combs:
        blocks.append(X_per_lineage[a - 1] - X_per_lineage[b - 1])
    L = np.vstack(blocks)  # shape (n_points * n_combs, n_coefs)
    return L.T  # shape (n_coefs, n_points * n_combs)


def _build_pairwise_L(
    dm: pd.DataFrame,
    lpmatrix_df: pd.DataFrame,
    pseudotime: np.ndarray,
    curves: tuple[int, int],
    knots: Optional[tuple[int, int]],
    knot_points: np.ndarray,
    n_points: int,
    conditions: Optional[pd.Categorical],
) -> np.ndarray:
    """Build the pairwise contrast matrix for one curve pair.

    R reference: ``earlyDETest.R:126-175``.
    """
    df_list = pattern_df_pairwise(
        dm,
        curves=list(curves),
        knots=knots,
        knot_points=knot_points,
        n_points=n_points,
    )
    X1 = predict_gam(
        lpmatrix=lpmatrix_df,
        df=df_list[0],
        pseudotime=pseudotime,
        conditions=conditions,
    ).to_numpy()
    X2 = predict_gam(
        lpmatrix=lpmatrix_df,
        df=df_list[1],
        pseudotime=pseudotime,
        conditions=conditions,
    ).to_numpy()
    return (X1 - X2).T  # shape (n_coefs, n_points)


def _wald_per_gene(
    beta_all: np.ndarray,
    sigma_all: np.ndarray,
    L: np.ndarray,
    l2fc: float,
    eigen_thresh: float,
    *,
    flavour: str = "fc",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the eigen-Wald to every gene; return (stat, df, pval).

    Two flavours mirroring the two R helpers:

    - ``"fc"`` -> ``getEigenStatGAMFC`` (with l2fc tube and configurable
      eigenThresh; returns NaN for r==1). Used by the global / AnnData
      non-pairwise branches.
    - ``"plain"`` -> ``getEigenStatGAM`` (no l2fc, fixed 1e-8 cutoff, no
      r==1 NaN sentinel). Used by the AnnData pairwise branch
      (``earlyDETest.R:173``).
    """
    n_genes = beta_all.shape[0]
    stats = np.full(n_genes, np.nan, dtype=np.float64)
    dfs = np.full(n_genes, np.nan, dtype=np.float64)
    for g in range(n_genes):
        beta_g = beta_all[g, :]
        sigma_g = sigma_all[g]
        if np.any(np.isnan(beta_g)) or np.any(np.isnan(sigma_g)):
            continue
        if flavour == "fc":
            stat, df = get_eigen_stat_gam_fc(
                beta_g, sigma_g, L, l2fc, eigen_thresh
            )
        elif flavour == "plain":
            stat, df = _get_eigen_stat_gam(beta_g, sigma_g, L)
        else:
            raise ValueError(f"Unknown flavour: {flavour!r}")
        stats[g] = stat
        dfs[g] = df
    # p-value = 1 - pchisq(stat, df). NaN inputs propagate.
    from scipy.stats import chi2

    pvals = np.full(n_genes, np.nan, dtype=np.float64)
    mask = ~np.isnan(stats) & ~np.isnan(dfs)
    if mask.any():
        pvals[mask] = chi2.sf(stats[mask], df=dfs[mask])
    return stats, dfs, pvals


def _fc_median_per_gene(beta_all: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Per-gene median of |contrast' β| across all columns of L.

    Mirrors ``.getFoldChanges`` (utils.R:426-428) followed by
    ``matrixStats::rowMedians(abs(...))`` (earlyDETest.R:197-203).
    """
    # FC = L' β per gene, shape (n_genes, ncol(L)).
    fc_all = beta_all @ L  # (n_genes, ncol(L))
    fc_median = np.median(np.abs(fc_all), axis=1)
    return fc_median


def _earlydetest_impl(
    adata: ad.AnnData,
    *,
    knots: Optional[Sequence[int]],
    global_: bool,
    pairwise: bool,
    l2fc: float,
    n_points: Optional[int],
    eigen_thresh: float,
    key: str,
    pseudotime_key: str,
) -> pd.DataFrame:
    """Shared implementation of ``earlyDETest`` / ``patternTest``.

    R reference: ``earlyDETest.R:4-211`` (the inner ``.earlyDETest``).
    """
    dm = read_design_matrix(adata, key=key)
    lpmatrix_df = _lpmatrix_dataframe(adata, key)
    knot_points = read_knots(adata, key=key)
    conditions = read_conditions(adata, key=key)
    pseudotime = _pseudotime_matrix(adata, pseudotime_key)
    beta_all = read_beta(adata, key=key)
    sigma_all = read_sigma(adata, key=key)

    n_knots = int(knot_points.shape[0])
    if n_points is None:
        n_points = 2 * n_knots
    n_lineages = _count_lineages(dm.columns)
    if n_lineages == 1:
        raise ValueError("You cannot run this test with only one lineage.")
    if n_lineages == 2 and pairwise:
        # R messages and silently disables pairwise; we mirror.
        import warnings

        warnings.warn(
            "Only two lineages; skipping pairwise comparison.", stacklevel=3
        )
        pairwise = False

    knots_tup: Optional[tuple[int, int]] = None
    if knots is not None:
        if len(knots) != 2:
            raise ValueError("knots must be a length-2 sequence")
        knots_tup = (int(knots[0]), int(knots[1]))

    var_index = adata.var_names

    # Omnibus L and Wald stats — `L` is reassigned in the pairwise loop below,
    # mirroring the R variable scoping at earlyDETest.R:80-94 / 144-166. The
    # fcMedian calculation at the end of the function uses *the final* value of
    # `L`, which is the global L when pairwise is off and the LAST pairwise L
    # when pairwise is on.
    L = _build_global_L(
        dm=dm,
        lpmatrix_df=lpmatrix_df,
        pseudotime=pseudotime,
        knots=knots_tup,
        knot_points=knot_points,
        n_points=n_points,
        n_lineages=n_lineages,
        conditions=conditions,
    )
    if global_:
        omnibus_stat, omnibus_df, omnibus_pval = _wald_per_gene(
            beta_all, sigma_all, L, l2fc, eigen_thresh
        )

    # Build columns in R order: [waldStat, df, pvalue], then pairwise blocks,
    # then fcMedian.
    columns: list[pd.Series] = []
    if global_:
        columns.append(pd.Series(omnibus_stat, index=var_index, name="waldStat"))
        columns.append(pd.Series(omnibus_df, index=var_index, name="df"))
        columns.append(pd.Series(omnibus_pval, index=var_index, name="pvalue"))

    if pairwise:
        combs = list(combinations(range(1, n_lineages + 1), 2))
        for (a, b) in combs:
            L = _build_pairwise_L(
                dm=dm,
                lpmatrix_df=lpmatrix_df,
                pseudotime=pseudotime,
                curves=(a, b),
                knots=knots_tup,
                knot_points=knot_points,
                n_points=n_points,
                conditions=conditions,
            )
            # R uses ``getEigenStatGAM`` (no FC, fixed 1e-8 thresh) in the AnnData
            # pairwise branch — earlyDETest.R:173.
            stat_p, df_p, pval_p = _wald_per_gene(
                beta_all,
                sigma_all,
                L,
                l2fc=0.0,
                eigen_thresh=1e-8,
                flavour="plain",
            )
            tag = f"{a}vs{b}"
            columns.append(pd.Series(stat_p, index=var_index, name=f"waldStat_{tag}"))
            columns.append(pd.Series(df_p, index=var_index, name=f"df_{tag}"))
            columns.append(pd.Series(pval_p, index=var_index, name=f"pvalue_{tag}"))

    # fcMedian uses the FINAL value of L — see comment above.
    fc_median = _fc_median_per_gene(beta_all, L)
    columns.append(pd.Series(fc_median, index=var_index, name="fcMedian"))
    return pd.concat(columns, axis=1)


def pattern_test(
    adata: ad.AnnData,
    *,
    global_: bool = True,
    pairwise: bool = False,
    l2fc: float = 0.0,
    n_points: Optional[int] = None,
    eigen_thresh: float = 1e-2,
    key: str = "tradeseq",
    pseudotime_key: str = "pseudotime",
) -> pd.DataFrame:
    """Test differential expression patterns between lineages.

    Port of ``tradeSeq::patternTest`` (R source: ``tradeSeq/R/patternTest.R``).
    Per the comment at ``patternTest.R:43-50``, ``patternTest`` is exactly
    ``earlyDETest(knots = NULL)`` — both names dispatch to the same impl.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`fit_gam` (mutable in-place).
    global_ : bool, default True
        If True, run the omnibus test across all pairwise lineage contrasts.
    pairwise : bool, default False
        If True, additionally return per-pair Wald columns. With only two
        lineages a warning is emitted and pairwise is silently disabled
        (R behaviour: ``earlyDETest.R:46-49``).
    l2fc : float, default 0.0
        log-2 fold-change threshold for the TREAT-style tube.
    n_points : int or None, default None
        Number of pseudotime grid points per lineage. ``None`` resolves to
        ``2 * nknots(adata)`` to mirror R's default.
    eigen_thresh : float, default 1e-2
        Relative eigenvalue cutoff for the eigen-decomposition Wald test.
    key : str, default "tradeseq"
        Namespace prefix in ``adata.uns`` / ``adata.varm`` / ``adata.var``.
    pseudotime_key : str, default "pseudotime"
        Key into ``adata.obsm`` for the per-cell pseudotime matrix.

    Returns
    -------
    pandas.DataFrame
        One row per gene (indexed by ``adata.var_names``) with at minimum
        ``waldStat``, ``df``, ``pvalue``, ``fcMedian``. When ``pairwise=True``
        and there are more than two lineages, additional
        ``waldStat_{a}vs{b}``, ``df_{a}vs{b}``, ``pvalue_{a}vs{b}`` columns are
        included for each pair.
    """
    return _earlydetest_impl(
        adata,
        knots=None,
        global_=global_,
        pairwise=pairwise,
        l2fc=l2fc,
        n_points=n_points,
        eigen_thresh=eigen_thresh,
        key=key,
        pseudotime_key=pseudotime_key,
    )


def early_de_test(
    adata: ad.AnnData,
    *,
    knots: Optional[Sequence[int]] = None,
    global_: bool = True,
    pairwise: bool = False,
    l2fc: float = 0.0,
    n_points: Optional[int] = None,
    eigen_thresh: float = 1e-2,
    key: str = "tradeseq",
    pseudotime_key: str = "pseudotime",
) -> pd.DataFrame:
    """Differential expression patterns in a user-defined knot region.

    Port of ``tradeSeq::earlyDETest`` (R source: ``tradeSeq/R/earlyDETest.R``).

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`fit_gam`.
    knots : sequence of int or None, default None
        1-indexed ``(k1, k2)`` pair specifying the knot range to test. When
        ``None``, the test covers the full pseudotime range and is equivalent
        to :func:`pattern_test`.
    global_, pairwise, l2fc, n_points, eigen_thresh, key, pseudotime_key
        See :func:`pattern_test`.

    Returns
    -------
    pandas.DataFrame
        See :func:`pattern_test`.
    """
    return _earlydetest_impl(
        adata,
        knots=knots,
        global_=global_,
        pairwise=pairwise,
        l2fc=l2fc,
        n_points=n_points,
        eigen_thresh=eigen_thresh,
        key=key,
        pseudotime_key=pseudotime_key,
    )
