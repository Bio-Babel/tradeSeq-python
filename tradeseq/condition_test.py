"""Condition test (R source: tradeSeq/R/conditionTest.R).

Implements the eigen-decomposition Wald battery for differential expression
between conditions within a lineage. Requires that :func:`fit_gam` was called
with a non-None ``conditions_key``.

Exports
-------
:func:`condition_test`
    Wald test for between-condition DE.
"""

from __future__ import annotations

import re
import warnings
from itertools import combinations
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._slot_io import (
    read_beta,
    read_conditions,
    read_design_matrix,
    read_knots,
    read_sigma,
)
from ._wald import get_eigen_stat_gam_fc

__all__ = ["condition_test"]


def _lpmatrix_columns(adata: ad.AnnData, key: str) -> list[str]:
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns has no key {key!r}; call fit_gam first"
        )
    uns = adata.uns[key]
    if "lpmatrix_columns" not in uns:
        raise KeyError(
            f"adata.uns[{key!r}] is missing 'lpmatrix_columns'; "
            "this slot is written by fit_gam"
        )
    return list(uns["lpmatrix_columns"])


def _nknots(adata: ad.AnnData, key: str) -> int:
    """Return the number of knots from the stored knot vector."""
    knot_points = read_knots(adata, key=key)
    return int(knot_points.shape[0])


def _find_conditions(
    adata: ad.AnnData, key: str
) -> tuple[pd.Categorical, int]:
    """R reference: ``conditionTest.R:2-16`` (`.find_conditions`).

    Raises if conditions are absent or have a single level.
    """
    conditions = read_conditions(adata, key=key)
    if conditions is None:
        raise ValueError("The models were not fitted with multiple conditions.")
    n_conditions = len(conditions.categories)
    if n_conditions == 1:
        raise ValueError("This can only be run with multiple conditions.")
    return conditions, n_conditions


def _count_curves_in_dm(dm_columns: Sequence[str]) -> int:
    """Number of ``l[(1-9)+]`` columns in the design matrix.

    R uses ``grep(pattern = "l[(1-9)+]")`` at ``conditionTest.R:84``. The
    character class ``[(1-9)+]`` matches '(', ')', '+', '1'..'9'. In practice,
    column names are like ``l1_1``/``l1_2``/.../``l2_K``, so the match
    coincides with R's behavior — we count any column starting with ``l``
    followed by a digit.
    """
    rx = re.compile(r"l[(1-9)+]")
    return sum(1 for c in dm_columns if rx.search(c) is not None)


def _construct_contrast_matrix_conditions(
    lpmatrix_columns: Sequence[str],
    n_conditions: int,
    n_lineages: int,
    n_knots_total: int,
    n_knots_block: int,
    knots: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    """Port of ``.construct_contrast_matrix_conditions`` (conditionTest.R:18-59).

    Parameters
    ----------
    lpmatrix_columns
        Column names of the GAM lpmatrix (i.e. ``colnames(X)``).
    n_conditions
        Number of condition levels.
    n_lineages
        Number of lineages.
    n_knots_total
        Total number of knots used by ``fit_gam`` (R's ``nknots(models)``).
    n_knots_block
        Number of knots in the range selected by ``knots`` (R's ``nKnots``).
    knots
        1-indexed ``(k1, k2)`` knot range.

    Returns
    -------
    L : np.ndarray
        Contrast matrix of shape ``(len(lpmatrix_columns), n_lineages * n_knots_block * nComparisonsPerCurve)``.
    colnames : list[str]
        Column labels, one of ``"lineage1"``, ``"lineage2"``, ... repeated
        ``n_knots_block * nComparisonsPerCurve`` times each. Used for the
        ``L[, colnames(L) == "lineageX"]`` slicing in the ``lineages`` branch.
    """
    combs_per_curve = list(combinations(range(1, n_conditions + 1), 2))
    n_comparisons_per_curve = len(combs_per_curve)
    # Lsub: nrow = nknots(models)*nConditions, ncol = nKnots*nComparisonsPerCurve
    Lsub = np.zeros(
        (n_knots_total * n_conditions, n_knots_block * n_comparisons_per_curve),
        dtype=np.float64,
    )
    for jj, (c1, c2) in enumerate(combs_per_curve):
        # R 1-based indices: comp1ID = ((c1-1)*nknots+knots[1]):((c1-1)*nknots+knots[2])
        comp1_lo = (c1 - 1) * n_knots_total + knots[0]
        comp1_hi = (c1 - 1) * n_knots_total + knots[1]
        comp2_lo = (c2 - 1) * n_knots_total + knots[0]
        comp2_hi = (c2 - 1) * n_knots_total + knots[1]
        comp1_ids = list(range(comp1_lo, comp1_hi + 1))
        comp2_ids = list(range(comp2_lo, comp2_hi + 1))
        for kk in range(len(comp1_ids)):
            row1 = comp1_ids[kk] - 1  # to 0-indexed
            row2 = comp2_ids[kk] - 1
            col = jj * n_knots_block + kk  # 0-indexed
            Lsub[row1, col] = 1.0
            Lsub[row2, col] = -1.0

    # L: nrow = ncol(X), ncol = nLineages * nKnots * nComparisonsPerCurve
    n_coefs = len(lpmatrix_columns)
    L = np.zeros(
        (n_coefs, n_lineages * n_knots_block * n_comparisons_per_curve),
        dtype=np.float64,
    )

    # Identify smooth coefficients. R uses ``grep("^s(t[1-9]+)*", colnames(X))``
    # — `(t[1-9]+)*` is a *zero-or-more* regex group, so the practical effect is
    # "starts with 's'" (any smooth s(...) column).
    smooth_rx = re.compile(r"^s")
    smooth_idx = [i for i, c in enumerate(lpmatrix_columns) if smooth_rx.match(c)]
    smooth_names = [lpmatrix_columns[i] for i in smooth_idx]
    # textAfterSplit: substring after ":l", first char identifies lineage.
    # e.g. ``s(t1):l1_1.1`` -> ``1_1.1`` -> first char "1"
    text_after = [name.split(":l", 1)[1] for name in smooth_names]

    for jj in range(1, n_lineages + 1):
        # curve mask: first char of text_after equals jj (string compare)
        curv_mask = [t[0] == str(jj) for t in text_after]
        # limits: ((jj-1)*nKnots*nComparisonsPerCurve + 1) : (jj*nKnots*nComparisonsPerCurve)
        col_lo = (jj - 1) * n_knots_block * n_comparisons_per_curve  # 0-indexed start
        col_hi = jj * n_knots_block * n_comparisons_per_curve  # 0-indexed exclusive end
        rows = [smooth_idx[i] for i, m in enumerate(curv_mask) if m]
        L[np.ix_(rows, np.arange(col_lo, col_hi))] = Lsub

    colnames = []
    for jj in range(1, n_lineages + 1):
        colnames.extend(
            [f"lineage{jj}"] * (n_knots_block * n_comparisons_per_curve)
        )
    return L, colnames


def _wald_per_gene(
    beta_all: np.ndarray,
    sigma_all: np.ndarray,
    L: np.ndarray,
    l2fc: float,
    eigen_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply ``get_eigen_stat_gam_fc`` to every gene; mirrors ``.allWaldStatGAMFC``."""
    n_genes = beta_all.shape[0]
    stats = np.full(n_genes, np.nan, dtype=np.float64)
    dfs = np.full(n_genes, np.nan, dtype=np.float64)
    for g in range(n_genes):
        beta_g = beta_all[g, :]
        sigma_g = sigma_all[g]
        if np.any(np.isnan(beta_g)) or np.any(np.isnan(sigma_g)):
            continue
        stat, df = get_eigen_stat_gam_fc(beta_g, sigma_g, L, l2fc, eigen_thresh)
        stats[g] = stat
        dfs[g] = df
    from scipy.stats import chi2

    pvals = np.full(n_genes, np.nan, dtype=np.float64)
    mask = ~np.isnan(stats) & ~np.isnan(dfs)
    if mask.any():
        pvals[mask] = chi2.sf(stats[mask], df=dfs[mask])
    return stats, dfs, pvals


def _filter_L_for_condition_pair(
    L: np.ndarray,
    lpmatrix_columns: Sequence[str],
    conds: tuple[int, int],
) -> np.ndarray:
    """Apply the R recipe to extract the L sub-matrix for one pair of conditions.

    R reference: ``conditionTest.R:120-130`` and ``143-155``::

        conds_L <- c(
          grep(rownames(L), pattern = paste0("_", conds[1], "\\.")),
          grep(rownames(L), pattern = paste0("_", conds[2], "\\."))
        )
        Lpair <- L
        Lpair[-conds_L, ] <- 0
        Lpair <- Lpair[, (colSums(Lpair) == 0) & (colSums(abs(Lpair)) != 0)]

    Note the final selection is ``colSums == 0 & colSums(abs) != 0``. The first
    condition retains only contrasts that balance (the +1/-1 are cancelled when
    one side is zeroed) — but R's bug-or-feature: it selects columns where the
    *signed* column sum vanishes and the *absolute* column sum is non-zero
    after zeroing out unrelated rows.
    """
    c1, c2 = conds
    pat1 = re.compile(rf"_{c1}\.")
    pat2 = re.compile(rf"_{c2}\.")
    keep_rows = set()
    for i, name in enumerate(lpmatrix_columns):
        if pat1.search(name) or pat2.search(name):
            keep_rows.add(i)
    Lpair = L.copy()
    drop_rows = [i for i in range(L.shape[0]) if i not in keep_rows]
    Lpair[drop_rows, :] = 0.0
    col_sum = Lpair.sum(axis=0)
    col_abs_sum = np.abs(Lpair).sum(axis=0)
    keep_cols = (col_sum == 0) & (col_abs_sum != 0)
    return Lpair[:, keep_cols]


def condition_test(
    adata: ad.AnnData,
    *,
    global_: bool = True,
    pairwise: bool = False,
    lineages: bool = False,
    l2fc: float = 0.0,
    eigen_thresh: float = 1e-2,
    knots: Optional[Sequence[int]] = None,
    key: str = "tradeseq",
) -> pd.DataFrame:
    """Test differential expression between conditions within a lineage.

    Port of ``tradeSeq::conditionTest`` (R source:
    ``tradeSeq/R/conditionTest.R``). Requires that :func:`fit_gam` was called
    with ``conditions_key`` non-None.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`fit_gam` with conditions.
    global_ : bool, default True
        Run the omnibus test across all (lineage, condition pair) contrasts.
    pairwise : bool, default False
        Return per condition-pair Wald columns. With only two conditions a
        warning is emitted and pairwise is silently disabled.
    lineages : bool, default False
        Return per-lineage Wald columns.
    l2fc : float, default 0.0
        log-2 fold-change threshold for the TREAT-style tube.
    eigen_thresh : float, default 1e-2
        Relative eigenvalue cutoff.
    knots : sequence of int or None, default None
        Optional 1-indexed ``(k1, k2)`` knot range. ``None`` covers ``(1, nKnots)``.
    key : str, default "tradeseq"
        Namespace prefix in ``adata.uns``.

    Returns
    -------
    pandas.DataFrame
        Per-gene table. Columns depend on the ``global_/pairwise/lineages``
        flags exactly as in R::

        - ``global=True``: ``waldStat, df, pvalue``
        - ``pairwise=True`` only: ``waldStat_condsAvsB, df_condsAvsB, pvalue_condsAvsB``
          for each pair.
        - ``lineages=True`` only: ``waldStat_lineageL, df_lineageL, pvalue_lineageL``
          for each lineage.
        - ``lineages=True & pairwise=True``: combined ``..._lineageL_condsAvsB``.
    """
    if not (global_ or pairwise or lineages):
        raise ValueError("One of global, pairwise or lineages must be true")

    conditions, n_conditions = _find_conditions(adata, key)
    dm = read_design_matrix(adata, key=key)
    lpmatrix_columns = _lpmatrix_columns(adata, key)
    n_knots_total = _nknots(adata, key)

    if knots is None:
        knots_tup: tuple[int, int] = (1, n_knots_total)
        n_knots_block = n_knots_total
    else:
        if (
            len(knots) != 2
            or not all(isinstance(k, (int, np.integer)) for k in knots)
            or any(k > n_knots_total for k in knots)
        ):
            raise ValueError(
                "knots must consists of 2 integers below the number of knots"
            )
        knots_tup = (int(knots[0]), int(knots[1]))
        n_knots_block = knots_tup[1] - knots_tup[0] + 1

    n_curves = _count_curves_in_dm(dm.columns)
    n_lineages = n_curves // n_conditions
    if n_lineages == 1 and lineages:
        warnings.warn(
            "Only one lineage; skipping single-lineage comparison.", stacklevel=2
        )
        lineages = False
    if n_conditions == 2 and pairwise:
        warnings.warn(
            "Only two conditions; skipping pairwise comparison.", stacklevel=2
        )
        pairwise = False

    L, L_colnames = _construct_contrast_matrix_conditions(
        lpmatrix_columns=lpmatrix_columns,
        n_conditions=n_conditions,
        n_lineages=n_lineages,
        n_knots_total=n_knots_total,
        n_knots_block=n_knots_block,
        knots=knots_tup,
    )

    beta_all = read_beta(adata, key=key)
    sigma_all = read_sigma(adata, key=key)
    var_index = adata.var_names

    # Omnibus
    om_stat, om_df, om_pval = _wald_per_gene(
        beta_all, sigma_all, L, l2fc, eigen_thresh
    )
    omnibus = pd.DataFrame(
        {"waldStat": om_stat, "df": om_df, "pvalue": om_pval},
        index=var_index,
    )

    # Per-lineage
    if lineages and not pairwise:
        per_lin: list[pd.DataFrame] = []
        for jj in range(1, n_lineages + 1):
            tag = f"lineage{jj}"
            keep = [i for i, c in enumerate(L_colnames) if c == tag]
            LLin = L[:, keep]
            stat_l, df_l, pval_l = _wald_per_gene(
                beta_all, sigma_all, LLin, l2fc, eigen_thresh
            )
            per_lin.append(
                pd.DataFrame(
                    {
                        f"waldStat_lineage{jj}": stat_l,
                        f"df_lineage{jj}": df_l,
                        f"pvalue_lineage{jj}": pval_l,
                    },
                    index=var_index,
                )
            )
        per_lin_df = pd.concat(per_lin, axis=1)

    # Per condition-pair across all lineages
    if pairwise and not lineages:
        per_pair: list[pd.DataFrame] = []
        for (c1, c2) in combinations(range(1, n_conditions + 1), 2):
            Lpair = _filter_L_for_condition_pair(
                L, lpmatrix_columns, (c1, c2)
            )
            stat_p, df_p, pval_p = _wald_per_gene(
                beta_all, sigma_all, Lpair, l2fc, eigen_thresh
            )
            tag = f"conds{c1}vs{c2}"
            per_pair.append(
                pd.DataFrame(
                    {
                        f"waldStat_{tag}": stat_p,
                        f"df_{tag}": df_p,
                        f"pvalue_{tag}": pval_p,
                    },
                    index=var_index,
                )
            )
        per_pair_df = pd.concat(per_pair, axis=1)

    # Per-lineage × per condition-pair
    if lineages and pairwise:
        per_all: list[pd.DataFrame] = []
        for ll in range(1, n_lineages + 1):
            tag_l = f"lineage{ll}"
            keep = [i for i, c in enumerate(L_colnames) if c == tag_l]
            LLin = L[:, keep]
            for (c1, c2) in combinations(range(1, n_conditions + 1), 2):
                Lpair = _filter_L_for_condition_pair(
                    LLin, lpmatrix_columns, (c1, c2)
                )
                stat_p, df_p, pval_p = _wald_per_gene(
                    beta_all, sigma_all, Lpair, l2fc, eigen_thresh
                )
                tag = f"lineage{ll}_conds{c1}vs{c2}"
                per_all.append(
                    pd.DataFrame(
                        {
                            f"waldStat_{tag}": stat_p,
                            f"df_{tag}": df_p,
                            f"pvalue_{tag}": pval_p,
                        },
                        index=var_index,
                    )
                )
        per_all_df = pd.concat(per_all, axis=1)

    # Return per the same combinations R uses (conditionTest.R:172-186).
    if global_ and not lineages and not pairwise:
        return omnibus
    if not global_ and lineages and not pairwise:
        return per_lin_df
    if not global_ and not lineages and pairwise:
        return per_pair_df
    if global_ and lineages and not pairwise:
        return pd.concat([omnibus, per_lin_df], axis=1)
    if global_ and not lineages and pairwise:
        return pd.concat([omnibus, per_pair_df], axis=1)
    if not global_ and lineages and pairwise:
        return per_all_df
    if global_ and lineages and pairwise:
        return pd.concat([omnibus, per_all_df], axis=1)
    # Should not be reachable given the leading guard.
    raise AssertionError(
        "Unreachable: global_/pairwise/lineages combination not handled"
    )
