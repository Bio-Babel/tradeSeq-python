"""Per-gene design-matrix and per-fit basis construction (R source: tradeSeq/R/fitGAM.R:253-310).

Two responsibilities:

* :func:`build_smooth_design` constructs the GAM design matrix following the R formula
  ``y ~ -1 + U + s(t1, by=l1, bs='cr', id=1, k=nknots) + ... + offset(offset)``.
  The natural cubic-regression-spline basis is shared across all lineages
  (mgcv's ``id=1``), so the same per-lineage (``cr_basis``) is built once on
  the shared knot vector and replicated across lineage by-vars.

* :func:`build_smooth_design_with_conditions` extends the above for the
  conditions-aware formula at ``fitGAM.R:282-292``: each lineage is split into
  ``nlevels(conditions)`` condition-specific smoothers
  ``s(tj, by=lj_k, bs='cr', id=1, k=nknots)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._basis import cr_basis

__all__ = [
    "DesignMatrices",
    "build_smooth_design",
    "build_smooth_design_with_conditions",
    "build_lineage_indicators",
]


class DesignMatrices:
    """Container holding the per-gene design pieces.

    Attributes
    ----------
    lpmatrix
        Linear-predictor model matrix ``(n_cells, n_coefs)`` with mgcv-style
        column names: ``U_1`` etc for fixed effects, ``s(tj):lj.k`` for
        smoother basis columns (one per (lineage, basis-index) pair).
    dm
        Per-cell long-form design ``y, U, t1..tL, l1..lL, offset(offset)``
        DataFrame — the payload R stores in its fitted container's ``colData``.
    knots
        Shared knot vector ``(n_knots,)`` used to construct the cr-basis.
    S
        Block-diagonal smoothness penalty matrix, shape ``(n_coefs, n_coefs)``.
        Zero blocks for fixed-effect columns; one block per smoother term
        sharing the same per-lineage cr-basis penalty.
    smooth_blocks
        List of (start, stop) column-index pairs identifying each smoother
        block within ``lpmatrix``. Used downstream for diagnostics.
    """

    __slots__ = ("lpmatrix", "dm", "knots", "S", "smooth_blocks")

    def __init__(
        self,
        lpmatrix: pd.DataFrame,
        dm: pd.DataFrame,
        knots: np.ndarray,
        S: np.ndarray,
        smooth_blocks: list[tuple[int, int]],
    ):
        self.lpmatrix = lpmatrix
        self.dm = dm
        self.knots = knots
        self.S = S
        self.smooth_blocks = smooth_blocks


def build_lineage_indicators(w_samp: np.ndarray) -> np.ndarray:
    """Convert the multinomial assignment matrix to 0/1 lineage indicators.

    Mirrors ``1*(wSamp[,ii] == 1)`` in ``fitGAM.R:237-238``.
    """
    return (w_samp == 1).astype(np.float64)


def build_smooth_design(
    pseudotime: np.ndarray,
    w_samp: np.ndarray,
    U: np.ndarray,
    knots_list: dict[str, np.ndarray],
    offset: np.ndarray,
) -> DesignMatrices:
    """Build the design matrix for the no-conditions formula.

    Parameters
    ----------
    pseudotime
        ``(n_cells, n_lineages)`` matrix of per-cell pseudotimes.
    w_samp
        ``(n_cells, n_lineages)`` 0/1 assignment matrix.
    U
        ``(n_cells, n_U_cols)`` fixed-effect design (usually a single
        intercept column of ones).
    knots_list
        Dict from ``find_knots(...)``: ``{"t1": knots, "t2": knots, ...}``
        — same knot vector per lineage thanks to mgcv ``id=1`` sharing.
    offset
        ``(n_cells,)`` log-offset.

    Returns
    -------
    DesignMatrices
    """
    n_cells, n_lin = pseudotime.shape
    L = build_lineage_indicators(w_samp)
    knots = np.asarray(knots_list["t1"], dtype=np.float64)
    n_knots = knots.shape[0]
    # Penalty matrix per smoother is the cr-basis penalty (constant across
    # lineages because we share `knots` and use `id=1`).
    # The cr-basis design columns differ per lineage because they are
    # multiplied by the lineage indicator `lj`.
    # Build basis once on the full pseudotime axis per lineage, then column-
    # scale by lj.

    smooth_blocks: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    column_names: list[str] = []
    S_blocks: list[np.ndarray] = []

    # Fixed effects first
    n_U = U.shape[1]
    for j in range(n_U):
        columns.append(U[:, j])
        column_names.append(f"U{j+1}" if n_U > 1 else "U")
    S_blocks.append(np.zeros((n_U, n_U)))

    # Per-lineage cr-basis blocks
    for j in range(n_lin):
        tj = pseudotime[:, j]
        lj = L[:, j]
        # Pre-compute basis at every cell, then mask by lj before storing.
        X_basis, S_basis = cr_basis(tj, knots, scale_penalty=True)
        # The "by" variable lj multiplies the basis columns (mgcv convention)
        X_basis_lin = X_basis * lj[:, None]
        start = len(columns)
        for k in range(n_knots):
            columns.append(X_basis_lin[:, k])
            column_names.append(f"s(t{j+1}):l{j+1}.{k+1}")
        smooth_blocks.append((start, start + n_knots))
        S_blocks.append(S_basis)

    lp = np.column_stack(columns).astype(np.float64)
    lpmatrix = pd.DataFrame(lp, columns=column_names)

    # Block-diagonal S
    sizes = [b.shape[0] for b in S_blocks]
    total = sum(sizes)
    S = np.zeros((total, total))
    pos = 0
    for blk, sz in zip(S_blocks, sizes):
        S[pos : pos + sz, pos : pos + sz] = blk
        pos += sz

    # Long-form per-cell `dm` matching R's `m$model[, -1]` columns:
    # ``U, offset(offset), t1, l1, t2, l2, ..., tL, lL``.
    # mgcv interleaves each smoother's (t, by) pair from the formula
    #   y ~ -1 + U + s(t1, by=l1, ...) + s(t2, by=l2, ...) + ... + offset(offset)
    # and reorders the offset to sit immediately after the fixed-effect U.
    # Downstream code (e.g. _design._max_time_in_lineage's positional
    # ``lineageIds + off`` indexing into dm) relies on this interleaved
    # layout — see R reference at fitGAM.R:264-270 and the dumped fitted
    # container design-matrix column names.
    dm_cols: dict[str, np.ndarray] = {}
    if n_U == 1:
        dm_cols["U"] = U[:, 0]
    else:
        for j in range(n_U):
            dm_cols[f"U{j+1}"] = U[:, j]
    dm_cols["offset(offset)"] = offset
    for j in range(n_lin):
        dm_cols[f"t{j+1}"] = pseudotime[:, j]
        dm_cols[f"l{j+1}"] = L[:, j]
    dm = pd.DataFrame(dm_cols)

    return DesignMatrices(
        lpmatrix=lpmatrix,
        dm=dm,
        knots=knots,
        S=S,
        smooth_blocks=smooth_blocks,
    )


def build_smooth_design_with_conditions(
    pseudotime: np.ndarray,
    w_samp: np.ndarray,
    U: np.ndarray,
    knots_list: dict[str, np.ndarray],
    offset: np.ndarray,
    conditions: pd.Categorical,
) -> DesignMatrices:
    """Build the design matrix for the conditions-aware formula.

    The R formula at ``fitGAM.R:282-292`` is::

        y ~ -1 + U + sum_{j,k} s(tj, by=lj_k, bs='cr', id=1, k=nknots)
            + offset(offset)

    where ``lj_k`` is the indicator ``lj == 1 AND conditions == k``. The
    cr-basis itself is unchanged; we replicate the design columns once per
    ``(lineage, condition)`` pair instead of once per lineage.
    """
    n_cells, n_lin = pseudotime.shape
    L = build_lineage_indicators(w_samp)
    conditions = pd.Categorical(conditions)
    cond_codes = np.asarray(conditions.codes, dtype=np.int64)
    cond_levels = conditions.categories
    n_cond = len(cond_levels)

    knots = np.asarray(knots_list["t1"], dtype=np.float64)
    n_knots = knots.shape[0]

    smooth_blocks: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    column_names: list[str] = []
    S_blocks: list[np.ndarray] = []

    n_U = U.shape[1]
    for j in range(n_U):
        columns.append(U[:, j])
        column_names.append(f"U{j+1}" if n_U > 1 else "U")
    S_blocks.append(np.zeros((n_U, n_U)))

    # Per (lineage, condition) cr-basis blocks
    cond_indicators = np.zeros((n_cells, n_cond), dtype=np.float64)
    for k in range(n_cond):
        cond_indicators[:, k] = (cond_codes == k).astype(np.float64)

    for j in range(n_lin):
        tj = pseudotime[:, j]
        lj = L[:, j]
        X_basis, S_basis = cr_basis(tj, knots, scale_penalty=True)
        for k in range(n_cond):
            ljk = lj * cond_indicators[:, k]
            X_basis_lin = X_basis * ljk[:, None]
            start = len(columns)
            for b in range(n_knots):
                columns.append(X_basis_lin[:, b])
                column_names.append(f"s(t{j+1}):l{j+1}_{k+1}.{b+1}")
            smooth_blocks.append((start, start + n_knots))
            S_blocks.append(S_basis)

    lp = np.column_stack(columns).astype(np.float64)
    lpmatrix = pd.DataFrame(lp, columns=column_names)

    sizes = [b.shape[0] for b in S_blocks]
    total = sum(sizes)
    S = np.zeros((total, total))
    pos = 0
    for blk, sz in zip(S_blocks, sizes):
        S[pos : pos + sz, pos : pos + sz] = blk
        pos += sz

    # Long-form per-cell `dm` matching R's `m$model[, -1]` columns for the
    # conditions-aware formula. The formula at fitGAM.R:283-292 generates
    # the per-lineage block ``s(tj, by=lj_1) + s(tj, by=lj_2) + ... +
    # s(tj, by=lj_K)`` — i.e. ``tj`` is followed by each ``lj_k`` in order,
    # then the next lineage. mgcv places ``offset(offset)`` right after the
    # fixed-effect U.
    # Final order: ``U, offset(offset), t1, l1_1, l1_2, ..., l1_K,
    #               t2, l2_1, ..., l2_K, ...``.
    dm_cols: dict[str, np.ndarray] = {}
    if n_U == 1:
        dm_cols["U"] = U[:, 0]
    else:
        for j in range(n_U):
            dm_cols[f"U{j+1}"] = U[:, j]
    dm_cols["offset(offset)"] = offset
    for j in range(n_lin):
        dm_cols[f"t{j+1}"] = pseudotime[:, j]
        for k in range(n_cond):
            dm_cols[f"l{j+1}_{k+1}"] = L[:, j] * cond_indicators[:, k]
    dm = pd.DataFrame(dm_cols)

    return DesignMatrices(
        lpmatrix=lpmatrix,
        dm=dm,
        knots=knots,
        S=S,
        smooth_blocks=smooth_blocks,
    )
