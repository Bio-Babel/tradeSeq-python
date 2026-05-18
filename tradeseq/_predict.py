"""Linear-predictor matrix reconstruction (R source: tradeSeq/R/utils.R:200-335).

Implements :func:`predict_gam`, the Python port of ``tradeSeq:::predictGAM``.

The function reconstructs the GAM linear-predictor matrix (``lpmatrix``) at new
pseudotime points by fitting a cubic spline through each per-lineage basis
column and evaluating it at the requested locations. This mirrors R's
``stats::splinefun(..., ties = mean)`` with the default ``fmm`` (not-a-knot)
method, which is the not-a-knot cubic spline.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

__all__ = ["predict_gam"]


_BASIS_RE = re.compile(r"[0-9]\):l[1-9]")
_LINEAGE_NOCOND_RE = re.compile(r"s\((t[0-9]+)\):.*")
_LINEAGE_COND_RE = re.compile(r"s\(t([0-9]+)\):.*")
_CURVE_COND_RE = re.compile(r".*:l([0-9]+_[0-9]+)\..*")


def _collapse_ties_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average ``y`` values that share the same ``x`` and return sorted unique x.

    Mirrors R's ``stats::splinefun(..., ties = mean)``: duplicate x-coordinates
    are pre-collapsed by averaging the corresponding y values so the cubic
    spline can be fitted on strictly increasing x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]
    uniq, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if uniq.size == x_sorted.size:
        return uniq, y_sorted
    y_mean = np.zeros(uniq.size, dtype=float)
    np.add.at(y_mean, inverse, y_sorted)
    y_mean /= counts
    return uniq, y_mean


def _spline_eval(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Fit a not-a-knot cubic spline through ``(x_train, y_train)`` and evaluate at ``x_eval``.

    Pre-collapses duplicate ``x_train`` by averaging y, matching
    ``stats::splinefun(ties = mean)``.
    """
    x_u, y_u = _collapse_ties_mean(x_train, y_train)
    cs = CubicSpline(x_u, y_u, bc_type="not-a-knot", extrapolate=True)
    return cs(np.asarray(x_eval, dtype=float))


def predict_gam(
    lpmatrix: pd.DataFrame,
    df: pd.DataFrame,
    pseudotime: np.ndarray,
    conditions: Sequence | None = None,
) -> pd.DataFrame:
    """Reconstruct the linear-predictor matrix at new pseudotime locations.

    Port of ``tradeSeq:::predictGAM`` (utils.R:200-335). For each lineage and
    each basis column of ``lpmatrix``, fits a cubic spline through the cells
    assigned to that lineage and evaluates the spline at the pseudotime grid
    given in ``df``. Non-basis columns (fixed covariates, intercept) are copied
    from ``df``.

    Parameters
    ----------
    lpmatrix : pd.DataFrame
        The GAM linear-predictor matrix (n_cells × n_features). Column names
        following the pattern ``s(tK):lK.J`` (no conditions) or
        ``s(tK):lK_C.J`` (with conditions) are treated as basis columns.
    df : pd.DataFrame
        Prediction grid. Must contain ``tI`` columns (pseudotime for lineage I)
        and ``lI`` (or ``lI_C``) lineage-weight columns. Other columns are
        copied through as fixed covariates.
    pseudotime : np.ndarray
        ``(n_cells, n_lineages)`` matrix of per-cell pseudotime values. May be
        passed as a 1-D array if there is a single lineage.
    conditions : sequence or None, optional
        Per-cell condition labels (factor). When provided, the lpmatrix columns
        are interpreted in the conditional form ``s(tK):lK_C.J``.

    Returns
    -------
    pd.DataFrame
        Output linear-predictor matrix of shape ``(nrow(df), ncol(lpmatrix))``
        with the same column ordering as ``lpmatrix``.
    """
    pseudotime = np.asarray(pseudotime, dtype=float)
    if pseudotime.ndim == 1:
        pseudotime = pseudotime.reshape(-1, 1)

    cond_present = conditions is not None
    if cond_present:
        conditions_arr = np.asarray(conditions)
        # match R's nlevels() — distinct labels in stable order
        cond_levels = pd.Categorical(conditions_arr).categories
        n_conditions = len(cond_levels)

    lp_cols = list(lpmatrix.columns)
    n_cols = len(lp_cols)
    lp_values = lpmatrix.to_numpy()

    # for each curve, specify basis function IDs for lpmatrix (R's `allBs`)
    all_bs = [i for i, c in enumerate(lp_cols) if _BASIS_RE.search(c)]

    # parse lineage / curve / condition identifiers
    if not cond_present:
        lineage_labels = []
        for i in all_bs:
            m = _LINEAGE_NOCOND_RE.match(lp_cols[i])
            # column form: s(tK):lK.J → take the inner "tK"
            lineage_labels.append(m.group(1))
        unique_lineages = []
        seen = set()
        for lab in lineage_labels:
            if lab not in seen:
                unique_lineages.append(lab)
                seen.add(lab)
        n_curves = len(unique_lineages)
        # ids[ii] = list of column indices belonging to lineage tii (1-indexed)
        ids: list[list[int]] = [[] for _ in range(n_curves)]
        for col_idx, lab in zip(all_bs, lineage_labels):
            lineage_idx = unique_lineages.index(lab)
            ids[lineage_idx].append(col_idx)
    else:
        # column form: s(tI):lI_C.J  ; lineage_labels collects "I", curve_labels collects "I_C"
        lineage_labels = []
        curve_labels = []
        for i in all_bs:
            col = lp_cols[i]
            m_lin = _LINEAGE_COND_RE.match(col)
            m_cur = _CURVE_COND_RE.match(col)
            lineage_labels.append(m_lin.group(1))
            curve_labels.append(m_cur.group(1))
        unique_lineages = []
        seen = set()
        for lab in lineage_labels:
            if lab not in seen:
                unique_lineages.append(lab)
                seen.add(lab)
        n_lineages = len(unique_lineages)
        unique_curves = []
        seen_c = set()
        for lab in curve_labels:
            if lab not in seen_c:
                unique_curves.append(lab)
                seen_c.add(lab)
        # ids_cond[(ii, kk)] = list of column indices for lineage ii, condition kk (1-indexed)
        ids_cond: dict[tuple[int, int], list[int]] = {}
        for ii in range(1, n_lineages + 1):
            for kk in range(1, n_conditions + 1):
                key = f"{ii}_{kk}"
                ids_cond[(ii, kk)] = [
                    col_idx
                    for col_idx, lab in zip(all_bs, curve_labels)
                    if lab == key
                ]

    # specify lineage assignment for each cell (row of lpmatrix)
    n_cells = lp_values.shape[0]
    lineage_id = np.zeros(n_cells, dtype=int)
    if not cond_present:
        for row in range(n_cells):
            assigned = 0
            for ii in range(n_curves):
                cols_ii = ids[ii]
                if not np.all(lp_values[row, cols_ii] == 0):
                    assigned = ii + 1
                    break
            lineage_id[row] = assigned
    else:
        # encode as numeric "II_KK" → as_numeric(paste0(ii, kk)); R uses `paste0(ii, kk)`.
        # Mirror exactly: 2 digits → ii*10+kk, but R uses paste0("11") → 11
        for row in range(n_cells):
            assigned = 0
            done = False
            for ii in range(1, n_lineages + 1):
                if done:
                    break
                for kk in range(1, n_conditions + 1):
                    cols_iikk = ids_cond[(ii, kk)]
                    if not np.all(lp_values[row, cols_iikk] == 0):
                        assigned = int(f"{ii}{kk}")
                        done = True
                        break
            lineage_id[row] = assigned

    # fit splinefun for each basis function based on assigned cells
    # We store the trained x/y per (lineage, basis_index) and evaluate per call.
    df_values = df.to_numpy()
    df_cols = list(df.columns)
    n_rows_out = df.shape[0]
    x_out = np.zeros((n_rows_out, n_cols), dtype=float)

    if not cond_present:
        n_basis_per_curve = len(all_bs) // n_curves
        for ii in range(1, n_curves + 1):
            # cell mask = cells assigned to lineage ii
            mask = lineage_id == ii
            x_train_curve = pseudotime[mask, ii - 1]  # column ii-1 of pseudotime
            l_col = df_cols.index(f"l{ii}")
            # only predict if df[, "lI"] all == 1
            if not np.all(df_values[:, l_col] == 1):
                continue
            t_col = df_cols.index(f"t{ii}")
            x_eval = df_values[:, t_col]
            for jj in range(n_basis_per_curve):
                basis_col = ids[ii - 1][jj]
                y_train = lp_values[mask, basis_col]
                x_out[:, basis_col] = _spline_eval(x_train_curve, y_train, x_eval)
    else:
        n_basis_per_curve = len(all_bs) // (n_lineages * n_conditions)
        for ii in range(1, n_lineages + 1):
            for kk in range(1, n_conditions + 1):
                lineage_code = int(f"{ii}{kk}")
                mask = lineage_id == lineage_code
                x_train_curve = pseudotime[mask, ii - 1]
                l_col_name = f"l{ii}_{kk}"
                l_col = df_cols.index(l_col_name)
                # only predict if df[, "lI_K"] != 0 (R: all(... != 0))
                if not np.all(df_values[:, l_col] != 0):
                    continue
                t_col = df_cols.index(f"t{ii}")
                x_eval = df_values[:, t_col]
                for jj in range(n_basis_per_curve):
                    basis_col = ids_cond[(ii, kk)][jj]
                    y_train = lp_values[mask, basis_col]
                    x_out[:, basis_col] = _spline_eval(x_train_curve, y_train, x_eval)

    # add fixed covariates as in df
    # R: dfSmoothID <- grep(x = colnames(df), pattern = "[t|l][1-9]")
    # The pattern "[t|l][1-9]" in R is a character class containing t, |, l, so it matches
    # any of t, |, or l followed by a digit 1-9. In practice colnames don't contain |.
    smooth_pattern = re.compile(r"[t|l][1-9]")
    offset_pattern = re.compile(r"offset")
    df_smooth_id = [i for i, c in enumerate(df_cols) if smooth_pattern.search(c)]
    df_offset_id = [i for i, c in enumerate(df_cols) if offset_pattern.search(c)]
    excluded_df = set(df_smooth_id) | set(df_offset_id)
    fixed_df_cols = [i for i in range(len(df_cols)) if i not in excluded_df]
    # write to columns of x_out that are NOT in all_bs
    other_cols = [i for i in range(n_cols) if i not in all_bs]
    # R does Xout[, -allBs] <- df[, -c(dfSmoothID, dfOffsetID)] (column-by-column)
    # The number of remaining columns must match.
    if len(other_cols) != len(fixed_df_cols):
        raise ValueError(
            f"Mismatch: {len(other_cols)} non-basis lpmatrix columns vs "
            f"{len(fixed_df_cols)} fixed df columns."
        )
    for out_idx, df_idx in zip(other_cols, fixed_df_cols):
        x_out[:, out_idx] = df_values[:, df_idx]

    return pd.DataFrame(x_out, columns=lp_cols)
