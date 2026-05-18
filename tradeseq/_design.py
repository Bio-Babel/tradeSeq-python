"""Design-matrix slicers (R source: tradeSeq/R/utils.R:1-196)."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "get_predict_start_point_df",
    "get_predict_end_point_df",
    "get_predict_custom_point_df",
    "get_predict_range_df",
    "pattern_df",
    "pattern_df_pairwise",
]

# R `grep` is unanchored, so these patterns intentionally do not start with '^'.
_OFFSET_NAME_RE = re.compile(r"offset")
_TIME_COL_RE = re.compile(r"t[1-9]")
_LINEAGE_COL_RE = re.compile(r"l[1-9]")
_OFFSET_PEEL_RE = re.compile(r"^offset\((.+)\)$")


def _grep(columns: Sequence[str], pattern: re.Pattern) -> list[int]:
    """Return positional indices of `columns` matching `pattern` (mirrors R `grep`)."""
    return [i for i, c in enumerate(columns) if pattern.search(c) is not None]


def _grep_str(columns: Sequence[str], pattern: str) -> list[int]:
    """Compile `pattern` once then run unanchored search across `columns`."""
    rx = re.compile(pattern)
    return [i for i, c in enumerate(columns) if rx.search(c) is not None]


def _peel_offset_name(name: str) -> str:
    """Convert `offset(x)` -> `x`. Mirrors the R `substr(.., 8, nchar - 1)` slice."""
    match = _OFFSET_PEEL_RE.match(name)
    if match is None:
        raise ValueError(
            f"offset column {name!r} does not match the pattern 'offset(<name>)'"
        )
    return match.group(1)


def _prepare_vars(dm: pd.DataFrame) -> tuple[pd.DataFrame, int, str, int]:
    """Replicate the shared preamble of all five R slicers.

    Returns
    -------
    vars : pd.DataFrame
        Single-row frame copied from ``dm.iloc[[0]]`` with the ``y`` column dropped
        (if present) and the ``offset(<name>)`` column renamed to ``<name>``.
    off : int
        ``1`` if the original ``dm`` had a ``y`` column, else ``0`` — used to shift
        column indices in ``dm`` for membership lookups against ``vars`` positions.
    offset_name : str
        The peeled offset name (e.g. ``"offset"``).
    offset_id_dm : int
        Position of the offset column in the ORIGINAL ``dm`` (with ``y``), used by
        ``mean(dm[, grep("offset", ...)])`` calls.
    """
    vars_df = dm.iloc[[0]].copy()
    if "y" in vars_df.columns:
        vars_df = vars_df.drop(columns=["y"])
        off = 1
    else:
        off = 0
    offset_positions = _grep(vars_df.columns, _OFFSET_NAME_RE)
    if len(offset_positions) != 1:
        raise ValueError(
            "expected exactly one 'offset(...)' column in dm, "
            f"found {len(offset_positions)}: {list(vars_df.columns)}"
        )
    offset_id = offset_positions[0]
    offset_name = _peel_offset_name(vars_df.columns[offset_id])
    new_cols = list(vars_df.columns)
    new_cols[offset_id] = offset_name
    vars_df.columns = pd.Index(new_cols)
    # Position of the offset column in the ORIGINAL dm:
    offset_id_dm = _grep(dm.columns, _OFFSET_NAME_RE)[0]
    return vars_df, off, offset_name, offset_id_dm


def _zero_time_and_lineage(vars_df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Zero all `t[1-9]` and `l[1-9]` columns. Returns (t_idx, l_idx)."""
    t_idx = _grep(vars_df.columns, _TIME_COL_RE)
    l_idx = _grep(vars_df.columns, _LINEAGE_COL_RE)
    for j in t_idx:
        vars_df.iloc[:, j] = 0
    for j in l_idx:
        vars_df.iloc[:, j] = 0
    return t_idx, l_idx


def _assign_lineage_weight(
    vars_df: pd.DataFrame, lineage_ids: list[int], weight: float
) -> None:
    """Set the columns at positions ``lineage_ids`` to ``weight``.

    Casts each touched column to ``float64`` first so a fractional ``1/k`` assignment
    on an integer column does not raise pandas' ``FutureWarning`` about silent
    dtype downcasting. R promotes atomic-vector types implicitly here.
    """
    for j in lineage_ids:
        col = vars_df.columns[j]
        if vars_df[col].dtype.kind in "iub":
            vars_df[col] = vars_df[col].astype(np.float64)
        vars_df.iloc[:, j] = weight


def _max_time_in_lineage(
    dm: pd.DataFrame,
    lineage_ids_vars: list[int],
    off: int,
    time_col: str,
) -> float:
    """Compute the max value of `time_col` in cells assigned to the lineage (or composite).

    Mirrors the R idiom:
        if length(lineageIds)==1:
            max(dm[dm[, lineageIds + off]==1, paste0("t", lineageId)])
        else:
            max(dm[rowSums(dm[, lineageIds + off])==1, paste0("t", lineageId)])
    """
    dm_cols_for_lineage = [j + off for j in lineage_ids_vars]
    if len(dm_cols_for_lineage) == 1:
        col_vals = dm.iloc[:, dm_cols_for_lineage[0]].to_numpy()
        mask = col_vals == 1
    else:
        sub = dm.iloc[:, dm_cols_for_lineage].to_numpy()
        mask = sub.sum(axis=1) == 1
    return float(dm.loc[mask, time_col].max())


def _all_times_in_lineage(
    dm: pd.DataFrame,
    lineage_ids_vars: list[int],
    off: int,
    time_col: str,
) -> np.ndarray:
    """Return the vector of `time_col` for cells in the (composite) lineage."""
    dm_cols_for_lineage = [j + off for j in lineage_ids_vars]
    if len(dm_cols_for_lineage) == 1:
        col_vals = dm.iloc[:, dm_cols_for_lineage[0]].to_numpy()
        mask = col_vals == 1
    else:
        sub = dm.iloc[:, dm_cols_for_lineage].to_numpy()
        mask = sub.sum(axis=1) == 1
    return dm.loc[mask, time_col].to_numpy()


def _mean_offset(dm: pd.DataFrame) -> float:
    """Return the mean of the original `offset(<name>)` column in dm."""
    offset_positions = _grep(dm.columns, _OFFSET_NAME_RE)
    if len(offset_positions) != 1:
        raise ValueError(
            "expected exactly one 'offset(...)' column in dm, "
            f"found {len(offset_positions)}: {list(dm.columns)}"
        )
    return float(dm.iloc[:, offset_positions[0]].mean())


def get_predict_start_point_df(dm: pd.DataFrame, lineage_id: int) -> pd.DataFrame:
    """Return the design-matrix row for the start of a smoother (R utils.R:39-65).

    Parameters
    ----------
    dm : pandas.DataFrame
        Per-cell design matrix with columns ``y`` (optional), ``U``,
        ``offset(<name>)``, and ``t1..tL``, ``l1..lL`` (single-condition) or
        ``l1_1..lL_K`` (multi-condition).
    lineage_id : int
        1-based lineage index.

    Returns
    -------
    pandas.DataFrame
        Single-row prediction frame: times zeroed, lineage indicators zeroed
        except the requested lineage set to ``1 / k`` (where ``k`` is the number
        of condition columns for that lineage), offset set to the mean of the
        original offset column. The ``y`` column is dropped and ``offset(<name>)``
        is renamed to ``<name>``.
    """
    vars_df, _off, offset_name, _offset_id_dm = _prepare_vars(dm)
    _zero_time_and_lineage(vars_df)
    lineage_ids = _grep_str(vars_df.columns, rf"l{lineage_id}($|_)")
    weight = 1.0 / len(lineage_ids)
    _assign_lineage_weight(vars_df, lineage_ids, weight)
    vars_df[offset_name] = _mean_offset(dm)
    return vars_df


def get_predict_end_point_df(dm: pd.DataFrame, lineage_id: int) -> pd.DataFrame:
    """Return the design-matrix row for the end of a smoother (R utils.R:3-37).

    Identical to :func:`get_predict_start_point_df` except that the ``t<lineage_id>``
    column is set to the maximum pseudotime observed in cells assigned to that
    lineage (or composite lineage when multiple condition columns exist).

    Parameters
    ----------
    dm : pandas.DataFrame
        Per-cell design matrix.
    lineage_id : int
        1-based lineage index.

    Returns
    -------
    pandas.DataFrame
        Single-row prediction frame.
    """
    vars_df, off, offset_name, _offset_id_dm = _prepare_vars(dm)
    _zero_time_and_lineage(vars_df)
    lineage_ids = _grep_str(vars_df.columns, rf"l{lineage_id}($|_)")
    time_col = f"t{lineage_id}"
    vars_df[time_col] = _max_time_in_lineage(dm, lineage_ids, off, time_col)
    weight = 1.0 / len(lineage_ids)
    _assign_lineage_weight(vars_df, lineage_ids, weight)
    vars_df[offset_name] = _mean_offset(dm)
    return vars_df


def get_predict_custom_point_df(
    dm: pd.DataFrame,
    lineage_id: int,
    pseudotime: float,
    condition: Optional[object] = None,
) -> pd.DataFrame:
    """Return the design-matrix row at a custom pseudotime (R utils.R:67-106).

    Parameters
    ----------
    dm : pandas.DataFrame
        Per-cell design matrix.
    lineage_id : int
        1-based lineage index.
    pseudotime : float
        Pseudotime to assign to the ``t<lineage_id>`` column.
    condition : object, optional
        If provided, the lineage columns matched are ``l<lineage_id>_<condition>``
        (so only one condition column is activated). If ``None``, all composite
        condition columns for the lineage are activated with weight ``1 / k``.

    Returns
    -------
    pandas.DataFrame
        Single-row prediction frame.
    """
    vars_df, _off, offset_name, _offset_id_dm = _prepare_vars(dm)
    _zero_time_and_lineage(vars_df)
    if condition is not None:
        lineage_ids = _grep_str(vars_df.columns, rf"l{lineage_id}_{condition}")
    else:
        lineage_ids = _grep_str(vars_df.columns, rf"l{lineage_id}($|_)")
    weight = 1.0 / len(lineage_ids)
    _assign_lineage_weight(vars_df, lineage_ids, weight)
    vars_df[f"t{lineage_id}"] = pseudotime
    vars_df[offset_name] = _mean_offset(dm)
    return vars_df


def get_predict_range_df(
    dm: pd.DataFrame,
    lineage_id: int,
    condition_id: Optional[object] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """Return an `n_points`-row design-matrix range for a smoother (R utils.R:108-154).

    Parameters
    ----------
    dm : pandas.DataFrame
        Per-cell design matrix.
    lineage_id : int
        1-based lineage index.
    condition_id : object, optional
        If provided, only the column ``l<lineage_id>_<condition_id>`` is activated
        (anchored at the end with ``$``). If ``None``, all composite condition
        columns for the lineage are activated with weight ``1 / k``.
    n_points : int, default 100
        Number of rows in the returned frame.

    Returns
    -------
    pandas.DataFrame
        ``n_points``-row frame where ``t<lineage_id>`` linearly interpolates the
        observed range of pseudotime in the lineage. If the minimum observed time
        is within 1% of the maximum, the minimum row is forced to ``0`` (R's
        guard against drifting starts).
    """
    vars_df, off, offset_name, _offset_id_dm = _prepare_vars(dm)
    _zero_time_and_lineage(vars_df)
    # Duplicate the single row to n_points rows. We avoid pd.concat warnings.
    vars_df = pd.DataFrame(
        {col: np.repeat(vars_df[col].to_numpy(), n_points) for col in vars_df.columns}
    )
    if condition_id is None:
        lineage_ids = _grep_str(vars_df.columns, rf"l{lineage_id}($|_)")
    else:
        lineage_ids = _grep_str(vars_df.columns, rf"l{lineage_id}_{condition_id}$")
    lineage_data = _all_times_in_lineage(dm, lineage_ids, off, f"t{lineage_id}")
    # R: if min / max < 0.01 -> push the minimum row to 0
    lineage_min = float(lineage_data.min())
    lineage_max = float(lineage_data.max())
    if lineage_min / lineage_max < 0.01:
        # Mutate a copy: the only consumer is min/max below.
        lineage_data = lineage_data.copy()
        lineage_data[int(np.argmin(lineage_data))] = 0.0
        lineage_min = float(lineage_data.min())
    weight = 1.0 / len(lineage_ids)
    _assign_lineage_weight(vars_df, lineage_ids, weight)
    vars_df[f"t{lineage_id}"] = np.linspace(lineage_min, lineage_max, n_points)
    vars_df[offset_name] = _mean_offset(dm)
    return vars_df


def pattern_df(
    dm: pd.DataFrame,
    knots: Optional[Sequence[int]] = None,
    knot_points: Optional[Sequence[float]] = None,
    n_points: int = 100,
) -> list[pd.DataFrame]:
    """Return a list of design-matrix ranges, one per lineage (R utils.R:156-175).

    Parameters
    ----------
    dm : pandas.DataFrame
        Per-cell design matrix.
    knots : sequence of int, optional
        1-based knot indices ``(k1, k2)`` selecting a sub-range of pseudotime to
        evaluate. If provided, ``knot_points[knots[0]-1]`` and
        ``knot_points[knots[1]-1]`` define the linear range overriding the
        observed lineage range from :func:`get_predict_range_df`.
    knot_points : sequence of float, optional
        Full knot vector. Required if ``knots`` is provided.
    n_points : int, default 100
        Number of evaluation points per lineage.

    Returns
    -------
    list of pandas.DataFrame
        One ``n_points``-row prediction frame per lineage.
    """
    n_lineages = len(_grep(dm.columns, _TIME_COL_RE))
    knot_active = knots is not None
    if knot_active:
        # R uses 1-based indexing: knotPoints[knots[1]] and knotPoints[knots[2]].
        t1 = knot_points[knots[0] - 1]
        t2 = knot_points[knots[1] - 1]
    df_list: list[pd.DataFrame] = []
    for jj in range(1, n_lineages + 1):
        df = get_predict_range_df(dm, jj, n_points=n_points)
        if knot_active:
            df[f"t{jj}"] = np.linspace(t1, t2, n_points)
        df_list.append(df)
    return df_list


def pattern_df_pairwise(
    dm: pd.DataFrame,
    curves: Sequence[int],
    knots: Optional[Sequence[int]] = None,
    knot_points: Optional[Sequence[float]] = None,
    n_points: int = 100,
) -> list[pd.DataFrame]:
    """Return two design-matrix ranges, one per `curves` entry (R utils.R:177-196).

    Parameters
    ----------
    dm : pandas.DataFrame
        Per-cell design matrix.
    curves : sequence of int
        Length-2 sequence of 1-based lineage indices to evaluate.
    knots : sequence of int, optional
        1-based knot indices ``(k1, k2)``. See :func:`pattern_df`.
    knot_points : sequence of float, optional
        Full knot vector.
    n_points : int, default 100
        Number of evaluation points per lineage.

    Returns
    -------
    list of pandas.DataFrame
        Two ``n_points``-row prediction frames, one per lineage in ``curves``.
    """
    knot_active = knots is not None
    if knot_active:
        t1 = knot_points[knots[0] - 1]
        t2 = knot_points[knots[1] - 1]
    df_list: list[pd.DataFrame] = []
    for jj in range(2):
        lineage = curves[jj]
        df = get_predict_range_df(dm, lineage, n_points=n_points)
        if knot_active:
            df[f"t{lineage}"] = np.linspace(t1, t2, n_points)
        df_list.append(df)
    return df_list
