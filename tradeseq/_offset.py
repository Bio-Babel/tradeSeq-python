"""Sequencing-depth offset for the NB-GAM (R source: tradeSeq/R/fitGAM.R:82-98).

Ports two helpers:

- :func:`calc_norm_factors_tmm`: NumPy port of ``edgeR::calcNormFactors`` with
  ``method="TMM"`` and edgeR's default trim parameters. Mirrors the algorithm
  of Robinson & Oshlack (2010), using the published edgeR source as gold
  reference (``edgeR:::.calcFactorTMM`` and ``edgeR:::.calcFactorQuantile``).
- :func:`compute_offset`: NumPy port of ``tradeSeq:::.get_offset``, which
  combines the TMM factors with library sizes to produce
  ``log(colSums(counts) * nf)`` as the GAM offset.

Input convention
----------------
Both functions accept a ``(n_genes, n_cells)`` matrix — the same orientation
as the R source. The orchestrator (``fit_gam``) transposes once from the
AnnData ``(n_cells, n_genes)`` layer.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Union

import numpy as np
from scipy import sparse

__all__ = ["calc_norm_factors_tmm", "compute_offset"]


_CountMatrix = Union[np.ndarray, sparse.spmatrix]


def _as_dense(counts: _CountMatrix) -> np.ndarray:
    """Densify a sparse matrix; pass dense arrays through."""
    if sparse.issparse(counts):
        return counts.toarray()
    return np.asarray(counts)


def _log2_libm(x: np.ndarray) -> np.ndarray:
    """Element-wise ``log2`` using the C math library, matching R bitwise.

    NumPy's vectorised ``np.log2`` calls SIMD/SVML implementations whose
    results can differ from libm by 1 ULP. R's ``log2`` defers to libm
    directly. When TMM's rank-based trimming consumes these values, even a
    1 ULP discrepancy can flip ties and change which genes are retained.
    Looping through ``math.log2`` (libm-backed) restores bitwise parity.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr)
    flat = arr.ravel()
    out_flat = out.ravel()
    for i, v in enumerate(flat):
        if v > 0.0 and math.isfinite(v):
            out_flat[i] = math.log2(v)
        elif v == 0.0:
            out_flat[i] = -math.inf
        elif math.isnan(v):
            out_flat[i] = math.nan
        elif v == math.inf:
            out_flat[i] = math.inf
        else:  # negative finite
            out_flat[i] = math.nan
    return out


def calc_norm_factors_tmm(
    counts: _CountMatrix,
    *,
    lib_size: Optional[np.ndarray] = None,
    ref_column: Optional[int] = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    a_cutoff: float = -1e10,
    p: float = 0.75,
) -> np.ndarray:
    """Compute TMM normalization factors (port of ``edgeR::calcNormFactors``).

    Parameters
    ----------
    counts
        ``(n_genes, n_cells)`` count matrix. Sparse inputs are densified once
        (TMM is dense-only inside edgeR).
    lib_size
        Optional per-cell library sizes. Defaults to ``colSums(counts)``.
    ref_column
        Optional 0-indexed reference column. Defaults to edgeR's rule: the
        column whose upper-quartile-normalised factor is closest to the mean
        of those factors; if the median of those factors is below ``1e-20``,
        the column with the largest ``colSums(sqrt(counts))`` is used.
    logratio_trim, sum_trim
        Trim quantiles for M (log fold-change) and A (mean log abundance),
        respectively. Defaults mirror edgeR (``0.3`` and ``0.05``).
    do_weighting
        Whether to weight by ``1 / asymptotic-variance`` in the trimmed mean.
        Defaults to ``True`` per edgeR.
    a_cutoff
        Minimum A value for inclusion. Defaults to edgeR's ``-1e10``
        (effectively no cutoff).
    p
        Quantile for the upper-quartile rule used in ref-column selection.

    Returns
    -------
    numpy.ndarray
        Length-``n_cells`` TMM factors, geometric-mean normalised to 1.

    Notes
    -----
    Algorithm (mirrors ``edgeR:::.calcFactorTMM``):

    1. Drop genes that are zero across all cells.
    2. Choose ``ref_column`` using upper-quartile-normalised factors.
    3. For each cell ``i``, compute trimmed weighted mean of
       ``log2((ki/Li) / (k_ref/L_ref))`` over genes passing M-trim and
       A-trim quantile masks; convert to a factor via ``2^mean``.
    4. Re-scale all factors so their geometric mean is 1.
    """
    x = _as_dense(counts)
    if np.any(np.isnan(x)):
        raise ValueError("NA counts not permitted")

    n_genes, n_samples = x.shape

    if lib_size is None:
        lib_size = x.sum(axis=0)
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)
        if np.any(np.isnan(lib_size)):
            raise ValueError("NA lib.sizes not permitted")
        if lib_size.shape[0] != n_samples:
            raise ValueError(
                "length(lib.size) doesn't match number of samples"
            )

    lib_size = np.asarray(lib_size, dtype=np.float64)

    # Drop all-zero genes (R: `allzero <- .rowSums(x > 0, ...) == 0`).
    allzero = (x > 0).sum(axis=1) == 0
    if np.any(allzero):
        x = x[~allzero, :]
        n_genes = x.shape[0]

    # Degenerate cases collapse to method="none" in edgeR.
    if n_genes == 0 or n_samples == 1:
        f = np.ones(n_samples, dtype=np.float64)
        return f / np.exp(np.mean(np.log(f)))

    # Reference column selection (R: upper-quartile-normalised factors).
    if ref_column is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f75 = _calc_factor_quantile(x, lib_size, p=p)
        if np.median(f75) < 1e-20:
            ref_column = int(np.argmax(np.sqrt(x).sum(axis=0)))
        else:
            ref_column = int(np.argmin(np.abs(f75 - np.mean(f75))))

    ref = x[:, ref_column].astype(np.float64)
    libsize_ref = float(lib_size[ref_column])

    f = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        f[i] = _calc_factor_tmm(
            obs=x[:, i].astype(np.float64),
            ref=ref,
            libsize_obs=float(lib_size[i]),
            libsize_ref=libsize_ref,
            logratio_trim=logratio_trim,
            sum_trim=sum_trim,
            do_weighting=do_weighting,
            a_cutoff=a_cutoff,
        )

    f = f / np.exp(np.mean(np.log(f)))
    return f


def _calc_factor_quantile(
    data: np.ndarray, lib_size: np.ndarray, p: float
) -> np.ndarray:
    """Port of ``edgeR:::.calcFactorQuantile``.

    R uses ``quantile(...)`` with default ``type=7`` interpolation; NumPy's
    ``np.quantile(..., method="linear")`` (default) matches that exactly.
    """
    f = np.empty(data.shape[1], dtype=np.float64)
    for j in range(data.shape[1]):
        f[j] = np.quantile(data[:, j], p)
    if np.min(f) == 0:
        warnings.warn("One or more quantiles are zero", stacklevel=2)
    return f / lib_size


def _calc_factor_tmm(
    obs: np.ndarray,
    ref: np.ndarray,
    libsize_obs: float,
    libsize_ref: float,
    logratio_trim: float,
    sum_trim: float,
    do_weighting: bool,
    a_cutoff: float,
) -> float:
    """Port of ``edgeR:::.calcFactorTMM`` for one observation column."""
    obs = obs.astype(np.float64)
    ref = ref.astype(np.float64)

    n_o = libsize_obs
    n_r = libsize_ref

    # log/division of zeros generates -inf/NaN/inf — R prints warnings; we
    # mask them via the `is.finite` filter below, matching R's semantics.
    # `_log2_libm` is used (instead of `np.log2`) because the latter can
    # differ from R by 1 ULP on identical inputs, which would flip rank ties
    # in the trim mask. See the docstring of :func:`_log2_libm`.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = _log2_libm((obs / n_o) / (ref / n_r))
        abs_e = (_log2_libm(obs / n_o) + _log2_libm(ref / n_r)) / 2.0
        v = (n_o - obs) / n_o / obs + (n_r - ref) / n_r / ref

    fin = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > a_cutoff)
    log_r = log_r[fin]
    abs_e = abs_e[fin]
    v = v[fin]

    if log_r.size == 0:
        return 1.0
    if np.max(np.abs(log_r)) < 1e-6:
        return 1.0

    n = log_r.size
    lo_l = int(np.floor(n * logratio_trim)) + 1
    hi_l = n + 1 - lo_l
    lo_s = int(np.floor(n * sum_trim)) + 1
    hi_s = n + 1 - lo_s

    # R's `rank(...)` defaults to `ties.method="average"`; mirror it via
    # scipy.stats.rankdata.
    from scipy.stats import rankdata

    rank_log_r = rankdata(log_r, method="average")
    rank_abs_e = rankdata(abs_e, method="average")

    keep = (
        (rank_log_r >= lo_l)
        & (rank_log_r <= hi_l)
        & (rank_abs_e >= lo_s)
        & (rank_abs_e <= hi_s)
    )

    if do_weighting:
        num = np.nansum(log_r[keep] / v[keep])
        den = np.nansum(1.0 / v[keep])
        f = num / den if den != 0 else np.nan
    else:
        f = np.nanmean(log_r[keep])

    if np.isnan(f):
        f = 0.0
    return float(2.0 ** f)


def compute_offset(
    offset: Optional[np.ndarray], counts: _CountMatrix
) -> np.ndarray:
    """Compute the NB-GAM sequencing-depth offset.

    Port of ``tradeSeq:::.get_offset`` (``R/fitGAM.R`` lines 82-98).

    Parameters
    ----------
    offset
        Optional user-supplied offset vector. When non-``None``, returned
        unchanged (mirrors R's early-return).
    counts
        ``(n_genes, n_cells)`` count matrix (sparse accepted).

    Returns
    -------
    numpy.ndarray
        Length-``n_cells`` offset ``log(colSums(counts) * nf)``. Cells with
        ``libSize == 0`` are set to ``0`` and a warning is emitted (mirrors
        R's ``message(...)``).

    Notes
    -----
    If TMM normalization fails (e.g. matrix is degenerate), the function
    falls back to ``nf = 1`` for all cells, matching R's
    ``try(calcNormFactors(counts), silent=TRUE)`` + ``rep(1, ncol)`` branch.
    """
    if offset is not None:
        return np.asarray(offset)

    dense = _as_dense(counts)
    n_cells = dense.shape[1]

    try:
        nf = calc_norm_factors_tmm(dense)
    except Exception:
        warnings.warn(
            "TMM normalization failed. Will use unnormalized library sizes "
            "as offset.",
            stacklevel=2,
        )
        nf = np.ones(n_cells, dtype=np.float64)

    lib_size = dense.sum(axis=0) * nf

    with np.errstate(divide="ignore"):
        out = np.log(lib_size)

    zero_mask = lib_size == 0
    if np.any(zero_mask):
        warnings.warn(
            "Some library sizes are zero. Offsetting these to 1.",
            stacklevel=2,
        )
        out[zero_mask] = 0.0

    return out
