"""Tests for tradeseq._offset (R source: tradeSeq/R/fitGAM.R:82-98).

Validates parity with ``tradeSeq:::.get_offset`` and the underlying
``edgeR::calcNormFactors(method="TMM")``:

- Bundled-fixture round-trip: ``compute_offset(None, counts)`` matches the
  R-side dump to ``atol=1e-12``.
- A small hand-rolled count matrix whose R-side ``nf`` and ``offset`` are
  embedded as literals reproduces those values bit-faithfully.
- The ``offset != None`` early-return is honoured.
- The zero-column fallback yields ``offset = 0`` for that cell and emits the
  R-style ``UserWarning``.
- Sparse CSR input matches dense input.
- Internal helpers (``_log2_libm``) handle the corner inputs that R's libm
  ``log2`` does (positive, zero, NaN, +/- inf).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from tradeseq._offset import (
    _log2_libm,
    calc_norm_factors_tmm,
    compute_offset,
)


# ---------------------------------------------------------------------------
# Embedded R-side reference values
# ---------------------------------------------------------------------------

# R: `edgeR::calcNormFactors(tradeSeq::countMatrix)` first 6 and last 6 entries.
PAUL15_NF_FIRST6 = np.array(
    [
        2.7235720831986114376377,
        0.7235033238498317365028,
        0.9478652199667029210062,
        0.7263337890805554586393,
        0.9328551520444094968809,
        0.8581169647281255974391,
    ]
)
PAUL15_NF_LAST6 = np.array(
    [
        0.8439788462280339853550,
        0.8482608874266945875320,
        0.8349199805900822113358,
        2.0412066187220698765259,
        0.7948896638818035986773,
        0.9007955732186293529296,
    ]
)
PAUL15_OFFSET_FIRST6 = np.array(
    [
        5.684075511734569374767,
        6.744521861592435385546,
        7.404643197474924498636,
        7.226700668503102775730,
        7.493695252486349822618,
        6.225411317393293231248,
    ]
)


# R: deterministic small example. Source:
# small <- matrix(c(
#     100, 200, 50, 30,
#     50, 100, 25, 15,
#     300, 600, 150, 90,
#     25, 50, 12, 7,
#     400, 800, 200, 120,
#     10, 20, 5, 3,
#     500, 1000, 250, 150,
#     5, 10, 3, 2,
#     1, 2, 1, 1,
#     20, 40, 10, 6
# ), nrow=10, byrow=TRUE)
SMALL_COUNTS = np.array(
    [
        [100, 200, 50, 30],
        [50, 100, 25, 15],
        [300, 600, 150, 90],
        [25, 50, 12, 7],
        [400, 800, 200, 120],
        [10, 20, 5, 3],
        [500, 1000, 250, 150],
        [5, 10, 3, 2],
        [1, 2, 1, 1],
        [20, 40, 10, 6],
    ],
    dtype=np.float64,
)
SMALL_NF_R = np.array(
    [
        1.000590367674801806075,
        1.000590367674801806075,
        0.9998817342699328758471,
        0.9989384496149612546745,
    ]
)
SMALL_OFFSET_R = np.array(
    [
        7.252644145329177227666,
        7.945791325889122624915,
        6.559496964769231830417,
        6.048671341003241330725,
    ]
)


# ---------------------------------------------------------------------------
# TMM tests
# ---------------------------------------------------------------------------

def _counts_genes_x_cells(adata) -> np.ndarray:
    """Helper: AnnData layer is cells x genes; R-source convention is the transpose."""
    layer = adata.layers["counts"]
    if sparse.issparse(layer):
        layer = layer.toarray()
    return np.asarray(layer).T.astype(np.float64)


def test_calc_norm_factors_tmm_paul15_first6(paul15_adata):
    counts = _counts_genes_x_cells(paul15_adata)
    nf = calc_norm_factors_tmm(counts)
    np.testing.assert_allclose(nf[:6], PAUL15_NF_FIRST6, atol=1e-12)


def test_calc_norm_factors_tmm_paul15_last6(paul15_adata):
    counts = _counts_genes_x_cells(paul15_adata)
    nf = calc_norm_factors_tmm(counts)
    np.testing.assert_allclose(nf[-6:], PAUL15_NF_LAST6, atol=1e-12)


def test_calc_norm_factors_tmm_paul15_geometric_mean_one(paul15_adata):
    """R: `f <- f / exp(mean(log(f)))` => geometric mean == 1."""
    counts = _counts_genes_x_cells(paul15_adata)
    nf = calc_norm_factors_tmm(counts)
    np.testing.assert_allclose(np.exp(np.mean(np.log(nf))), 1.0, atol=1e-12)


def test_calc_norm_factors_tmm_small_matrix():
    nf = calc_norm_factors_tmm(SMALL_COUNTS)
    np.testing.assert_allclose(nf, SMALL_NF_R, atol=1e-12)


def test_calc_norm_factors_tmm_sparse_matches_dense(paul15_adata):
    counts = _counts_genes_x_cells(paul15_adata)
    nf_dense = calc_norm_factors_tmm(counts)
    nf_sparse = calc_norm_factors_tmm(sparse.csr_matrix(counts))
    np.testing.assert_allclose(nf_dense, nf_sparse, atol=1e-14)


def test_calc_norm_factors_tmm_drops_allzero_genes():
    """edgeR drops rows that are all zero before computing TMM."""
    counts = SMALL_COUNTS.copy()
    # prepend an all-zero row
    counts = np.vstack([np.zeros(SMALL_COUNTS.shape[1]), counts])
    nf = calc_norm_factors_tmm(counts)
    np.testing.assert_allclose(nf, SMALL_NF_R, atol=1e-12)


def test_calc_norm_factors_tmm_single_sample_returns_one():
    """edgeR collapses to `method="none"` when n_samples == 1 => nf = 1."""
    counts = SMALL_COUNTS[:, :1]
    nf = calc_norm_factors_tmm(counts)
    np.testing.assert_array_equal(nf, np.array([1.0]))


def test_calc_norm_factors_tmm_rejects_nan():
    counts = SMALL_COUNTS.copy()
    counts[0, 0] = np.nan
    with pytest.raises(ValueError, match="NA counts not permitted"):
        calc_norm_factors_tmm(counts)


# ---------------------------------------------------------------------------
# compute_offset tests
# ---------------------------------------------------------------------------

def test_compute_offset_paul15_first6(paul15_adata):
    counts = _counts_genes_x_cells(paul15_adata)
    offset = compute_offset(None, counts)
    np.testing.assert_allclose(offset[:6], PAUL15_OFFSET_FIRST6, atol=1e-12)


def test_compute_offset_paul15_shape(paul15_adata):
    counts = _counts_genes_x_cells(paul15_adata)
    offset = compute_offset(None, counts)
    assert offset.shape == (paul15_adata.n_obs,)


def test_compute_offset_small_matrix():
    offset = compute_offset(None, SMALL_COUNTS)
    np.testing.assert_allclose(offset, SMALL_OFFSET_R, atol=1e-12)


def test_compute_offset_returns_user_offset_unchanged():
    """R: `if (is.null(offset))` is the only path that recomputes."""
    user = np.array([1.0, 2.0, 3.0, 4.0])
    out = compute_offset(user, SMALL_COUNTS)
    np.testing.assert_array_equal(out, user)


def test_compute_offset_zero_library_sets_zero_and_warns():
    """Zero-library cells get offset=0 with a UserWarning (R: message())."""
    counts = SMALL_COUNTS.copy()
    counts[:, 0] = 0  # zero out column 0 => libSize == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        offset = compute_offset(None, counts)
    assert offset[0] == 0.0
    assert any("library sizes are zero" in str(w.message) for w in caught)


def test_compute_offset_sparse_input(paul15_adata):
    counts = _counts_genes_x_cells(paul15_adata)
    off_dense = compute_offset(None, counts)
    off_sparse = compute_offset(None, sparse.csr_matrix(counts))
    np.testing.assert_allclose(off_dense, off_sparse, atol=1e-14)


# ---------------------------------------------------------------------------
# _log2_libm corner-case tests
# ---------------------------------------------------------------------------

def test_log2_libm_positive_matches_math_log2():
    """The wrapper must agree with C libm log2 to the bit."""
    import math
    x = np.array([1.0, 2.0, 8.0, 0.125, (5 / 143) / (10 / 953)])
    out = _log2_libm(x)
    for v, expected in zip(x, [math.log2(v) for v in x]):
        np.testing.assert_equal(_log2_libm(np.array([v]))[0], expected)
    np.testing.assert_array_equal(out, np.array([math.log2(v) for v in x]))


def test_log2_libm_zero_yields_neg_inf():
    """log2(0) == -inf, no exception (mirrors R's `log2(0) == -Inf`)."""
    out = _log2_libm(np.array([0.0]))
    assert out[0] == -np.inf


def test_log2_libm_negative_yields_nan():
    out = _log2_libm(np.array([-1.0]))
    assert np.isnan(out[0])


def test_log2_libm_nan_propagates():
    out = _log2_libm(np.array([np.nan]))
    assert np.isnan(out[0])


def test_log2_libm_infinity():
    out = _log2_libm(np.array([np.inf]))
    assert out[0] == np.inf


def test_log2_libm_fixes_np_log2_tie_breaking():
    """The whole reason this helper exists: `np.log2((1/143)/(2/953))` and
    `np.log2((5/143)/(10/953))` can differ by 1 ULP, which flips a rank tie
    in the TMM trim mask. libm produces identical results for both inputs,
    keeping rank parity with R."""
    a = (1.0 / 143.0) / (2.0 / 953.0)
    b = (5.0 / 143.0) / (10.0 / 953.0)
    assert _log2_libm(np.array([a]))[0] == _log2_libm(np.array([b]))[0]
