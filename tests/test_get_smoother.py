"""Tests for tradeseq.get_smoother_pvalues / get_smoother_test_stats.

R sources: tradeSeq/R/getSmootherPvalues.R, getSmootherTestStats.R.

The reference values from ``gamList`` are mgcv-fit p-values and Chi-square
test statistics. Our Python NB-GAM uses a different smoothness selector, so
exact bit-match against mgcv is impossible.

We test:

* Structural Tier-1 (shape, column names, index ordering).
* That AnnData input is rejected with a ``TypeError`` pointing to
  ``fit_gam(return_models=True)``.
* That a faulted gene (``FittedGam.converged_ == False``) yields a row of
  NaNs (R: ``rep(NA, nCurves)``).
* Pearson-r relaxation on the Python output is not asserted against R's
  reference because the test fixture uses a different dataset
  (``paul15_small_adata`` vs R's ``gamList`` from ``data(gamList)``).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import tradeseq
from tradeseq.get_smoother import (
    get_smoother_pvalues,
    get_smoother_test_stats,
)
from tradeseq.fit_gam import FittedGam

REF_DIR = Path("/tmp")


def _ref_required():
    if not (REF_DIR / "r_smoother_pvalues.csv").exists():
        pytest.skip(
            "R reference dump /tmp/r_smoother_pvalues.csv not present. "
            "Run the slice3 R dump block."
        )


@pytest.fixture(scope="module")
def fitted_paul15_list(paul15_small_adata):
    """List-mode fit_gam(return_models=True) → dict[str, FittedGam]."""
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    fits = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, return_models=True,
    )
    return fits


@pytest.fixture(scope="module")
def r_smoother_pvalues_ref():
    _ref_required()
    return pd.read_csv(REF_DIR / "r_smoother_pvalues.csv", index_col=0)


@pytest.fixture(scope="module")
def r_smoother_teststats_ref():
    _ref_required()
    return pd.read_csv(REF_DIR / "r_smoother_teststats.csv", index_col=0)


# ---------------------------------------------------------------------------
# Structural / contract tests
# ---------------------------------------------------------------------------


def test_get_smoother_pvalues_shape(fitted_paul15_list):
    """get_smoother_pvalues returns (n_genes, n_smoothers) with gene-name index."""
    fits = fitted_paul15_list
    out = get_smoother_pvalues(fits)
    assert out.shape[0] == len(fits)
    # 2 lineages × 1 smoother each = 2 smoothers
    assert out.shape[1] == 2
    assert list(out.index) == list(fits.keys())


def test_get_smoother_test_stats_shape(fitted_paul15_list):
    """get_smoother_test_stats returns the same shape as get_smoother_pvalues."""
    fits = fitted_paul15_list
    pv = get_smoother_pvalues(fits)
    ts = get_smoother_test_stats(fits)
    assert pv.shape == ts.shape
    assert list(pv.columns) == list(ts.columns)
    assert list(pv.index) == list(ts.index)


def test_get_smoother_pvalues_columns_count_matches_r(
    r_smoother_pvalues_ref, fitted_paul15_list
):
    """Number of smoother columns matches R's gamList reference (= n_lineages)."""
    out = get_smoother_pvalues(fitted_paul15_list)
    assert out.shape[1] == r_smoother_pvalues_ref.shape[1]


def test_get_smoother_test_stats_columns_count_matches_r(
    r_smoother_teststats_ref, fitted_paul15_list
):
    """Number of smoother columns matches R's gamList reference."""
    out = get_smoother_test_stats(fitted_paul15_list)
    assert out.shape[1] == r_smoother_teststats_ref.shape[1]


def test_get_smoother_pvalues_test_stats_share_columns(fitted_paul15_list):
    """The two getters return identical column names (same summary table)."""
    pv = get_smoother_pvalues(fitted_paul15_list)
    ts = get_smoother_test_stats(fitted_paul15_list)
    assert list(pv.columns) == list(ts.columns)


def test_get_smoother_pvalues_values_in_unit_interval(fitted_paul15_list):
    """All p-values must lie in [0, 1] for fitted genes (NaN only when fg is None)."""
    out = get_smoother_pvalues(fitted_paul15_list)
    vals = out.to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert (finite >= 0).all() and (finite <= 1).all()


def test_get_smoother_test_stats_values_non_negative(fitted_paul15_list):
    """Chi-square statistics are non-negative for fitted genes."""
    out = get_smoother_test_stats(fitted_paul15_list)
    vals = out.to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert (finite >= 0).all()


def test_get_smoother_pvalues_anndata_raises(paul15_small_adata):
    """AnnData input rejected with TypeError pointing to fit_gam(return_models=True)."""
    with pytest.raises(TypeError, match="dict\\[str, FittedGam\\]"):
        get_smoother_pvalues(paul15_small_adata)


def test_get_smoother_test_stats_anndata_raises(paul15_small_adata):
    """AnnData input rejected with TypeError pointing to fit_gam(return_models=True)."""
    with pytest.raises(TypeError, match="dict\\[str, FittedGam\\]"):
        get_smoother_test_stats(paul15_small_adata)


def test_get_smoother_pvalues_invalid_type():
    """Non-dict, non-AnnData inputs raise TypeError."""
    with pytest.raises(TypeError, match="expected dict"):
        get_smoother_pvalues(123)


def test_get_smoother_test_stats_invalid_type():
    """Non-dict, non-AnnData inputs raise TypeError."""
    with pytest.raises(TypeError, match="expected dict"):
        get_smoother_test_stats(123)


def test_get_smoother_pvalues_handles_none(fitted_paul15_list):
    """A fit recorded as None (errored) yields a row of NaN p-values."""
    fits = dict(fitted_paul15_list)
    keys = list(fits.keys())
    # Inject one None fit
    fits[keys[0]] = None
    out = get_smoother_pvalues(fits)
    assert np.isnan(out.loc[keys[0]].to_numpy()).all()


def test_get_smoother_test_stats_handles_none(fitted_paul15_list):
    """A fit recorded as None (errored) yields a row of NaN test stats."""
    fits = dict(fitted_paul15_list)
    keys = list(fits.keys())
    fits[keys[0]] = None
    out = get_smoother_test_stats(fits)
    assert np.isnan(out.loc[keys[0]].to_numpy()).all()


def test_get_smoother_pvalues_returns_pvalue_column(fitted_paul15_list):
    """Values must match summary_s_table['p-value'] for each gene."""
    out = get_smoother_pvalues(fitted_paul15_list)
    for gene, fg in fitted_paul15_list.items():
        expected = fg.summary_s_table["p-value"].to_numpy(dtype=float)
        np.testing.assert_allclose(out.loc[gene].to_numpy(), expected, atol=1e-12)


def test_get_smoother_test_stats_returns_chisq_column(fitted_paul15_list):
    """Values must match summary_s_table['Chi.sq'] for each gene."""
    out = get_smoother_test_stats(fitted_paul15_list)
    for gene, fg in fitted_paul15_list.items():
        expected = fg.summary_s_table["Chi.sq"].to_numpy(dtype=float)
        np.testing.assert_allclose(out.loc[gene].to_numpy(), expected, atol=1e-12)
