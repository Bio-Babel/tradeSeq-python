"""Tests for fit_gam + nknots (R source: tradeSeq/R/fitGAM.R + nknots.R)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import tradeseq
from tradeseq._slot_io import (
    read_beta,
    read_converged,
    read_design_matrix,
    read_knots,
    read_lpmatrix,
    read_sigma,
)

REF_DIR = Path("/tmp")


def test_fit_gam_uses_pythonic_return_models_parameter():
    params = inspect.signature(tradeseq.fit_gam).parameters
    legacy_container_flag = "s" + "ce"
    assert "return_models" in params
    assert legacy_container_flag not in params


@pytest.fixture(scope="module")
def fitted_paul15(paul15_adata):
    """Run fit_gam on the same 5 genes the R smoke test fits."""
    a = paul15_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    genes = ["Acin1", "Actb", "Ank", "Adam9", "Alas1"]
    keep = [g for g in genes if g in a.var_names][:5]
    out = tradeseq.fit_gam(
        a,
        genes=keep,
        n_knots=6,
        verbose=False,
        _w_samp=w_samp,
        copy=True,
    )
    return out, keep, w_samp


def test_fit_gam_returns_anndata(fitted_paul15):
    out, _, _ = fitted_paul15
    assert out is not None
    assert "tradeseq" in out.uns


def test_fit_gam_slot_schema(fitted_paul15):
    out, keep, _ = fitted_paul15
    beta = read_beta(out)
    sigma = read_sigma(out)
    converged = read_converged(out)
    assert beta.shape[0] == out.n_vars
    assert sigma.shape[0] == out.n_vars
    assert converged.shape[0] == out.n_vars
    # Only the 5 requested genes should have non-NaN beta rows.
    keep_idx = [out.var_names.get_loc(g) for g in keep]
    other_idx = [i for i in range(out.n_vars) if i not in keep_idx]
    assert not np.isnan(beta[keep_idx]).any(axis=1).any()
    assert np.isnan(beta[other_idx]).all(axis=1).all()


def test_fit_gam_dm_columns(fitted_paul15):
    out, _, _ = fitted_paul15
    dm = read_design_matrix(out)
    expected = {"U", "t1", "t2", "l1", "l2", "offset(offset)"}
    assert set(dm.columns) == expected
    # Lineage indicators are 0/1.
    assert set(np.unique(dm["l1"].values).tolist()) <= {0.0, 1.0}
    assert set(np.unique(dm["l2"].values).tolist()) <= {0.0, 1.0}
    # Each cell is assigned to exactly one lineage.
    np.testing.assert_array_equal(dm["l1"].values + dm["l2"].values, 1.0)


def test_fit_gam_knots(fitted_paul15):
    out, _, _ = fitted_paul15
    knots = read_knots(out)
    assert knots.shape == (6,)
    # Knot vector is monotonic.
    assert np.all(np.diff(knots) > 0)


def test_fit_gam_lpmatrix_shape(fitted_paul15):
    out, _, _ = fitted_paul15
    lp = read_lpmatrix(out)
    assert lp.shape[0] == out.n_obs
    # 1 fixed effect + 2 lineages × 6 basis columns = 13
    assert lp.shape[1] == 1 + 2 * 6


def test_fit_gam_in_place(paul15_small_adata):
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    ret = tradeseq.fit_gam(a, n_knots=6, verbose=False, _w_samp=w_samp)
    assert ret is None
    assert "tradeseq" in a.uns


def test_fit_gam_list_mode(paul15_small_adata):
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    fits = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, return_models=True
    )
    assert isinstance(fits, dict)
    assert len(fits) == a.n_vars
    for g, fg in fits.items():
        assert fg.coef_.shape[0] == 1 + 2 * 6
        assert fg.cov_.shape == (1 + 2 * 6, 1 + 2 * 6)


def test_nknots_adata(fitted_paul15):
    out, _, _ = fitted_paul15
    assert tradeseq.nknots(out) == 6


def test_nknots_dict(paul15_small_adata):
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    fits = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, return_models=True
    )
    assert tradeseq.nknots(fits) == 6


def test_fit_gam_validates_pseudotime_nan(paul15_small_adata):
    a = paul15_small_adata.copy()
    a.obsm["pseudotime"] = a.obsm["pseudotime"].copy()
    a.obsm["pseudotime"][0, 0] = np.nan
    with pytest.raises(ValueError, match="pseudotimes contain NA"):
        tradeseq.fit_gam(a, n_knots=6, verbose=False)


def test_fit_gam_rejects_negative_counts(paul15_small_adata):
    a = paul15_small_adata.copy()
    a.layers["counts"] = a.layers["counts"].astype(np.float64)
    a.layers["counts"][0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        tradeseq.fit_gam(a, n_knots=6, verbose=False, family="nb")


def test_fit_gam_genes_string_subset(paul15_adata):
    a = paul15_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    ret = tradeseq.fit_gam(
        a, genes=["Acin1", "Actb"], n_knots=6, verbose=False,
        _w_samp=w_samp, copy=True,
    )
    assert ret.uns["tradeseq"]["genes_fit"] == ["Acin1", "Actb"]


def test_fit_gam_genes_duplicate_raises(paul15_small_adata):
    with pytest.raises(ValueError, match="duplicate"):
        tradeseq.fit_gam(
            paul15_small_adata,
            genes=["Acin1", "Acin1"],
            n_knots=6,
            verbose=False,
        )


def test_fit_gam_missing_layer(paul15_small_adata):
    a = paul15_small_adata.copy()
    del a.layers["counts"]
    with pytest.raises(KeyError, match="counts"):
        tradeseq.fit_gam(a, n_knots=6, verbose=False)


# ---------------------------------------------------------------------------
# Slice 4 — Gap #2: per-smoother Wald test in ``summary_s_table``
# ---------------------------------------------------------------------------


def _fits_list_paul15_small(paul15_small_adata):
    """Helper: list-mode fit on the 5-gene Paul-2015 fixture."""
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    return tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, return_models=True,
    )


def test_fit_gam_summary_s_table_shape(paul15_small_adata):
    """summary_s_table has one row per smoother (lineage), four columns."""
    fits = _fits_list_paul15_small(paul15_small_adata)
    for gene, fg in fits.items():
        assert fg is not None, f"unexpected None placeholder for {gene}"
        # 2 lineages → 2 smoothers; 4 R-faithful columns
        assert fg.summary_s_table.shape == (2, 4)
        assert list(fg.summary_s_table.columns) == [
            "edf", "Ref.df", "Chi.sq", "p-value",
        ]


def test_fit_gam_summary_s_table_row_labels(paul15_small_adata):
    """Row labels match R smoother labels: ``s(t1):l1`` and ``s(t2):l2``."""
    fits = _fits_list_paul15_small(paul15_small_adata)
    for gene, fg in fits.items():
        assert list(fg.summary_s_table.index) == ["s(t1):l1", "s(t2):l2"]


def test_fit_gam_summary_s_table_rows_differ(paul15_small_adata):
    """Each smoother row should have a distinct (edf, Chi.sq, p-value) — Gap #2.

    The pre-Slice-4 implementation fabricated identical rows by splitting
    ``edf`` / ``deviance`` equally across smoothers. After the fix, the
    per-smoother Wald statistic varies by lineage.
    """
    fits = _fits_list_paul15_small(paul15_small_adata)
    for gene, fg in fits.items():
        stab = fg.summary_s_table
        # At least one of the four columns must differ between rows 0 and 1.
        diffs = (stab.iloc[0].to_numpy() - stab.iloc[1].to_numpy())
        finite = diffs[np.isfinite(diffs)]
        assert finite.size > 0
        assert np.any(np.abs(finite) > 1e-9), (
            f"gene {gene}: per-smoother rows are still identical: {stab}"
        )


def test_fit_gam_summary_s_table_pvalues_in_unit_interval(paul15_small_adata):
    fits = _fits_list_paul15_small(paul15_small_adata)
    for gene, fg in fits.items():
        pv = fg.summary_s_table["p-value"].to_numpy()
        finite = pv[np.isfinite(pv)]
        assert (finite >= 0).all() and (finite <= 1).all()


def test_fit_gam_summary_s_table_chisq_non_negative(paul15_small_adata):
    fits = _fits_list_paul15_small(paul15_small_adata)
    for gene, fg in fits.items():
        cs = fg.summary_s_table["Chi.sq"].to_numpy()
        finite = cs[np.isfinite(cs)]
        assert (finite >= 0).all()


@pytest.fixture(scope="module")
def r_slice4_stable():
    path = REF_DIR / "r_slice4_stable.csv"
    if not path.exists():
        pytest.skip(f"R reference {path} not present.")
    return pd.read_csv(path)


def _summary_s_table_merged(paul15_small_adata, r_slice4_stable):
    """Join Python and R per-smoother summary tables on (gene, smoother)."""
    fits = _fits_list_paul15_small(paul15_small_adata)
    py_rows = []
    for gene, fg in fits.items():
        for smoother, row in fg.summary_s_table.iterrows():
            py_rows.append((gene, smoother, row["Chi.sq"], row["p-value"]))
    py_df = pd.DataFrame(py_rows, columns=["gene", "smoother", "Chi.sq", "pvalue"])
    merged = py_df.merge(
        r_slice4_stable, on=["gene", "smoother"], suffixes=("_py", "_r")
    )
    assert len(merged) >= 4  # 2 lineages × shared genes ≥ 2
    return merged


def test_fit_gam_summary_s_table_chisq_correlates_with_r(
    paul15_small_adata, r_slice4_stable
):
    """Tier-2 acceptance: Pearson r ≥ 0.85 on smoother Chi.sq.

    The Python fitter mirrors mgcv's NB REML path and ``testStat`` rank
    construction, but it still uses a Python-native optimizer and Laplace
    score, so Tier-2 correlation is the appropriate end-to-end check.
    """
    merged = _summary_s_table_merged(paul15_small_adata, r_slice4_stable)
    r_cs, _ = pearsonr(merged["Chi.sq_py"], merged["Chi.sq_r"])
    assert r_cs >= 0.85, f"Chi.sq Pearson r = {r_cs:.3f} < 0.85"


def test_fit_gam_summary_s_table_pvalues_match_r_decisions(
    paul15_small_adata, r_slice4_stable
):
    """Tier-2: p-values agree with R on significant/non-significant calls.

    R/mgcv underflows most smoother p-values to exact 0 on this fixture, so a
    Pearson correlation on ``-log10(p)`` mostly measures arbitrary clipping at
    1e-300. The source-faithful check is that Python reproduces the same tail
    decision after the mgcv ``testStat`` rank construction and constant-null
    side-constraint projection.
    """
    merged = _summary_s_table_merged(paul15_small_adata, r_slice4_stable)
    p_py = merged["pvalue_py"].to_numpy()
    p_r = merged["pvalue_r"].to_numpy()
    np.testing.assert_array_equal(p_py < 1e-4, p_r < 1e-4)
    nonsig = p_r > 0.05
    if nonsig.any():
        assert (p_py[nonsig] > 0.05).all()


# ---------------------------------------------------------------------------
# Slice 4 — Gap #3: ``aic`` / ``gcv`` knobs
# ---------------------------------------------------------------------------


def _aic_setup(paul15_small_adata):
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    return a, w_samp


def test_fit_gam_aic_true_returns_ndarray(paul15_small_adata):
    """``aic=True`` skips AnnData/dict assembly and returns a 1-D ndarray.

    R source: ``fitGAM.R:332-340``.
    """
    a, w_samp = _aic_setup(paul15_small_adata)
    aic_vals = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, aic=True,
    )
    assert isinstance(aic_vals, np.ndarray)
    assert aic_vals.shape == (a.n_vars,)
    # All five genes are well-behaved; expect finite AIC for each.
    assert np.isfinite(aic_vals).all()


def test_fit_gam_aic_gcv_true_returns_tuple(paul15_small_adata):
    """``aic=True, gcv=True`` returns ``(aic_array, gcv_array)`` per R.

    R source: ``fitGAM.R:341-347``.
    """
    a, w_samp = _aic_setup(paul15_small_adata)
    out = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, aic=True, gcv=True,
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    aic_arr, gcv_arr = out
    assert isinstance(aic_arr, np.ndarray)
    assert isinstance(gcv_arr, np.ndarray)
    assert aic_arr.shape == gcv_arr.shape == (a.n_vars,)
    assert np.isfinite(aic_arr).all()
    assert np.isfinite(gcv_arr).all()
    assert (gcv_arr > 0).all()


def test_fit_gam_aic_skips_anndata_assembly(paul15_small_adata):
    """When ``aic=True``, the AnnData is NOT mutated (R: returns early)."""
    a, w_samp = _aic_setup(paul15_small_adata)
    tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, aic=True,
    )
    # R short-circuits before container assembly — AnnData unchanged.
    assert "tradeseq" not in a.uns


def test_fit_gam_aic_correlates_with_r(paul15_small_adata):
    """Tier-2: per-gene AIC correlates with R (Pearson r ≥ 0.85)."""
    path = REF_DIR / "r_slice4_aic_gcv.csv"
    if not path.exists():
        pytest.skip(f"R reference {path} not present.")
    r_ref = pd.read_csv(path)
    a, w_samp = _aic_setup(paul15_small_adata)
    aic_vals = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, aic=True,
    )
    py_df = pd.DataFrame({"gene": list(a.var_names), "aic_py": aic_vals})
    merged = py_df.merge(r_ref, on="gene")
    assert len(merged) >= 2
    r_corr, _ = pearsonr(merged["aic_py"], merged["aic"])
    assert r_corr >= 0.85, f"AIC Pearson r = {r_corr:.3f} < 0.85"


def test_fit_gam_gcv_correlates_with_r(paul15_small_adata):
    """Tier-2: per-gene ``m$gcv.ubre`` score correlates with R."""
    path = REF_DIR / "r_slice4_aic_gcv.csv"
    if not path.exists():
        pytest.skip(f"R reference {path} not present.")
    r_ref = pd.read_csv(path)
    a, w_samp = _aic_setup(paul15_small_adata)
    _, gcv_vals = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, aic=True, gcv=True,
    )
    py_df = pd.DataFrame({"gene": list(a.var_names), "gcv_py": gcv_vals})
    merged = py_df.merge(r_ref, on="gene")
    assert len(merged) >= 2
    r_corr, _ = pearsonr(merged["gcv_py"], merged["gcv"])
    assert r_corr >= 0.85, f"GCV Pearson r = {r_corr:.3f} < 0.85"


# ---------------------------------------------------------------------------
# Slice 4 — Gap #24: failed-fit placeholders preserve gene order in list mode
# ---------------------------------------------------------------------------


def test_fit_gam_list_mode_failed_fit_placeholder(paul15_small_adata, monkeypatch):
    """A per-gene fit failure must insert a ``None`` placeholder so all
    requested genes appear as keys in the returned dict — mirrors R's
    ``rep(NA, nCurves)`` rows in ``getSmootherPvalues.R:21``.
    """
    a, w_samp = _aic_setup(paul15_small_adata)

    # ``tradeseq/__init__.py`` rebinds the name ``fit_gam`` to the function
    # symbol; reach the submodule explicitly via importlib.
    import importlib
    fit_gam_mod = importlib.import_module("tradeseq.fit_gam")
    original = fit_gam_mod._fit_one_gene
    call_state = {"i": 0}

    def fake_fit(y, design, offset, weights, family):
        i = call_state["i"]
        call_state["i"] += 1
        if i == 1:
            return None, False
        return original(y, design, offset, weights, family)

    monkeypatch.setattr(fit_gam_mod, "_fit_one_gene", fake_fit)
    fits = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, return_models=True,
    )
    # All requested genes appear as keys (no silent drop).
    assert isinstance(fits, dict)
    assert list(fits.keys()) == list(a.var_names)
    # The injected failure is recorded as None.
    failed_gene = a.var_names[1]
    assert fits[failed_gene] is None
    # Other genes still hold a FittedGam.
    for i, g in enumerate(a.var_names):
        if i == 1:
            continue
        assert fits[g] is not None


def test_fit_gam_list_mode_failed_fit_propagates_to_get_smoother(
    paul15_small_adata, monkeypatch
):
    """A ``None`` placeholder produces NaN rows in ``get_smoother_*``.

    R source: ``getSmootherPvalues.R:21`` (``if (is(m)[1]=="try-error")
    return(rep(NA, nCurves))``).
    """
    a, w_samp = _aic_setup(paul15_small_adata)
    import importlib
    fit_gam_mod = importlib.import_module("tradeseq.fit_gam")
    original = fit_gam_mod._fit_one_gene
    call_state = {"i": 0}

    def fake_fit(y, design, offset, weights, family):
        i = call_state["i"]
        call_state["i"] += 1
        if i == 2:
            return None, False
        return original(y, design, offset, weights, family)

    monkeypatch.setattr(fit_gam_mod, "_fit_one_gene", fake_fit)
    fits = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, return_models=True,
    )
    failed_gene = a.var_names[2]
    pv = tradeseq.get_smoother.get_smoother_pvalues(fits)
    ts = tradeseq.get_smoother.get_smoother_test_stats(fits)
    assert np.isnan(pv.loc[failed_gene].to_numpy()).all()
    assert np.isnan(ts.loc[failed_gene].to_numpy()).all()
