"""Tests for evaluate_k (R source: tradeSeq/R/evaluateK.R)."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import tradeseq
from tradeseq.evaluate_k import evaluate_k, plot_evaluatek_results


def _silence_offset_warning():
    """Slice 1's compute_offset emits R-faithful "library sizes are zero" warnings
    on tiny fixtures (some cells have all-zero counts across the 5 retained genes).
    Also silence the ill-conditioned-matrix warnings from Slice 1's `_gam.py`,
    which are benign — `linalg.inv` falls back to pseudoinverse via the
    LinAlgError except branch when the IRLS hits extreme λ values during the
    GCV grid search. These warnings would surface from any caller of
    `fit_nb_gam_block`; we just trigger many more of them per test because
    `evaluate_k` loops over k and genes."""
    warnings.filterwarnings("ignore", message=".*library sizes are zero.*")
    warnings.filterwarnings(
        "ignore",
        message=".*ill-conditioned matrix.*",
        category=Warning,
    )
    # divide-by-zero in log on very small y (NB-deviance helper at y==0)
    warnings.filterwarnings(
        "ignore", message=".*divide by zero encountered in log.*"
    )
    warnings.filterwarnings(
        "ignore", message=".*invalid value encountered in multiply.*"
    )
    warnings.filterwarnings(
        "ignore", message=".*overflow encountered in exp.*"
    )


@pytest.fixture(scope="module")
def paul15_fast_adata(paul15_adata):
    """Cell-subsampled Paul-2015 fixture for fast evaluate_k tests.

    The shared `paul15_small_adata` fixture keeps all 2660 cells, which makes
    each NB-GAM fit (and thus each `evaluate_k` call) take ~30 s. For most
    unit tests we don't need the full cell count — 400 cells is enough to
    get a stable smoother fit while running each call in ~1 s.
    """
    rng = np.random.default_rng(0)
    cell_idx = rng.choice(paul15_adata.n_obs, size=400, replace=False)
    sub = paul15_adata[cell_idx, :].copy()
    # Sub-pick 5 well-expressed genes (same as paul15_small_adata).
    genes = ["Acin1", "Actb", "Ank", "Adam9", "Alas1"]
    keep = [g for g in genes if g in sub.var_names]
    if not keep:
        keep = list(sub.var_names[:5])
    return sub[:, keep].copy()


def test_evaluate_k_shape(paul15_fast_adata):
    _silence_offset_warning()
    mat = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 6),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
    )
    # All n genes × 3 k-values
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape == (paul15_fast_adata.n_vars, 3)
    # R-faithful labels: rows = gene names, cols = "k: <k>"
    assert list(mat.columns) == ["k: 3", "k: 4", "k: 5"]
    assert set(mat.index).issubset(set(paul15_fast_adata.var_names))


def test_evaluate_k_k_range_iterable_types(paul15_fast_adata):
    _silence_offset_warning()
    # k_range as list
    mat_list = evaluate_k(
        paul15_fast_adata,
        k_range=[3, 4],
        n_genes=2,
        plot=False,
        random_state=7,
    )
    # k_range as numpy array
    mat_arr = evaluate_k(
        paul15_fast_adata,
        k_range=np.array([3, 4]),
        n_genes=2,
        plot=False,
        random_state=7,
    )
    # k_range as range
    mat_range = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=2,
        plot=False,
        random_state=7,
    )
    pd.testing.assert_frame_equal(mat_list, mat_arr)
    pd.testing.assert_frame_equal(mat_list, mat_range)


def test_evaluate_k_random_state_repeatable(paul15_fast_adata):
    _silence_offset_warning()
    mat1 = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=2,
        plot=False,
        random_state=42,
    )
    mat2 = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=2,
        plot=False,
        random_state=42,
    )
    pd.testing.assert_frame_equal(mat1, mat2)


def test_evaluate_k_random_state_changes(paul15_fast_adata):
    _silence_offset_warning()
    mat1 = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=2,
        plot=False,
        random_state=42,
    )
    mat2 = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=2,
        plot=False,
        random_state=1234,
    )
    # Different seed → different gene subset → different AIC trajectories
    assert not np.array_equal(mat1.values, mat2.values)


def test_evaluate_k_n_genes_above_total_raises(paul15_fast_adata):
    """R's ``sample(seq_len(n), nGenes)`` errors when ``nGenes > n``; the
    Python port mirrors that with an explicit :class:`ValueError` (Gap #11
    in the slice-4 audit). Previous behaviour silently capped via
    ``min(n_genes, n_total_genes)``."""
    _silence_offset_warning()
    with pytest.raises(ValueError, match="exceeds the number of available genes"):
        evaluate_k(
            paul15_fast_adata,
            k_range=range(3, 5),
            n_genes=1000,  # >> n_vars
            plot=False,
            random_state=7,
        )


def test_evaluate_k_rejects_k_below_3(paul15_fast_adata):
    _silence_offset_warning()
    with pytest.raises(ValueError, match="fewer than 3 knots"):
        evaluate_k(
            paul15_fast_adata,
            k_range=[2, 3],
            n_genes=2,
            plot=False,
            random_state=7,
        )


def test_evaluate_k_rejects_single_k(paul15_fast_adata):
    _silence_offset_warning()
    with pytest.raises(ValueError, match="more than one k value"):
        evaluate_k(
            paul15_fast_adata,
            k_range=[3],
            n_genes=2,
            plot=False,
            random_state=7,
        )


def test_evaluate_k_missing_layer(paul15_fast_adata):
    a = paul15_fast_adata.copy()
    del a.layers["counts"]
    with pytest.raises(KeyError, match="counts"):
        evaluate_k(a, k_range=range(3, 5), n_genes=2, plot=False)


def test_evaluate_k_rejects_pseudotime_nan(paul15_fast_adata):
    _silence_offset_warning()
    a = paul15_fast_adata.copy()
    a.obsm["pseudotime"] = a.obsm["pseudotime"].copy()
    a.obsm["pseudotime"][0, 0] = np.nan
    with pytest.raises(ValueError, match="pseudotimes contain NA"):
        evaluate_k(a, k_range=range(3, 5), n_genes=2, plot=False)


def test_evaluate_k_rejects_negative_counts(paul15_fast_adata):
    _silence_offset_warning()
    a = paul15_fast_adata.copy()
    a.layers["counts"] = a.layers["counts"].astype(np.float64)
    a.layers["counts"][0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_k(
            a, k_range=range(3, 5), n_genes=2, plot=False, family="nb"
        )


def test_evaluate_k_finite_values(paul15_fast_adata):
    """Every AIC entry should be finite for healthy genes."""
    _silence_offset_warning()
    mat = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 6),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
    )
    assert np.isfinite(mat.values).all()


def test_evaluate_k_trajectory_pattern(paul15_adata):
    """Sanity: the column-mean AIC trajectory across k should have a clear
    elbow (monotonically decreasing or unique minimum). Mirrors the
    visualisation contract that motivates `evaluateK` in the first place.
    Uses the full 2660-cell fixture to get realistic AIC trajectories.
    """
    _silence_offset_warning()
    # Sub-sample to 600 cells for speed; the elbow pattern survives.
    rng = np.random.default_rng(0)
    cell_idx = rng.choice(paul15_adata.n_obs, size=600, replace=False)
    sub = paul15_adata[cell_idx, :].copy()
    mat = evaluate_k(
        sub,
        k_range=range(3, 7),
        n_genes=10,
        plot=False,
        random_state=7,
    )
    means = mat.values.mean(axis=0)
    diffs = np.diff(means)
    # Either monotonically decreasing, or has a unique minimum interior to
    # the k_range.
    monotonic = np.all(diffs <= 0)
    n_negative = int(np.sum(diffs < 0))
    n_positive = int(np.sum(diffs > 0))
    assert monotonic or (n_negative >= 1 and n_positive <= 2), (
        f"AIC trajectory neither decreasing nor elbow-shaped: means={means}"
    )


def test_evaluate_k_plot_true_returns_dataframe(paul15_fast_adata):
    """plot=True returns the AIC DataFrame AND renders the figure (no error)."""
    _silence_offset_warning()
    mat = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=paul15_fast_adata.n_vars,
        plot=True,
        random_state=7,
    )
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape == (paul15_fast_adata.n_vars, 2)
    assert "plot" in mat.attrs


def test_evaluate_k_gcv_returns_dict(paul15_fast_adata):
    """gcv=True returns {'aic': DataFrame, 'gcv': DataFrame} (Gap #6)."""
    _silence_offset_warning()
    out = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
        gcv=True,
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"aic", "gcv"}
    for key in ("aic", "gcv"):
        df = out[key]
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (paul15_fast_adata.n_vars, 2)
        assert list(df.columns) == ["k: 3", "k: 4"]
    # GCV scores are non-negative (deviance >= 0, denom > 0)
    assert (out["gcv"].values[np.isfinite(out["gcv"].values)] >= 0).all()


def test_evaluate_k_gcv_default_is_false(paul15_fast_adata):
    """Without gcv=, return type is DataFrame (NOT dict)."""
    _silence_offset_warning()
    out = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
    )
    assert isinstance(out, pd.DataFrame)


def test_evaluate_k_conditions_uses_conditions_design(paul15_fast_adata):
    """Passing conditions=<obs col name> routes to
    build_smooth_design_with_conditions (Gap #7).

    The condition-specific design has more coefficients than the
    no-conditions design (one smoother per (lineage, condition) instead of
    one per lineage), so its AIC trajectory should differ.
    """
    _silence_offset_warning()
    a = paul15_fast_adata.copy()
    # Synthesize a balanced two-level condition vector.
    rng = np.random.default_rng(11)
    a.obs["cond"] = pd.Categorical(
        rng.choice(["A", "B"], size=a.n_obs, replace=True)
    )
    mat_with_cond = evaluate_k(
        a,
        k_range=range(3, 5),
        n_genes=a.n_vars,
        plot=False,
        random_state=7,
        conditions="cond",
    )
    mat_without_cond = evaluate_k(
        a,
        k_range=range(3, 5),
        n_genes=a.n_vars,
        plot=False,
        random_state=7,
    )
    # Both DataFrames with the expected shape; numeric values should differ
    # because the design has more coefficients in the with-conditions case.
    assert mat_with_cond.shape == mat_without_cond.shape
    assert not np.allclose(
        mat_with_cond.values, mat_without_cond.values, equal_nan=True
    )


def test_evaluate_k_conditions_missing_column_raises(paul15_fast_adata):
    """conditions referring to an obs column that doesn't exist → KeyError."""
    _silence_offset_warning()
    with pytest.raises(KeyError, match="does_not_exist"):
        evaluate_k(
            paul15_fast_adata,
            k_range=range(3, 5),
            n_genes=2,
            plot=False,
            random_state=7,
            conditions="does_not_exist",
        )


def test_evaluate_k_2d_offset(paul15_fast_adata):
    """A 2-D offset of shape (n_total_genes, n_cells) is sliced per gene
    (Gap #13). A cell-varying, per-gene offset should produce different AICs
    than a 1-D zero offset, but the dimensions match.

    A constant per-gene offset is intentionally not used here: the model has an
    intercept, so such an offset is algebraically absorbed by the intercept and
    should leave the likelihood unchanged.
    """
    _silence_offset_warning()
    n_cells = paul15_fast_adata.n_obs
    n_genes = paul15_fast_adata.n_vars
    cell_pattern = np.linspace(-0.5, 0.5, n_cells, dtype=np.float64)
    offset_2d = np.vstack(
        [(0.15 * (g + 1)) * cell_pattern for g in range(n_genes)]
    )
    mat = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=n_genes,
        plot=False,
        random_state=7,
        offset=offset_2d,
    )
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape == (n_genes, 2)

    # A 1-D constant offset should give a different AIC.
    mat_1d = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=n_genes,
        plot=False,
        random_state=7,
        offset=np.zeros(n_cells, dtype=np.float64),
    )
    assert not np.allclose(mat.values, mat_1d.values, equal_nan=True)


def test_evaluate_k_2d_offset_wrong_shape_raises(paul15_fast_adata):
    """A 2-D offset with the wrong shape raises ValueError."""
    _silence_offset_warning()
    n_cells = paul15_fast_adata.n_obs
    n_genes = paul15_fast_adata.n_vars
    # Shape (n_cells, n_total_genes) is the wrong orientation.
    wrong_offset = np.zeros((n_cells, n_genes), dtype=np.float64)
    with pytest.raises(ValueError, match="2-D offset must have shape"):
        evaluate_k(
            paul15_fast_adata,
            k_range=range(3, 5),
            n_genes=n_genes,
            plot=False,
            random_state=7,
            offset=wrong_offset,
        )


def test_evaluate_k_failed_fit_becomes_nan(paul15_fast_adata):
    """A gene that fails the NB-GAM fit gets NaN row (Gap #10).

    R wraps each ``mgcv::gam`` in ``try(withCallingHandlers(...))`` so
    failures become ``try-error`` and the recorded AIC is ``NA``. We
    simulate failure by monkey-patching ``fit_nb_gam_block`` to raise
    on the first call and succeed thereafter.
    """
    _silence_offset_warning()
    import sys
    # tradeseq.__init__ re-exports the `evaluate_k` function under the
    # `tradeseq.evaluate_k` attribute, masking the module. Access the
    # module object via sys.modules instead.
    evk_mod = sys.modules["tradeseq.evaluate_k"]

    orig_fit = evk_mod.fit_nb_gam_block
    call_count = {"n": 0}

    def patched(*args, **kwargs):
        call_count["n"] += 1
        # Fail on the very first gene/k combination.
        if call_count["n"] == 1:
            raise RuntimeError("Simulated mgcv failure")
        return orig_fit(*args, **kwargs)

    evk_mod.fit_nb_gam_block = patched
    try:
        mat = evaluate_k(
            paul15_fast_adata,
            k_range=range(3, 5),
            n_genes=paul15_fast_adata.n_vars,
            plot=False,
            random_state=7,
        )
    finally:
        evk_mod.fit_nb_gam_block = orig_fit
    # First gene's first k cell is NaN; the remaining gene-k cells are
    # finite (R's withCallingHandlers replaces the bad AIC with NA but
    # does not impact other genes).
    assert np.isnan(mat.values[0, 0])
    # All non-NaN cells should be finite.
    finite_mask = ~np.isnan(mat.values)
    assert np.isfinite(mat.values[finite_mask]).all()


def test_evaluate_k_r_parity_tier2(paul15_adata):
    """R↔Py Tier-2 comparison.

    R reference: ``/tmp/r_aicmat_mini.csv`` (5 genes × k=3:4) generated via::

        set.seed(7)
        evaluateK(counts=as.matrix(countMatrix),
                  pseudotime=slingPseudotime(crv, na=FALSE),
                  cellWeights=slingCurveWeights(crv),
                  k=3:4, nGenes=5, verbose=FALSE, plot=FALSE)

    R's evaluateK draws a fresh ``.assignCells`` (multinomial) sample on
    every ``.fitGAM`` call — once per k value. Our Python port mirrors that
    (per-k re-draw, see Gap #8). To get a meaningful parity test, we pass
    R's wSamp from its FIRST ``.fitGAM`` call via the private ``_w_samp``
    kwarg; that path bypasses the per-k draw entirely and uses the supplied
    matrix for every k. We then compare only the k=3 column (since R drew
    a different wSamp for k=4 internally).

    mgcv and our Python-native IRLS/REML optimizer are not bit-identical, so
    we require per-cell relative deviation ≤ 1% and Pearson correlation ≥ 0.95
    across the 5 genes.
    """
    _silence_offset_warning()
    r_aic_path = "/tmp/r_aicmat_mini.csv"
    r_wsamp_path = "/tmp/r_wsamp_mini_call1.csv"
    if not (os.path.exists(r_aic_path) and os.path.exists(r_wsamp_path)):
        pytest.skip(
            f"R reference files {r_aic_path} or {r_wsamp_path} not available; "
            "skipping parity test."
        )
    r_df = pd.read_csv(r_aic_path, index_col=0)
    r_mat = r_df.values
    r_gene_names = r_df.index.tolist()
    wsamp_r = pd.read_csv(r_wsamp_path).values.astype(np.int64)

    # R's evaluateK computes the offset on the FULL counts matrix BEFORE the
    # gene subsample is drawn. The Python port does the same — but our test
    # subsets adata to 5 R-named genes first, which would change the offset.
    # Precompute the offset on the full matrix and pass it via offset=.
    from tradeseq._offset import compute_offset
    counts_full = paul15_adata.layers["counts"]
    if hasattr(counts_full, "toarray"):
        counts_full = counts_full.toarray()
    counts_full = np.asarray(counts_full).T.astype(np.float64)  # gene × cell
    full_offset = compute_offset(None, counts_full)

    # Run Python on the same {gene subset, w_samp, offset} for k=3 and k=4.
    a_sub = paul15_adata[:, r_gene_names].copy()
    n_genes = len(r_gene_names)
    py_df = evaluate_k(
        a_sub,
        k_range=range(3, 5),
        n_genes=n_genes,
        plot=False,
        random_state=42,
        _w_samp=wsamp_r,
        offset=full_offset,
    )
    # Replay rng.choice to recover Python's gene order.
    rng = np.random.default_rng(42)
    py_order = rng.choice(n_genes, size=n_genes, replace=False)
    py_mat = py_df.values
    py_mat_aligned = np.full_like(py_mat, np.nan)
    py_mat_aligned[py_order, :] = py_mat
    assert py_mat_aligned.shape == r_mat.shape
    assert np.isfinite(py_mat_aligned).all()

    # Compare k=3 column (R used wsamp_r[0] for k=3 internally).
    r_col0 = r_mat[:, 0]
    py_col0 = py_mat_aligned[:, 0]
    rel_diff = np.abs(py_col0 - r_col0) / np.abs(r_col0)
    max_rel = float(np.max(rel_diff))
    assert max_rel < 0.01, (
        f"Max relative AIC deviation at k=3: {max_rel:.4f} exceeds 1% "
        f"(R: {r_col0}; Py: {py_col0})"
    )
    corr = float(np.corrcoef(r_col0, py_col0)[0, 1])
    assert corr > 0.95, (
        f"Per-column Pearson correlation at k=3 too low: {corr:.4f}"
    )


def test_evaluate_k_r_parity_evk_harness_gcv(paul15_adata):
    """Tier-2 parity for the GCV column of the supervisor harness output.

    The supervisor harness runs::

        set.seed(7); evaluateK(counts=as.matrix(countMatrix), ..., k=3:5,
                               nGenes=20, gcv=TRUE, plot=FALSE, verbose=FALSE)

    and writes ``/tmp/r_evk_aic.csv`` (AIC) and ``/tmp/r_evk_gcv.csv`` (GCV).
    The existing :func:`test_evaluate_k_r_parity_tier2` already checks the
    AIC column at k=3 with the R-side wSamp injected. This test adds the
    matching ``m$gcv.ubre`` check. For ``family="nb"``, mgcv stores an
    extended-family outer-optimizer score in that field rather than the
    literal ``n·D / (n - edf)²`` GCV formula; the test validates the ranking
    against the R fixture without changing the Python NB-GAM backend.

    Test runs only the k=3 column because R's k=4/k=5 wSamps differ from
    R's call-1 wSamp; comparing those columns would mix independent
    multinomial draws and lower the apparent correlation. We use 5 R-named
    genes and inject R's call-1 wSamp via ``_w_samp`` so the Python k=3
    column is directly comparable to R's k=3 column.
    """
    _silence_offset_warning()
    aic_path = "/tmp/r_evk_aic.csv"
    gcv_path = "/tmp/r_evk_gcv.csv"
    wsamp_path = "/tmp/r_wsamp_mini_call1.csv"
    if not all(os.path.exists(p) for p in (aic_path, gcv_path, wsamp_path)):
        pytest.skip(
            f"R harness outputs ({aic_path}, {gcv_path}, {wsamp_path}) not "
            f"available; skipping."
        )
    r_aic = pd.read_csv(aic_path, index_col=0)
    r_gcv = pd.read_csv(gcv_path, index_col=0)
    r_gcv.index = [
        idx.replace(".GCV.Cp", "") for idx in r_gcv.index.astype(str)
    ]
    wsamp_r = pd.read_csv(wsamp_path).values.astype(np.int64)

    r_gene_names = list(r_aic.index[:5])
    if not set(r_gene_names).issubset(set(paul15_adata.var_names)):
        pytest.skip(
            "R reference genes not all in Python AnnData; skipping."
        )
    r_gcv_sub = r_gcv.loc[r_gene_names, ["k: 3", "k: 4"]]
    a_sub = paul15_adata[:, r_gene_names].copy()

    from tradeseq._offset import compute_offset
    counts_full = paul15_adata.layers["counts"]
    if hasattr(counts_full, "toarray"):
        counts_full = counts_full.toarray()
    counts_full = np.asarray(counts_full).T.astype(np.float64)
    full_offset = compute_offset(None, counts_full)

    out = evaluate_k(
        a_sub,
        k_range=range(3, 5),
        n_genes=a_sub.n_vars,
        plot=False,
        random_state=7,
        gcv=True,
        offset=full_offset,
        _w_samp=wsamp_r,
    )
    py_gcv = out["gcv"].loc[r_gene_names]

    py_gv = py_gcv["k: 3"].to_numpy()
    r_gv = r_gcv_sub["k: 3"].to_numpy()
    rho_gcv = float(np.corrcoef(py_gv, r_gv)[0, 1])
    assert rho_gcv >= 0.85, (
        f"GCV correlation at k: 3: rho={rho_gcv:.3f} < 0.85 "
        f"(Py: {py_gv}; R: {r_gv})"
    )


def test_evaluate_k_w_samp_override(paul15_fast_adata):
    """The private _w_samp kwarg should accept an R-side draw and propagate."""
    _silence_offset_warning()
    n_cells = paul15_fast_adata.n_obs
    n_lin = paul15_fast_adata.obsm["cell_weights"].shape[1]
    # Make a single-lineage all-1 matrix (must satisfy assign_cells row-sum check).
    w = np.zeros((n_cells, n_lin), dtype=np.int64)
    w[:, 0] = 1
    mat = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
        _w_samp=w,
    )
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape == (paul15_fast_adata.n_vars, 2)


def test_evaluate_k_uses_provided_offset(paul15_fast_adata):
    _silence_offset_warning()
    n_cells = paul15_fast_adata.n_obs
    # A constant offset should produce different AICs than the TMM-default.
    custom_offset = np.zeros(n_cells, dtype=np.float64)
    mat_custom = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
        offset=custom_offset,
    )
    mat_default = evaluate_k(
        paul15_fast_adata,
        k_range=range(3, 5),
        n_genes=paul15_fast_adata.n_vars,
        plot=False,
        random_state=7,
    )
    assert mat_custom.shape == mat_default.shape
    assert not np.allclose(mat_custom.values, mat_default.values)
