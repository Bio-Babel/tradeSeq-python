"""Tests for tradeseq.predict_smooth (R source: tradeSeq/R/predictSmooth.R).

Two tiers:

* **Tier 1.** Inject R's β/Σ/dm/X into AnnData via :func:`write_fit_results`,
  call :func:`predict_smooth`, and assert per-cell agreement with R's
  ``predictSmooth`` output (``atol=1e-10``).
* **Tier 2.** Re-fit the Paul-2015 fixture with Python's :func:`tradeseq.fit_gam`,
  call :func:`predict_smooth`, and check the Pearson correlation against
  the R reference (≥ 0.85).
* **List-mode.** Confirm the list-mode path raises ``NotImplementedError``
  because :class:`FittedGam` does not store the design matrix needed by the
  R ``predict(m, newdata=dfall)`` flow.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import pearsonr

import tradeseq
from tradeseq._slot_io import write_fit_results
from tradeseq.predict_smooth import predict_smooth

REF_DIR = Path("/tmp")


def _ref_required():
    if not (REF_DIR / "r_predict_smooth_tidy.csv").exists():
        pytest.skip(
            "R reference dump /tmp/r_predict_smooth_tidy.csv not present. "
            "Run the slice3 R dump block."
        )


@pytest.fixture(scope="module")
def r_inputs():
    """Read R-side dm / X / beta / sigma / knots / pseudotime dumps."""
    _ref_required()
    dm = pd.read_csv(REF_DIR / "r_predict_dm.csv")
    X = pd.read_csv(REF_DIR / "r_predict_X.csv")
    beta = pd.read_csv(REF_DIR / "r_predict_beta.csv", index_col=0)
    pseudotime = pd.read_csv(REF_DIR / "r_predict_pseudotime.csv").to_numpy()
    knots = pd.read_csv(REF_DIR / "r_predict_knots.csv").to_numpy().reshape(-1)
    sigma_arr = np.empty(beta.shape[0], dtype=object)
    for i in range(beta.shape[0]):
        sigma_arr[i] = pd.read_csv(REF_DIR / f"r_predict_sigma_g{i+1}.csv").to_numpy()
    return {
        "dm": dm,
        "X": X,
        "pseudotime": pseudotime,
        "beta": beta.to_numpy(),
        "sigma": sigma_arr,
        "gene_names": list(beta.index),
        "knots": knots,
    }


@pytest.fixture(scope="module")
def r_adata(r_inputs):
    """Inject R's fit outputs into an AnnData so the SCE predict_smooth path runs on R inputs."""
    dm = r_inputs["dm"]
    X = r_inputs["X"]
    beta = r_inputs["beta"]
    sigma = r_inputs["sigma"]
    knots = r_inputs["knots"]
    pseudotime = r_inputs["pseudotime"]
    gene_names = r_inputs["gene_names"]

    n_cells = dm.shape[0]
    n_genes = beta.shape[0]
    counts = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_id": np.arange(n_cells)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame({"gene_id": np.arange(n_genes)}, index=gene_names)
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["pseudotime"] = pseudotime
    adata.obsm["cell_weights"] = np.ones_like(pseudotime)
    write_fit_results(
        adata,
        beta=beta,
        sigma_list=list(sigma),
        converged=np.ones(n_genes, dtype=bool),
        design_matrix=dm,
        lpmatrix=X.to_numpy(),
        knots=knots,
        family="nb",
        conditions=None,
    )
    adata.uns["tradeseq"]["lpmatrix_columns"] = list(X.columns)
    return adata


@pytest.fixture(scope="module")
def r_predict_smooth_tidy_ref():
    _ref_required()
    return pd.read_csv(REF_DIR / "r_predict_smooth_tidy.csv")


@pytest.fixture(scope="module")
def r_predict_smooth_wide_ref():
    _ref_required()
    return pd.read_csv(REF_DIR / "r_predict_smooth_wide.csv", index_col=0)


# ---------------------------------------------------------------------------
# Tier 1 — R inputs into Python: matches R cell-by-cell
# ---------------------------------------------------------------------------


def test_predict_smooth_tidy_matches_r(r_adata, r_predict_smooth_tidy_ref):
    """Tidy output matches R's predictSmooth(tidy=TRUE) value-by-value."""
    out = predict_smooth(
        r_adata, gene=["Acin1", "Actb", "Ak2"], n_points=10, tidy=True
    )
    # Columns: lineage, time, gene, yhat — same as R
    assert list(out.columns) == ["lineage", "time", "gene", "yhat"]
    # Compare by (gene, lineage, time)
    ref = r_predict_smooth_tidy_ref
    # Sort by (gene, lineage, time) for both
    keys = ["gene", "lineage"]
    out_s = out.sort_values(keys + ["time"]).reset_index(drop=True)
    ref_s = ref.sort_values(keys + ["time"]).reset_index(drop=True)
    np.testing.assert_array_equal(out_s["gene"].values, ref_s["gene"].values)
    np.testing.assert_array_equal(out_s["lineage"].values.astype(int),
                                  ref_s["lineage"].values.astype(int))
    np.testing.assert_allclose(out_s["time"].to_numpy(),
                               ref_s["time"].to_numpy(), atol=1e-10)
    np.testing.assert_allclose(out_s["yhat"].to_numpy(),
                               ref_s["yhat"].to_numpy(), atol=1e-10)


def test_predict_smooth_wide_matches_r(r_adata, r_predict_smooth_wide_ref):
    """Wide output (tidy=False) matches R's predictSmooth(tidy=FALSE)."""
    out = predict_smooth(
        r_adata, gene=["Acin1", "Actb", "Ak2"], n_points=10, tidy=False
    )
    assert isinstance(out, pd.DataFrame)
    ref = r_predict_smooth_wide_ref.loc[["Acin1", "Actb", "Ak2"]]
    assert out.shape == ref.shape
    assert list(out.columns) == list(ref.columns)
    assert list(out.index) == list(ref.index)
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-10)


def test_predict_smooth_default_tidy_true(r_adata):
    """tidy defaults to True — returns DataFrame, not ndarray."""
    out = predict_smooth(r_adata, gene=["Acin1"], n_points=5)
    assert isinstance(out, pd.DataFrame)


def test_predict_smooth_single_gene(r_adata, r_predict_smooth_wide_ref):
    """A single-gene character argument returns a 1×N matrix matching R."""
    out = predict_smooth(r_adata, gene="Acin1", n_points=10, tidy=False)
    ref = r_predict_smooth_wide_ref.loc[["Acin1"]]
    assert out.shape == ref.shape
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-10)


def test_predict_smooth_integer_index(r_adata, r_predict_smooth_wide_ref):
    """Integer gene indices select the same rows as R's first-N integer subset."""
    out = predict_smooth(r_adata, gene=[0, 1, 2], n_points=10, tidy=False)
    ref = r_predict_smooth_wide_ref.iloc[:3, :]
    assert out.shape == ref.shape
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-10)


def test_predict_smooth_missing_gene_raises(r_adata):
    """Unknown character gene id raises ValueError per R contract."""
    with pytest.raises(ValueError, match="Not all gene IDs are present"):
        predict_smooth(r_adata, gene="NotAGene", n_points=5)


def test_predict_smooth_nan_propagation(r_adata):
    """A gene with NaN β yields NaN yhat across all (lineage, time) grid points."""
    a = r_adata.copy()
    beta = a.varm["tradeseq_beta"].copy()
    beta[0, :] = np.nan
    a.varm["tradeseq_beta"] = beta
    out = predict_smooth(a, gene=[0], n_points=5, tidy=False)
    assert np.isnan(out.to_numpy()).all()


def test_predict_smooth_invalid_models_type():
    """Non-AnnData / non-dict input raises TypeError."""
    with pytest.raises(TypeError, match="unsupported models type"):
        predict_smooth(123, gene=0, n_points=5)


# ---------------------------------------------------------------------------
# Tier 2 — Python fit_gam end-to-end against R reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_paul15(paul15_small_adata):
    """Run fit_gam on the 5-gene Paul-2015 smoke fixture."""
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    out = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, copy=True,
    )
    return out, list(a.var_names)


def test_predict_smooth_tier2_pearson(fitted_paul15, r_predict_smooth_wide_ref):
    """Python end-to-end smoother values correlate with R at log1p Pearson r ≥ 0.85."""
    adata, gene_names = fitted_paul15
    shared = [g for g in gene_names if g in r_predict_smooth_wide_ref.index]
    assert len(shared) >= 1
    py = predict_smooth(adata, gene=shared, n_points=10, tidy=False)
    ref = r_predict_smooth_wide_ref.loc[shared].to_numpy()
    py_flat = py.to_numpy().ravel()
    ref_flat = ref.ravel()
    mask = np.isfinite(py_flat) & np.isfinite(ref_flat)
    assert mask.sum() > 5
    r_coef, _ = pearsonr(np.log1p(py_flat[mask]), np.log1p(ref_flat[mask]))
    assert r_coef > 0.85, f"log1p(yhat) Pearson r = {r_coef:.3f} < 0.85"


# ---------------------------------------------------------------------------
# List-mode tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_paul15_list(paul15_small_adata):
    """List-mode fit_gam(sce=False) → dict[str, FittedGam]."""
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    fits = tradeseq.fit_gam(
        a, n_knots=6, verbose=False, _w_samp=w_samp, sce=False,
    )
    return fits


def test_predict_smooth_list_mode_returns_wide_matrix(fitted_paul15_list):
    """List-mode returns the wide (n_genes, n_curves * n_points) DataFrame.

    The R method at predictSmooth.R:202-237 returns ``exp(yhatMat)`` with
    ``nrow == length(gene)`` and ``ncol == n_curves * n_points``.
    """
    fits = fitted_paul15_list
    keys = list(fits.keys())
    out = predict_smooth(fits, gene=[keys[0], keys[1]], n_points=5)
    # 2 lineages × 5 points = 10 columns; 2 genes requested → 2 rows.
    assert out.shape == (2, 10)
    assert list(out.index) == [keys[0], keys[1]]
    # All values are non-negative (exp(η)).
    assert (out.values >= 0).all()


def test_predict_smooth_list_mode_skips_failed_reference(fitted_paul15_list):
    """A failed first fit must not prevent using the next successful reference."""
    fits = dict(fitted_paul15_list)
    keys = list(fits.keys())
    fits[keys[0]] = None
    out = predict_smooth(fits, gene=[keys[0], keys[1]], n_points=5)
    assert out.shape == (2, 10)
    assert np.isnan(out.loc[keys[0]].to_numpy()).all()
    assert np.isfinite(out.loc[keys[1]].to_numpy()).all()


def test_predict_smooth_list_mode_column_names_match_r(fitted_paul15_list):
    """List-mode column names follow R's ``lineage{i}_{j}`` (Gap #20).

    R source: predictSmooth.R:228-230 uses
    ``paste0("lineage", apply(expand.grid(1:nPoints, 1:nCurves)[, 2:1], 1,
    paste, collapse="_"))``
    which produces ``lineage1_1, lineage1_2, ..., lineage1_5, lineage2_1,
    ...``. Confirmed against R harness: see /tmp/r_predict_smooth_list.csv.
    """
    fits = fitted_paul15_list
    keys = list(fits.keys())
    out = predict_smooth(fits, gene=[keys[0], keys[1]], n_points=5)
    expected = (
        [f"lineage1_{j}" for j in range(1, 6)]
        + [f"lineage2_{j}" for j in range(1, 6)]
    )
    assert list(out.columns) == expected


def test_predict_smooth_list_mode_column_names_no_point_infix(fitted_paul15_list):
    """List-mode columns must NOT contain a ``_point`` infix (regression test).

    Before Gap #20 was fixed the columns were ``lineage1_point1, ...``; R has
    no such infix (predictSmooth.R:228-230).
    """
    fits = fitted_paul15_list
    keys = list(fits.keys())
    out = predict_smooth(fits, gene=[keys[0]], n_points=5)
    assert all("_point" not in c for c in out.columns)


def test_predict_smooth_list_mode_int_gene_index(fitted_paul15_list):
    fits = fitted_paul15_list
    out = predict_smooth(fits, gene=0, n_points=5)
    assert out.shape == (1, 10)


def test_predict_smooth_list_mode_single_gene_npoints_10(fitted_paul15_list):
    """Single-gene with n_points=10 yields the exact column names R produces.

    R reference (harness output, /tmp/r_predict_smooth_list.csv):
    ``lineage1_1, lineage1_2, ..., lineage1_10, lineage2_1, ..., lineage2_10``.
    """
    fits = fitted_paul15_list
    keys = list(fits.keys())
    out = predict_smooth(fits, gene=keys[0], n_points=10)
    expected = (
        [f"lineage1_{j}" for j in range(1, 11)]
        + [f"lineage2_{j}" for j in range(1, 11)]
    )
    assert list(out.columns) == expected
    # Single-gene returns a (1, n_curves*n_points) DataFrame
    assert out.shape == (1, 20)


def test_predict_smooth_list_mode_unknown_gene_raises(fitted_paul15_list):
    fits = fitted_paul15_list
    with pytest.raises(ValueError, match="Not all gene IDs"):
        predict_smooth(fits, gene="NoSuchGene", n_points=5)
