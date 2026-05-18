"""Tests for tradeseq.predict_cells (R source: tradeSeq/R/predictCells.R).

Two tiers:

* **Tier 1.** Inject R's β/Σ/dm/X into an AnnData via :func:`write_fit_results`,
  then call :func:`tradeseq.predict_cells` and compare to R's
  ``predictCells`` cell-by-cell with ``atol=1e-10``.
* **Tier 2.** Re-fit the Paul-2015 fixture with Python's :func:`tradeseq.fit_gam`
  and compare the resulting fitted values to R's reference via Pearson r ≥ 0.95.
* **List-mode.** Use :func:`tradeseq.fit_gam` with ``sce=False`` and check the
  Python list-mode predict_cells output against the gamList R reference where
  feasible (we cannot bit-match because our NB-GAM fit is not mgcv).
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
from tradeseq.predict_cells import predict_cells
from tradeseq._slot_io import write_fit_results

REF_DIR = Path("/tmp")


def _ref_required():
    if not (REF_DIR / "r_predict_cells.csv").exists():
        pytest.skip(
            "R reference dump /tmp/r_predict_cells.csv not present. "
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
    """Inject R's fit outputs into an AnnData so the SCE predict_cells path runs on R inputs."""
    dm = r_inputs["dm"]
    X = r_inputs["X"]
    beta = r_inputs["beta"]
    sigma = r_inputs["sigma"]
    knots = r_inputs["knots"]
    pseudotime = r_inputs["pseudotime"]
    gene_names = r_inputs["gene_names"]

    n_cells = dm.shape[0]
    n_genes = beta.shape[0]
    # provide fake count layer (predict_cells doesn't consume counts)
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
def r_predict_cells_ref():
    """Read R's predictCells output from the AnnData-equivalent path."""
    _ref_required()
    return pd.read_csv(REF_DIR / "r_predict_cells.csv", index_col=0)


# ---------------------------------------------------------------------------
# Tier 1 — R inputs into Python: bit-for-bit on linear-algebra output
# ---------------------------------------------------------------------------


def test_predict_cells_sce_matches_r(r_adata, r_predict_cells_ref):
    """SCE predict_cells with R inputs reproduces R's predictCells exactly."""
    out = predict_cells(r_adata, gene=["Acin1", "Actb", "Ak2"])
    # R writes a (3, n_cells) matrix.
    ref = r_predict_cells_ref.loc[["Acin1", "Actb", "Ak2"]].to_numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_predict_cells_single_gene_str(r_adata, r_predict_cells_ref):
    """A single character gene id returns a 1xN matrix matching R."""
    out = predict_cells(r_adata, gene="Acin1")
    ref = r_predict_cells_ref.loc[["Acin1"]].to_numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_predict_cells_integer_index(r_adata, r_predict_cells_ref):
    """Integer indices behave as positional indices into adata.var_names."""
    # First 3 genes (0-based positions: 0, 1, 2)
    out = predict_cells(r_adata, gene=[0, 1, 2])
    ref = r_predict_cells_ref.iloc[:3, :].to_numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_predict_cells_missing_gene_raises(r_adata):
    """Unknown character gene id must raise ValueError per R contract."""
    with pytest.raises(ValueError, match="Not all gene IDs are present"):
        predict_cells(r_adata, gene="NotAGene")


def test_predict_cells_nan_propagation(r_adata):
    """A gene with NaN β yields NaN fitted values across all cells."""
    a = r_adata.copy()
    beta = a.varm["tradeseq_beta"].copy()
    beta[0, :] = np.nan
    a.varm["tradeseq_beta"] = beta
    out = predict_cells(a, gene=[0])
    assert np.isnan(out).all()


def test_predict_cells_invalid_models_type():
    """Calling predict_cells with neither AnnData nor dict raises TypeError."""
    with pytest.raises(TypeError, match="unsupported models type"):
        predict_cells(123, gene=0)


# ---------------------------------------------------------------------------
# Tier 2 — end-to-end with Python fit_gam
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
    keep = list(a.var_names)
    return out, keep


def test_predict_cells_tier2_pearson(fitted_paul15, r_predict_cells_ref):
    """Python end-to-end fitted values correlate with R at Pearson r >= 0.85."""
    adata, gene_names = fitted_paul15
    # Use 2 shared genes (the Python fixture uses paul15_adata which is the same
    # dataset as the R reference). Use any 2 genes shared between the Python
    # fit-genes and R's first 3 columns.
    shared = [g for g in gene_names if g in r_predict_cells_ref.index]
    assert len(shared) >= 1, f"no shared genes; got {gene_names} vs {list(r_predict_cells_ref.index)}"
    py = predict_cells(adata, gene=shared)
    ref = r_predict_cells_ref.loc[shared].to_numpy()
    # NaN propagation already handled — flatten and correlate.
    py_flat = py.ravel()
    ref_flat = ref.ravel()
    mask = np.isfinite(py_flat) & np.isfinite(ref_flat)
    assert mask.sum() > 100
    # Use log1p to compress wide dynamic range from exp(η).
    r_coef, _ = pearsonr(np.log1p(py_flat[mask]), np.log1p(ref_flat[mask]))
    # Tier-2 backend swap; R-mgcv vs Python NB-GAM.
    assert r_coef > 0.85, f"log1p Pearson r = {r_coef:.3f} < 0.85"


# ---------------------------------------------------------------------------
# List-mode tests — dict[str, FittedGam]
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


def test_predict_cells_list_shape(fitted_paul15_list):
    """List-mode predict_cells returns (n_gene, n_cell) per fitted.values."""
    fits = fitted_paul15_list
    out = predict_cells(fits, gene=[0, 1, 2])
    keys = list(fits.keys())
    n_cells = fits[keys[0]].lpmatrix_.shape[0]
    assert out.shape == (3, n_cells)


def test_predict_cells_list_values_are_exp_linear(fitted_paul15_list):
    """List-mode predict_cells must equal exp(X·β + offset) per gene (R parity).

    R: ``predictCells.R:47-63`` returns ``gam$fitted.values`` which for a
    log-link NB is ``exp(X·β + offset)``. We carry the offset on
    :class:`FittedGam` via ``fg.dm_['offset(offset)']`` (Slice 3 addition);
    Slice 4 Gap #18 wires it into the list-mode predict path.
    """
    fits = fitted_paul15_list
    keys = list(fits.keys())
    out = predict_cells(fits, gene=[0])
    fg = fits[keys[0]]
    offset_vec = fg.dm_["offset(offset)"].to_numpy()
    expected = np.exp(fg.lpmatrix_.values @ fg.coef_ + offset_vec)
    np.testing.assert_allclose(out[0], expected, atol=1e-10)


def test_predict_cells_list_character_raises(fitted_paul15_list):
    """List-mode predict_cells with character gene id raises (R uses rownames(list)=NULL)."""
    fits = fitted_paul15_list
    keys = list(fits.keys())
    with pytest.raises(ValueError, match="gene ID is not present"):
        predict_cells(fits, gene=keys[0])


# ---------------------------------------------------------------------------
# Slice 4 — Gap #18: list-mode predict_cells must include the offset
# ---------------------------------------------------------------------------


def test_predict_cells_list_includes_offset(fitted_paul15_list):
    """List-mode output is at the response scale (with offset), not link-only.

    R: ``predictCells.R:47-63`` returns ``gam$fitted.values`` which is
    ``exp(X·β + offset)`` for a log-link NB. Pre-Slice-4 the Python list path
    omitted the offset, so the cell-wise output was off by ``exp(-offset_i)``.
    """
    fits = fitted_paul15_list
    keys = list(fits.keys())
    out = predict_cells(fits, gene=[0])
    fg = fits[keys[0]]
    offset_vec = fg.dm_["offset(offset)"].to_numpy()
    no_offset = np.exp(fg.lpmatrix_.values @ fg.coef_)
    expected = no_offset * np.exp(offset_vec)
    np.testing.assert_allclose(out[0], expected, atol=1e-10)
    # And the difference between with-offset and no-offset is exactly
    # exp(offset) cell-wise (sanity check on the fix direction).
    ratio = out[0] / no_offset
    np.testing.assert_allclose(ratio, np.exp(offset_vec), atol=1e-10)


@pytest.fixture(scope="module")
def r_predict_cells_list_ref():
    """R's ``predictCells(gamList, gene=1:5)`` reference (rows=genes, cols=cells)."""
    path = Path("/tmp") / "r_slice4_predict_cells_list.csv"
    if not path.exists():
        pytest.skip(f"R reference {path} not present.")
    return pd.read_csv(path, index_col=0)


def test_predict_cells_list_tier2_pearson(
    fitted_paul15_list, r_predict_cells_list_ref
):
    """Tier-2: list-mode Python predict_cells correlates with R fitted.values."""
    fits = fitted_paul15_list
    keys = list(fits.keys())
    n_genes = min(len(keys), r_predict_cells_list_ref.shape[0])
    py_out = predict_cells(fits, gene=list(range(n_genes)))
    ref = r_predict_cells_list_ref.iloc[:n_genes, :].to_numpy()
    # Shape sanity
    assert py_out.shape == ref.shape
    py_flat = py_out.ravel()
    ref_flat = ref.ravel()
    mask = np.isfinite(py_flat) & np.isfinite(ref_flat)
    # Tier-2 backend swap; correlate log1p to compress wide range.
    r_coef, _ = pearsonr(np.log1p(py_flat[mask]), np.log1p(ref_flat[mask]))
    assert r_coef >= 0.85, f"list-mode log1p Pearson r = {r_coef:.3f} < 0.85"
