"""Tests for tradeseq.start_vs_end_test (R source: tradeSeq/R/startVsEndTest.R:1-269).

Two-tier validation:

* **Tier 1.** The L matrix has rank ``n_lineages`` (full-rank), so the QR
  pivot mismatch that affects :func:`tradeseq.association_test` is absent
  here. The Wald output therefore matches R bit-for-bit (atol=1e-9).
* **Tier 2.** Re-fit with :func:`tradeseq.fit_gam` and compare via Pearson r.
"""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import pearsonr

import tradeseq
from tradeseq._slot_io import write_fit_results
from tradeseq._test_common import (
    build_sve_contrast,
    get_fold_changes,
    n_curves_from_dm,
    run_wald_table,
)

REF_DIR = Path("/tmp/r_refs2")
_R_PIVOT_RE = re.compile(r"s\.(t\d+)\.\.(l\d+)\.(\d+)")


def _ref_required():
    if not (REF_DIR / "r_sve_gl.csv").exists():
        pytest.skip(
            "R reference dumps not present at /tmp/r_refs2/. "
            "Re-run the slice2 R dump block."
        )


def _restore_lpmatrix_colname(c: str) -> str:
    if c == "U":
        return "U"
    m = _R_PIVOT_RE.match(c)
    if m:
        return f"s({m.group(1)}):{m.group(2)}.{m.group(3)}"
    return c


@pytest.fixture(scope="module")
def r_inputs():
    _ref_required()
    dm = pd.read_csv(REF_DIR / "r_dm_assoc.csv", index_col=0)
    X = pd.read_csv(REF_DIR / "r_X_assoc.csv", index_col=0)
    X.columns = [_restore_lpmatrix_colname(c) for c in X.columns]
    pt = pd.read_csv(REF_DIR / "r_pseudotime_assoc.csv", index_col=0).to_numpy()
    beta = pd.read_csv(REF_DIR / "r_beta_assoc.csv", index_col=0)
    sigma_arr = np.empty(beta.shape[0], dtype=object)
    for i in range(beta.shape[0]):
        sigma_arr[i] = pd.read_csv(
            REF_DIR / f"r_sigma_assoc_g{i + 1}.csv"
        ).to_numpy()
    knots = pd.read_csv(REF_DIR / "r_knots_assoc.csv").to_numpy().reshape(-1)
    return {
        "dm": dm, "X": X, "pseudotime": pt,
        "beta": beta.to_numpy(), "sigma": sigma_arr,
        "gene_names": list(beta.index), "knots": knots,
    }


@pytest.fixture(scope="module")
def r_adata(r_inputs):
    dm = r_inputs["dm"]; X = r_inputs["X"]
    n_cells = dm.shape[0]
    n_genes = r_inputs["beta"].shape[0]
    counts = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_id": np.arange(n_cells)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_id": np.arange(n_genes)}, index=r_inputs["gene_names"]
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["pseudotime"] = r_inputs["pseudotime"]
    adata.obsm["cell_weights"] = np.ones_like(r_inputs["pseudotime"])
    write_fit_results(
        adata,
        beta=r_inputs["beta"],
        sigma_list=list(r_inputs["sigma"]),
        converged=np.ones(n_genes, dtype=bool),
        design_matrix=dm,
        lpmatrix=X.to_numpy(),
        knots=r_inputs["knots"],
        family="nb",
        conditions=None,
    )
    adata.uns["tradeseq"]["lpmatrix_columns"] = list(X.columns)
    return adata


@pytest.fixture(scope="module")
def fitted_paul15_g10(paul15_adata):
    """Five well-fit Paul-2015 genes; see association_test.py fixture for rationale."""
    a = paul15_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    genes = ["Acin1", "Actb", "Alas1", "Anapc5", "Apoe"]
    keep = [g for g in genes if g in a.var_names]
    out = tradeseq.fit_gam(
        a, genes=keep, n_knots=6, verbose=False, _w_samp=w_samp, copy=True,
    )
    return out, keep


# ---------------------------------------------------------------------------
# Tier 1 — L-matrix + Wald exactness with R inputs
# ---------------------------------------------------------------------------


def test_L_matrix_sve_matches_r(r_inputs):
    """The SVE L matrix matches R's bit-for-bit (the L is full-rank in p)."""
    dm = r_inputs["dm"]; X = r_inputs["X"]; pt = r_inputs["pseudotime"]
    n_lineages = n_curves_from_dm(dm)
    L = build_sve_contrast(
        lpmatrix=X, dm=dm, pseudotime=pt, conditions=None,
        n_lineages=n_lineages, pseudotime_values=None,
    )
    ref = pd.read_csv(REF_DIR / "r_L_sve.csv").to_numpy()
    np.testing.assert_allclose(L, ref, atol=1e-12)


def test_start_vs_end_test_global_matches_r_exactly(r_adata):
    """Global Wald output matches R bit-for-bit (L is full-rank → no QR pivot ambiguity)."""
    res = tradeseq.start_vs_end_test(r_adata, global_=True, lineages=False)
    ref = pd.read_csv(REF_DIR / "r_sve_g.csv", index_col=0)
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )
    np.testing.assert_allclose(
        res["df"].to_numpy(), ref["df"].to_numpy(), atol=1e-12,
    )
    # pvalues: compare on R-finite & non-zero rows only (CSV truncation).
    py = res["pvalue"].to_numpy(); r_pval = ref["pvalue"].to_numpy()
    mask = (r_pval > 0) & np.isfinite(r_pval)
    np.testing.assert_allclose(py[mask], r_pval[mask], atol=1e-9, rtol=1e-7)
    # signed logFC
    for jj in (1, 2):
        np.testing.assert_allclose(
            res[f"logFClineage{jj}"].to_numpy(),
            ref[f"logFClineage{jj}"].to_numpy(),
            atol=1e-12,
        )


def test_start_vs_end_test_lineages_matches_r_exactly(r_adata):
    """Per-lineage Wald output matches R bit-for-bit."""
    res = tradeseq.start_vs_end_test(r_adata, global_=False, lineages=True)
    ref = pd.read_csv(REF_DIR / "r_sve_l.csv", index_col=0)
    # R writes the columns as waldStat_lineage1, df_lineage1, pvalue_lineage1, ...
    for jj in (1, 2):
        np.testing.assert_allclose(
            res[f"waldStat_lineage{jj}"].to_numpy(),
            ref[f"waldStat_lineage{jj}"].to_numpy(),
            atol=1e-9, rtol=1e-9,
        )
        np.testing.assert_allclose(
            res[f"df_lineage{jj}"].to_numpy(),
            ref[f"df_lineage{jj}"].to_numpy(),
            atol=1e-12,
        )
        py = res[f"pvalue_lineage{jj}"].to_numpy()
        r_pval = ref[f"pvalue_lineage{jj}"].to_numpy()
        mask = (r_pval > 0) & np.isfinite(r_pval)
        np.testing.assert_allclose(
            py[mask], r_pval[mask], atol=1e-9, rtol=1e-7,
        )


def test_start_vs_end_test_global_and_lineages_columns(r_adata):
    """Combined block returns ALL columns in the documented order."""
    res = tradeseq.start_vs_end_test(r_adata, global_=True, lineages=True)
    expected = [
        "waldStat", "df", "pvalue",
        "waldStat_lineage1", "df_lineage1", "pvalue_lineage1",
        "waldStat_lineage2", "df_lineage2", "pvalue_lineage2",
        "logFClineage1", "logFClineage2",
    ]
    assert list(res.columns) == expected


def test_start_vs_end_test_global_columns(r_adata):
    """global=True, lineages=False returns (waldStat, df, pvalue, logFC...)."""
    res = tradeseq.start_vs_end_test(r_adata, global_=True, lineages=False)
    assert list(res.columns) == ["waldStat", "df", "pvalue",
                                  "logFClineage1", "logFClineage2"]


def test_start_vs_end_test_pseudotime_values_matches_r_exactly(r_adata):
    """``pseudotime_values=(t0, t1)`` matches the R reference."""
    res = tradeseq.start_vs_end_test(
        r_adata, global_=True, lineages=False, pseudotime_values=(0.1, 0.5)
    )
    ref = pd.read_csv(REF_DIR / "r_sve_pvalues.csv", index_col=0)
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )
    np.testing.assert_allclose(
        res["logFClineage1"].to_numpy(),
        ref["logFClineage1"].to_numpy(), atol=1e-12,
    )


def test_start_vs_end_test_invalid_pseudotime_values_raises(r_adata):
    """Pseudotime values exceeding the lineage maximum should error."""
    with pytest.raises(ValueError, match="larger than the maximum"):
        tradeseq.start_vs_end_test(
            r_adata, global_=True, pseudotime_values=(0.0, 100.0)
        )


def test_start_vs_end_test_pseudotime_values_wrong_length_raises(r_adata):
    """Non-length-2 pseudotime_values should raise."""
    with pytest.raises(ValueError, match="length-2"):
        tradeseq.start_vs_end_test(r_adata, pseudotime_values=(0.1,))


def test_start_vs_end_test_l2fc_matches_r_exactly(r_adata):
    """l2fc=1 (still QR for SVE) matches R exactly (L is full-rank)."""
    res = tradeseq.start_vs_end_test(
        r_adata, global_=True, lineages=False, l2fc=1.0
    )
    ref = pd.read_csv(REF_DIR / "r_sve_l2fc1.csv", index_col=0)
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )


def test_start_vs_end_test_invalid_branch_raises(r_adata):
    """At least one of global_ or lineages must be True."""
    with pytest.raises(ValueError, match="global_.*lineages"):
        tradeseq.start_vs_end_test(r_adata, global_=False, lineages=False)


def test_start_vs_end_test_nan_propagation(r_adata):
    """A gene with all-NaN β yields NaN for stat/df/pvalue and the FC columns."""
    a = r_adata.copy()
    beta = a.varm["tradeseq_beta"].copy()
    beta[0, :] = np.nan
    a.varm["tradeseq_beta"] = beta
    res = tradeseq.start_vs_end_test(a, global_=True, lineages=True)
    row = res.iloc[0]
    assert np.isnan(row["waldStat"])
    assert np.isnan(row["waldStat_lineage1"])
    assert np.isnan(row["logFClineage1"])


# ---------------------------------------------------------------------------
# Tier 2 — End-to-end with Python's fit_gam
# ---------------------------------------------------------------------------


def test_start_vs_end_test_end_to_end_paul15(fitted_paul15_g10):
    """Python-fit end-to-end output correlates with R's reference (Tier-2)."""
    adata, gene_names = fitted_paul15_g10
    py_res = tradeseq.start_vs_end_test(adata, global_=True, lineages=True)
    r_ref = pd.read_csv(REF_DIR / "r_sve_gl.csv", index_col=0)
    common = [g for g in gene_names if g in r_ref.index]
    py_sub = py_res.loc[common]; r_sub = r_ref.loc[common]
    # logFC sign correlation
    mask = np.isfinite(py_sub["logFClineage1"]) & np.isfinite(r_sub["logFClineage1"])
    r1, _ = pearsonr(
        py_sub["logFClineage1"].to_numpy()[mask],
        r_sub["logFClineage1"].to_numpy()[mask],
    )
    assert r1 > 0.90, f"lineage1 FC Pearson r = {r1:.3f} < 0.90"

    mask = np.isfinite(py_sub["logFClineage2"]) & np.isfinite(r_sub["logFClineage2"])
    r2, _ = pearsonr(
        py_sub["logFClineage2"].to_numpy()[mask],
        r_sub["logFClineage2"].to_numpy()[mask],
    )
    assert r2 > 0.90, f"lineage2 FC Pearson r = {r2:.3f} < 0.90"
