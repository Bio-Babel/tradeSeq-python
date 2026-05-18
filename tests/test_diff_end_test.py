"""Tests for tradeseq.diff_end_test (R source: tradeSeq/R/diffEndTest.R:1-237).

The endpoint contrast L is rank ``n_pairs`` (full-column-rank), so the QR
pivot mismatch that affects :func:`tradeseq.association_test` is absent.
Tier-1 tests therefore assert bit-for-bit Wald-output parity with R.
"""

from __future__ import annotations

import re
import warnings
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
    build_diffend_pair_contrasts,
    get_fold_changes,
    n_curves_from_dm,
    run_wald_table,
)

REF_DIR = Path("/tmp/r_refs2")
_R_PIVOT_RE = re.compile(r"s\.(t\d+)\.\.(l\d+)\.(\d+)")


def _ref_required():
    if not (REF_DIR / "r_diffend_g.csv").exists():
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
    n_cells = dm.shape[0]; n_genes = r_inputs["beta"].shape[0]
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
        design_matrix=dm, lpmatrix=X.to_numpy(),
        knots=r_inputs["knots"], family="nb", conditions=None,
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


def test_L_matrix_diffend_matches_r(r_inputs):
    """The DiffEnd L matrix matches R's bit-for-bit (full column rank)."""
    dm = r_inputs["dm"]; X = r_inputs["X"]; pt = r_inputs["pseudotime"]
    n_lineages = n_curves_from_dm(dm)
    L, pairs = build_diffend_pair_contrasts(
        lpmatrix=X, dm=dm, pseudotime=pt, conditions=None, n_lineages=n_lineages,
    )
    ref = pd.read_csv(REF_DIR / "r_L_diff.csv").to_numpy()
    np.testing.assert_allclose(L, ref, atol=1e-12)
    # combn(2, 2) yields a single (1, 2) pair.
    assert pairs == [(1, 2)]


def test_diff_end_test_global_matches_r_exactly(r_adata):
    """global=True, pairwise=False on a 2-lineage fixture matches R bit-for-bit."""
    res = tradeseq.diff_end_test(r_adata, global_=True, pairwise=False)
    ref = pd.read_csv(REF_DIR / "r_diffend_g.csv", index_col=0)
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )
    np.testing.assert_allclose(
        res["df"].to_numpy(), ref["df"].to_numpy(), atol=1e-12,
    )
    py = res["pvalue"].to_numpy(); r_pval = ref["pvalue"].to_numpy()
    mask = (r_pval > 0) & np.isfinite(r_pval)
    np.testing.assert_allclose(py[mask], r_pval[mask], atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(
        res["logFC1_2"].to_numpy(), ref["logFC1_2"].to_numpy(), atol=1e-12,
    )


def test_diff_end_test_pairwise_demoted_for_two_lineages(r_adata):
    """With 2 lineages, pairwise=True is silently coerced to False (R parity)."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        res = tradeseq.diff_end_test(r_adata, global_=True, pairwise=True)
    # R emits a message(), we emit a warning.
    assert any("Only two lineages" in str(w.message) for w in captured)
    # Output is identical to pairwise=False (no per-pair columns).
    ref = tradeseq.diff_end_test(r_adata, global_=True, pairwise=False)
    pd.testing.assert_frame_equal(res, ref)


def test_diff_end_test_l2fc_matches_r_exactly(r_adata):
    """l2fc=1 (still QR; L is single-column rank-1) matches R bit-for-bit."""
    res = tradeseq.diff_end_test(r_adata, global_=True, pairwise=False, l2fc=1.0)
    ref = pd.read_csv(REF_DIR / "r_diffend_l2fc1.csv", index_col=0)
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )


def test_diff_end_test_global_columns(r_adata):
    """global=True, pairwise=False on a 2-lineage fixture has 4 columns."""
    res = tradeseq.diff_end_test(r_adata, global_=True, pairwise=False)
    assert list(res.columns) == ["waldStat", "df", "pvalue", "logFC1_2"]


def test_diff_end_test_invalid_branch_raises(r_adata):
    """At least one of global_ or pairwise must be True."""
    with pytest.raises(ValueError, match="global_.*pairwise"):
        tradeseq.diff_end_test(r_adata, global_=False, pairwise=False)


def test_diff_end_test_single_lineage_raises():
    """A single-lineage AnnData must raise (R parity)."""
    n_cells = 50
    counts = sparse.csr_matrix((n_cells, 1), dtype=np.float32)
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"x": np.arange(n_cells)}, index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame({"g": [0]}, index=["g1"]),
    )
    adata.obsm["pseudotime"] = np.linspace(0, 1, n_cells)[:, None]
    adata.obsm["cell_weights"] = np.ones((n_cells, 1))
    dm = pd.DataFrame(
        {"U": np.ones(n_cells), "offset(offset)": np.zeros(n_cells),
         "t1": adata.obsm["pseudotime"][:, 0], "l1": np.ones(n_cells)}
    )
    write_fit_results(
        adata,
        beta=np.zeros((1, 2)),
        sigma_list=[np.eye(2)],
        converged=np.array([True]),
        design_matrix=dm,
        lpmatrix=np.zeros((n_cells, 2)),
        knots=np.array([0.0, 1.0]),
        family="nb", conditions=None,
    )
    adata.uns["tradeseq"]["lpmatrix_columns"] = ["U", "s(t1):l1.1"]
    with pytest.raises(ValueError, match="only one lineage"):
        tradeseq.diff_end_test(adata, global_=True)


def test_diff_end_test_nan_propagation(r_adata):
    """A gene with all-NaN β returns NaN for stat/df/pvalue and FC."""
    a = r_adata.copy()
    beta = a.varm["tradeseq_beta"].copy()
    beta[0, :] = np.nan
    a.varm["tradeseq_beta"] = beta
    res = tradeseq.diff_end_test(a, global_=True)
    row = res.iloc[0]
    assert np.isnan(row["waldStat"])
    assert np.isnan(row["df"])
    assert np.isnan(row["logFC1_2"])


# ---------------------------------------------------------------------------
# Three-lineage exact-match against R reference (combn(3, 2) = (1,2), (1,3), (2,3))
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_inputs_3lin():
    """Read R β/Σ/dm/X/pt for the 3-lineage diffEndTest reference."""
    _ref_required()
    if not (REF_DIR / "r_dm_3lin.csv").exists():
        pytest.skip("3-lineage R dumps not present at /tmp/r_refs2/.")
    dm = pd.read_csv(REF_DIR / "r_dm_3lin.csv", index_col=0)
    X = pd.read_csv(REF_DIR / "r_X_3lin.csv", index_col=0)
    X.columns = [_restore_lpmatrix_colname(c) for c in X.columns]
    pt = pd.read_csv(REF_DIR / "r_pseudotime_3lin.csv", index_col=0).to_numpy()
    beta = pd.read_csv(REF_DIR / "r_beta_3lin.csv", index_col=0)
    sigma_arr = np.empty(beta.shape[0], dtype=object)
    for i in range(beta.shape[0]):
        sigma_arr[i] = pd.read_csv(
            REF_DIR / f"r_sigma_3lin_g{i + 1}.csv"
        ).to_numpy()
    knots = pd.read_csv(REF_DIR / "r_knots_3lin.csv").to_numpy().reshape(-1)
    return {
        "dm": dm, "X": X, "pseudotime": pt,
        "beta": beta.to_numpy(), "sigma": sigma_arr,
        "gene_names": list(beta.index), "knots": knots,
    }


@pytest.fixture(scope="module")
def r_adata_3lin(r_inputs_3lin):
    dm = r_inputs_3lin["dm"]; X = r_inputs_3lin["X"]
    n_cells = dm.shape[0]; n_genes = r_inputs_3lin["beta"].shape[0]
    counts = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_id": np.arange(n_cells)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_id": np.arange(n_genes)}, index=r_inputs_3lin["gene_names"]
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["pseudotime"] = r_inputs_3lin["pseudotime"]
    adata.obsm["cell_weights"] = np.ones_like(r_inputs_3lin["pseudotime"])
    write_fit_results(
        adata,
        beta=r_inputs_3lin["beta"],
        sigma_list=list(r_inputs_3lin["sigma"]),
        converged=np.ones(n_genes, dtype=bool),
        design_matrix=dm, lpmatrix=X.to_numpy(),
        knots=r_inputs_3lin["knots"], family="nb", conditions=None,
    )
    adata.uns["tradeseq"]["lpmatrix_columns"] = list(X.columns)
    return adata


def test_L_matrix_diffend_three_lineages_columns(r_inputs_3lin):
    """combn(3, 2) yields three pairs (1,2), (1,3), (2,3) and L shape is correct."""
    dm = r_inputs_3lin["dm"]; X = r_inputs_3lin["X"]; pt = r_inputs_3lin["pseudotime"]
    L, pairs = build_diffend_pair_contrasts(
        lpmatrix=X, dm=dm, pseudotime=pt, conditions=None, n_lineages=3,
    )
    assert pairs == [(1, 2), (1, 3), (2, 3)]
    assert L.shape == (X.shape[1], 3)


def test_diff_end_test_three_lineages_pairwise_matches_r_exactly(r_adata_3lin):
    """3-lineage pairwise output matches R bit-for-bit (full-rank L)."""
    res = tradeseq.diff_end_test(r_adata_3lin, global_=True, pairwise=True)
    ref = pd.read_csv(REF_DIR / "r_diffend_3lin.csv", index_col=0)
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )
    for pair in ("1vs2", "1vs3", "2vs3"):
        np.testing.assert_allclose(
            res[f"waldStat_{pair}"].to_numpy(),
            ref[f"waldStat_{pair}"].to_numpy(), atol=1e-9, rtol=1e-9,
        )
    for fc in ("logFC1_2", "logFC1_3", "logFC2_3"):
        np.testing.assert_allclose(
            res[fc].to_numpy(), ref[fc].to_numpy(), atol=1e-12,
        )


def test_diff_end_test_three_lineages_columns(r_adata_3lin):
    """3-lineage diff_end_test global+pairwise has the documented column ordering."""
    res = tradeseq.diff_end_test(r_adata_3lin, global_=True, pairwise=True)
    assert list(res.columns) == [
        "waldStat", "df", "pvalue",
        "waldStat_1vs2", "df_1vs2", "pvalue_1vs2",
        "waldStat_1vs3", "df_1vs3", "pvalue_1vs3",
        "waldStat_2vs3", "df_2vs3", "pvalue_2vs3",
        "logFC1_2", "logFC1_3", "logFC2_3",
    ]


# ---------------------------------------------------------------------------
# Conditions branch — L-column count drives the n==1 / n==2 guards.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_inputs_cond():
    """Read the conditions-aware fixture (2 lineages × 2 conditions)."""
    _ref_required()
    if not (REF_DIR / "r_dm_conditions.csv").exists():
        pytest.skip("conditions R dumps not present at /tmp/r_refs2/.")
    dm = pd.read_csv(REF_DIR / "r_dm_conditions.csv", index_col=0)
    X = pd.read_csv(REF_DIR / "r_X_conditions.csv", index_col=0)
    cond_re = re.compile(r"s\.(t\d+)\.\.(l\d+_\d+)\.(\d+)")
    def _restore_cond_colname(c: str) -> str:
        if c == "U":
            return "U"
        m = cond_re.match(c)
        if m:
            return f"s({m.group(1)}):{m.group(2)}.{m.group(3)}"
        return c
    X.columns = [_restore_cond_colname(c) for c in X.columns]
    pt = pd.read_csv(
        REF_DIR / "r_pseudotime_conditions.csv", index_col=0
    ).to_numpy()
    beta = pd.read_csv(REF_DIR / "r_beta_conditions.csv", index_col=0)
    sigma_arr = np.empty(beta.shape[0], dtype=object)
    for i in range(beta.shape[0]):
        sigma_arr[i] = pd.read_csv(
            REF_DIR / f"r_sigma_conditions_g{i + 1}.csv"
        ).to_numpy()
    knots = pd.read_csv(REF_DIR / "r_knots_conditions.csv").to_numpy().reshape(-1)
    conds_csv = pd.read_csv(REF_DIR / "r_conditions_vec.csv")
    cond_col = conds_csv.iloc[:, 0].astype(str).to_numpy()
    conditions = pd.Categorical(cond_col, categories=sorted(set(cond_col)))
    return {
        "dm": dm, "X": X, "pseudotime": pt,
        "beta": beta.to_numpy(), "sigma": sigma_arr,
        "gene_names": list(beta.index), "knots": knots,
        "conditions": conditions,
    }


@pytest.fixture(scope="module")
def r_adata_cond(r_inputs_cond):
    dm = r_inputs_cond["dm"]; X = r_inputs_cond["X"]
    n_cells = dm.shape[0]; n_genes = r_inputs_cond["beta"].shape[0]
    counts = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_id": np.arange(n_cells)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_id": np.arange(n_genes)}, index=r_inputs_cond["gene_names"]
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["pseudotime"] = r_inputs_cond["pseudotime"]
    adata.obsm["cell_weights"] = np.ones_like(r_inputs_cond["pseudotime"])
    write_fit_results(
        adata,
        beta=r_inputs_cond["beta"],
        sigma_list=list(r_inputs_cond["sigma"]),
        converged=np.ones(n_genes, dtype=bool),
        design_matrix=dm, lpmatrix=X.to_numpy(),
        knots=r_inputs_cond["knots"], family="nb",
        conditions=r_inputs_cond["conditions"],
    )
    adata.uns["tradeseq"]["lpmatrix_columns"] = list(X.columns)
    return adata


def test_diff_end_test_conditions_no_pairwise_warning(r_adata_cond):
    """2 lineages × 2 conditions: nCurves (l-cols) = 4 → no pairwise-skip warning.

    Mirrors the bug fix described in gap #16: with conditions present and 2
    underlying lineages, R's ``nCurves = length(grep("l[1-9]"))`` is 4, so the
    ``if (nCurves == 2 & pairwise) {pairwise <- FALSE}`` guard does NOT fire.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        res = tradeseq.diff_end_test(
            r_adata_cond, global_=True, pairwise=True
        )
    # No "Only two lineages" warning should fire.
    assert not any("Only two lineages" in str(w.message) for w in captured), \
        f"Unexpected pairwise-skip warning: {[str(w.message) for w in captured]}"
    # Pairwise columns must be present (one pair from combn(2, 2)).
    assert "waldStat_1vs2" in res.columns
    assert "df_1vs2" in res.columns
    assert "pvalue_1vs2" in res.columns
    assert "logFC1_2" in res.columns


def test_diff_end_test_conditions_pairwise_columns(r_adata_cond):
    """With conditions + 2 lineages: pairwise=True yields columns for the (1,2) pair."""
    res = tradeseq.diff_end_test(
        r_adata_cond, global_=True, pairwise=True
    )
    expected = [
        "waldStat", "df", "pvalue",
        "waldStat_1vs2", "df_1vs2", "pvalue_1vs2",
        "logFC1_2",
    ]
    assert list(res.columns) == expected


def test_diff_end_test_conditions_matches_r_exactly(r_adata_cond):
    """Conditions fixture: bit-exact match against R's diffEndTest output."""
    res = tradeseq.diff_end_test(
        r_adata_cond, global_=True, pairwise=True
    )
    ref = pd.read_csv(
        REF_DIR / "r_diffend_conditions_pairwise.csv", index_col=0
    )
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )
    np.testing.assert_allclose(
        res["waldStat_1vs2"].to_numpy(), ref["waldStat_1vs2"].to_numpy(),
        atol=1e-9, rtol=1e-9,
    )
    np.testing.assert_allclose(
        res["logFC1_2"].to_numpy(), ref["logFC1_2"].to_numpy(),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Tier 2 — End-to-end with Python's fit_gam
# ---------------------------------------------------------------------------


def test_diff_end_test_end_to_end_paul15(fitted_paul15_g10):
    """Python-fit end-to-end output correlates with R's reference (Tier-2)."""
    adata, gene_names = fitted_paul15_g10
    py_res = tradeseq.diff_end_test(adata, global_=True)
    r_ref = pd.read_csv(REF_DIR / "r_diffend_g.csv", index_col=0)
    common = [g for g in gene_names if g in r_ref.index]
    py_sub = py_res.loc[common]; r_sub = r_ref.loc[common]
    # logFC1_2: NB-GAM divergence allowed within Tier-2 thresholds.
    mask = np.isfinite(py_sub["logFC1_2"]) & np.isfinite(r_sub["logFC1_2"])
    r_coef, _ = pearsonr(
        py_sub["logFC1_2"].to_numpy()[mask],
        r_sub["logFC1_2"].to_numpy()[mask],
    )
    assert r_coef > 0.90, f"logFC1_2 Pearson r = {r_coef:.3f} < 0.90"
