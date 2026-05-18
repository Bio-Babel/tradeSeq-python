"""Tests for condition_test (R source: tradeSeq/R/conditionTest.R).

Two test tiers, matching test_pattern_test.py:

- Tier-1: load R β/Σ/X/dm/knots/conditions from a 3-condition fitted SCE and
  assert ``atol=1e-10`` agreement with R's reported Wald output.
- Tier-2: API-level guards (raises, warnings, NaN propagation).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from tradeseq.condition_test import (
    _construct_contrast_matrix_conditions,
    _count_curves_in_dm,
    _filter_L_for_condition_pair,
    _find_conditions,
    _wald_per_gene,
    condition_test,
)

REF_DIR = Path("/tmp")


def _require(name: str) -> Path:
    p = REF_DIR / name
    if not p.exists():
        pytest.skip(f"R reference CSV not found: {p}")
    return p


@pytest.fixture(scope="module")
def r_cond_inputs():
    """Load R-side fitted artefacts for the 2-lineage, 3-condition scenario."""
    _ = _require("r_condition.csv")
    _ = _require("r_L_cond.csv")
    _ = _require("r_dm_cond_full.csv")
    _ = _require("r_lpmatrix_cond_full.csv")
    _ = _require("r_beta_cond.csv")
    _ = _require("r_knots_cond.csv")
    _ = _require("r_conditions_cond.csv")

    lpmatrix = pd.read_csv(REF_DIR / "r_lpmatrix_cond_full.csv")
    dm = pd.read_csv(REF_DIR / "r_dm_cond_full.csv")
    knots = pd.read_csv(REF_DIR / "r_knots_cond.csv").iloc[:, 0].to_numpy()
    conditions = pd.read_csv(REF_DIR / "r_conditions_cond.csv").iloc[:, 0].astype(str).tolist()
    conditions_cat = pd.Categorical(conditions)
    beta = pd.read_csv(REF_DIR / "r_beta_cond.csv", index_col=0)
    gene_names = list(beta.index)
    beta_arr = beta.to_numpy()
    n_genes = beta_arr.shape[0]
    sigma_arr = np.empty(n_genes, dtype=object)
    for g in range(n_genes):
        sigma_arr[g] = pd.read_csv(REF_DIR / f"r_sigma_cond_g{g + 1}.csv").to_numpy()
    L_cond = pd.read_csv(REF_DIR / "r_L_cond.csv", index_col=0).to_numpy()
    cond_ref = pd.read_csv(REF_DIR / "r_condition.csv", index_col=0)
    cond_pw_ref = pd.read_csv(REF_DIR / "r_condition_pw.csv", index_col=0)
    cond_ln_ref = pd.read_csv(REF_DIR / "r_condition_ln.csv", index_col=0)
    cond_pwln_ref = pd.read_csv(REF_DIR / "r_condition_pwln.csv", index_col=0)
    cond_knot_ref = pd.read_csv(REF_DIR / "r_condition_knot.csv", index_col=0)
    return {
        "lpmatrix": lpmatrix,
        "dm": dm,
        "knots": knots,
        "conditions": conditions_cat,
        "beta": beta_arr,
        "sigma": sigma_arr,
        "gene_names": gene_names,
        "L_cond": L_cond,
        "cond_ref": cond_ref,
        "cond_pw_ref": cond_pw_ref,
        "cond_ln_ref": cond_ln_ref,
        "cond_pwln_ref": cond_pwln_ref,
        "cond_knot_ref": cond_knot_ref,
    }


def _make_adata_from_r(inputs: dict, *, key: str = "tradeseq") -> ad.AnnData:
    n_cells = inputs["lpmatrix"].shape[0]
    n_genes = inputs["beta"].shape[0]
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.var_names = inputs["gene_names"]
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.varm[f"{key}_beta"] = inputs["beta"]
    sigma_arr = np.empty(n_genes, dtype=object)
    for g, s in enumerate(inputs["sigma"]):
        sigma_arr[g] = s
    adata.varm[f"{key}_Sigma"] = sigma_arr
    adata.var[f"{key}_converged"] = np.ones(n_genes, dtype=bool)
    adata.uns[key] = {
        "design_matrix": inputs["dm"],
        "lpmatrix": inputs["lpmatrix"].to_numpy(),
        "lpmatrix_columns": list(inputs["lpmatrix"].columns),
        "knots": inputs["knots"],
        "family": "nb",
        "conditions": inputs["conditions"],
        "slingshot_coldata": None,
    }
    return adata


# -----------------------------------------------------------------------------
# Tier-1: L matrix construction
# -----------------------------------------------------------------------------


def test_contrast_matrix_matches_R(r_cond_inputs):
    """The condition contrast matrix must match R's `.construct_contrast_matrix_conditions`."""
    L_py, colnames = _construct_contrast_matrix_conditions(
        lpmatrix_columns=list(r_cond_inputs["lpmatrix"].columns),
        n_conditions=3,
        n_lineages=2,
        n_knots_total=6,
        n_knots_block=6,
        knots=(1, 6),
    )
    np.testing.assert_allclose(L_py, r_cond_inputs["L_cond"], atol=1e-10)
    # column labels are 18 lineage1 then 18 lineage2 (6 knots * 3 comparisons)
    assert colnames[:18] == ["lineage1"] * 18
    assert colnames[18:] == ["lineage2"] * 18


def test_contrast_matrix_knot_subrange(r_cond_inputs):
    """A sub-range of knots produces a narrower L matrix."""
    L_py, _ = _construct_contrast_matrix_conditions(
        lpmatrix_columns=list(r_cond_inputs["lpmatrix"].columns),
        n_conditions=3,
        n_lineages=2,
        n_knots_total=6,
        n_knots_block=2,
        knots=(1, 2),
    )
    # 3 condition pairs × 2 knots × 2 lineages = 12 columns
    assert L_py.shape == (37, 12)


# -----------------------------------------------------------------------------
# Tier-1: end-to-end Wald output parity with R
# -----------------------------------------------------------------------------


def test_condition_test_global_matches_R(r_cond_inputs):
    """conditionTest(global=TRUE) matches R Wald output cell-for-cell."""
    adata = _make_adata_from_r(r_cond_inputs)
    res = condition_test(adata, global_=True)
    ref = r_cond_inputs["cond_ref"]
    assert list(res.index) == list(ref.index)
    assert list(res.columns) == ["waldStat", "df", "pvalue"]
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["df"].to_numpy(), ref["df"].to_numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        res["pvalue"].to_numpy(), ref["pvalue"].to_numpy(), atol=1e-10
    )


def test_condition_test_pairwise_matches_R(r_cond_inputs):
    """conditionTest(global=TRUE, pairwise=TRUE) matches R for all condition pairs."""
    adata = _make_adata_from_r(r_cond_inputs)
    res = condition_test(adata, global_=True, pairwise=True)
    ref = r_cond_inputs["cond_pw_ref"]
    expected_cols = [
        "waldStat", "df", "pvalue",
        "waldStat_conds1vs2", "df_conds1vs2", "pvalue_conds1vs2",
        "waldStat_conds1vs3", "df_conds1vs3", "pvalue_conds1vs3",
        "waldStat_conds2vs3", "df_conds2vs3", "pvalue_conds2vs3",
    ]
    assert list(res.columns) == expected_cols
    for col in expected_cols:
        np.testing.assert_allclose(
            res[col].to_numpy(), ref[col].to_numpy(), atol=1e-10,
            err_msg=f"col {col} mismatch",
        )


def test_condition_test_lineages_matches_R(r_cond_inputs):
    """conditionTest(global=TRUE, lineages=TRUE) matches R per-lineage output."""
    adata = _make_adata_from_r(r_cond_inputs)
    res = condition_test(adata, global_=True, lineages=True)
    ref = r_cond_inputs["cond_ln_ref"]
    expected_cols = [
        "waldStat", "df", "pvalue",
        "waldStat_lineage1", "df_lineage1", "pvalue_lineage1",
        "waldStat_lineage2", "df_lineage2", "pvalue_lineage2",
    ]
    assert list(res.columns) == expected_cols
    for col in expected_cols:
        np.testing.assert_allclose(
            res[col].to_numpy(), ref[col].to_numpy(), atol=1e-10,
            err_msg=f"col {col} mismatch",
        )


def test_condition_test_pairwise_lineages_matches_R(r_cond_inputs):
    """conditionTest(pairwise=TRUE, lineages=TRUE) matches R for combined output."""
    adata = _make_adata_from_r(r_cond_inputs)
    res = condition_test(adata, global_=True, pairwise=True, lineages=True)
    ref = r_cond_inputs["cond_pwln_ref"]
    # Expected columns: omnibus + lineage × condition pair combinations
    expected_cols = ["waldStat", "df", "pvalue"]
    for ll in (1, 2):
        for (a, b) in [(1, 2), (1, 3), (2, 3)]:
            tag = f"lineage{ll}_conds{a}vs{b}"
            expected_cols.extend(
                [f"waldStat_{tag}", f"df_{tag}", f"pvalue_{tag}"]
            )
    assert list(res.columns) == expected_cols
    for col in expected_cols:
        np.testing.assert_allclose(
            res[col].to_numpy(), ref[col].to_numpy(), atol=1e-10,
            err_msg=f"col {col} mismatch",
        )


def test_condition_test_with_knots_matches_R(r_cond_inputs):
    """conditionTest(knots=(1,2)) restricts the knot range and matches R."""
    adata = _make_adata_from_r(r_cond_inputs)
    res = condition_test(adata, global_=True, knots=(1, 2))
    ref = r_cond_inputs["cond_knot_ref"]
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["df"].to_numpy(), ref["df"].to_numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        res["pvalue"].to_numpy(), ref["pvalue"].to_numpy(), atol=1e-10
    )


# -----------------------------------------------------------------------------
# Tier-2: API guards
# -----------------------------------------------------------------------------


def test_condition_test_raises_when_no_conditions(r_cond_inputs):
    """conditionTest must fail if the AnnData was fit without conditions."""
    adata = _make_adata_from_r(r_cond_inputs)
    adata.uns["tradeseq"]["conditions"] = None
    with pytest.raises(ValueError, match="multiple conditions"):
        condition_test(adata)


def test_condition_test_raises_single_condition(r_cond_inputs):
    """conditionTest must fail if conditions has only one level."""
    adata = _make_adata_from_r(r_cond_inputs)
    single = pd.Categorical(["a"] * adata.n_obs, categories=["a"])
    adata.uns["tradeseq"]["conditions"] = single
    with pytest.raises(ValueError, match="multiple conditions"):
        condition_test(adata)


def test_condition_test_requires_one_of_global_pairwise_lineages(r_cond_inputs):
    """At least one of global, pairwise, lineages must be True."""
    adata = _make_adata_from_r(r_cond_inputs)
    with pytest.raises(ValueError, match="One of global"):
        condition_test(adata, global_=False, pairwise=False, lineages=False)


def test_condition_test_two_conditions_pairwise_warns():
    """With only two conditions, ``pairwise=True`` emits a warning and is disabled."""
    # Build a tiny adata with 2 conditions
    n_cells = 6
    n_genes = 2
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.var_names = ["g1", "g2"]
    cond = pd.Categorical([1, 1, 1, 2, 2, 2])
    dm = pd.DataFrame({
        "U": np.ones(n_cells),
        "offset(offset)": np.zeros(n_cells),
        "t1": np.linspace(0, 1, n_cells),
        "l1_1": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "l1_2": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        "t2": np.linspace(0, 1, n_cells),
        "l2_1": np.zeros(n_cells),
        "l2_2": np.zeros(n_cells),
    })
    n_knots = 3
    n_coefs = 1 + 2 * 2 * n_knots  # U + 2 lineages * 2 conditions * 3 knots
    lp_cols = ["U"]
    for ii in (1, 2):
        for kk in (1, 2):
            for jj in range(1, n_knots + 1):
                lp_cols.append(f"s(t{ii}):l{ii}_{kk}.{jj}")
    lpmat = np.zeros((n_cells, n_coefs))
    rng2 = np.random.default_rng(1)
    beta = rng2.normal(size=(n_genes, n_coefs))
    sigma_arr = np.empty(n_genes, dtype=object)
    for g in range(n_genes):
        sigma_arr[g] = np.eye(n_coefs) * 0.1
    adata.varm["tradeseq_beta"] = beta
    adata.varm["tradeseq_Sigma"] = sigma_arr
    adata.var["tradeseq_converged"] = np.ones(n_genes, dtype=bool)
    adata.uns["tradeseq"] = {
        "design_matrix": dm,
        "lpmatrix": lpmat,
        "lpmatrix_columns": lp_cols,
        "knots": np.array([0.0, 0.5, 1.0]),
        "family": "nb",
        "conditions": cond,
        "slingshot_coldata": None,
    }
    with pytest.warns(UserWarning, match="two conditions"):
        res = condition_test(adata, global_=True, pairwise=True)
    # pairwise disabled, only omnibus columns
    assert list(res.columns) == ["waldStat", "df", "pvalue"]


def test_condition_test_one_lineage_lineages_warns(r_cond_inputs):
    """With only one lineage, ``lineages=True`` must warn and disable lineages."""
    # Fake a single-lineage scenario by stripping the second lineage's columns from dm/lpmatrix.
    adata = _make_adata_from_r(r_cond_inputs)
    dm = adata.uns["tradeseq"]["design_matrix"].drop(columns=["t2", "l2_1", "l2_2", "l2_3"])
    adata.uns["tradeseq"]["design_matrix"] = dm
    # And shrink the lpmatrix to drop t2-related smooth coefs
    lp_cols = adata.uns["tradeseq"]["lpmatrix_columns"]
    keep = [i for i, c in enumerate(lp_cols) if not c.startswith("s(t2)")]
    adata.uns["tradeseq"]["lpmatrix_columns"] = [lp_cols[i] for i in keep]
    adata.uns["tradeseq"]["lpmatrix"] = adata.uns["tradeseq"]["lpmatrix"][:, keep]
    # Shrink beta / sigma
    adata.varm["tradeseq_beta"] = adata.varm["tradeseq_beta"][:, keep]
    old_sigma = adata.varm["tradeseq_Sigma"]
    # AnnData may have squeezed object-array varm to shape (n_vars, 1); flatten
    if old_sigma.ndim == 2 and old_sigma.shape[1] == 1:
        old_sigma = old_sigma[:, 0]
    new_sigma = np.empty(adata.n_vars, dtype=object)
    for g in range(adata.n_vars):
        new_sigma[g] = np.asarray(old_sigma[g])[np.ix_(keep, keep)]
    adata.varm["tradeseq_Sigma"] = new_sigma
    with pytest.warns(UserWarning, match="one lineage"):
        res = condition_test(adata, global_=True, lineages=True)
    assert list(res.columns) == ["waldStat", "df", "pvalue"]


def test_condition_test_invalid_knots_raises(r_cond_inputs):
    """Knots argument outside the allowed range must raise."""
    adata = _make_adata_from_r(r_cond_inputs)
    with pytest.raises(ValueError, match="below the number of knots"):
        condition_test(adata, knots=(1, 99))
    with pytest.raises(ValueError, match="below the number of knots"):
        condition_test(adata, knots=(1, 2, 3))  # wrong length


def test_condition_test_nan_propagates(r_cond_inputs):
    """NaN β for one gene must yield NaN waldStat/df/pvalue for that gene."""
    adata = _make_adata_from_r(r_cond_inputs)
    beta = adata.varm["tradeseq_beta"].copy()
    beta[0, :] = np.nan
    adata.varm["tradeseq_beta"] = beta
    res = condition_test(adata, global_=True)
    assert np.isnan(res.iloc[0]["waldStat"])
    assert np.isnan(res.iloc[0]["df"])
    assert np.isnan(res.iloc[0]["pvalue"])
    assert not np.isnan(res.iloc[1]["waldStat"])


def test_find_conditions_returns_n_conditions(r_cond_inputs):
    """`_find_conditions` returns the conditions categorical and its level count."""
    adata = _make_adata_from_r(r_cond_inputs)
    cond, n_cond = _find_conditions(adata, "tradeseq")
    assert n_cond == 3
    assert len(cond) == adata.n_obs


def test_count_curves_in_dm(r_cond_inputs):
    """`_count_curves_in_dm` mirrors R's `grep('l[(1-9)+]', colnames(dm))`."""
    dm_cols = list(r_cond_inputs["dm"].columns)
    assert _count_curves_in_dm(dm_cols) == 6  # 2 lineages × 3 conditions


def test_filter_L_for_condition_pair_keeps_balanced_columns(r_cond_inputs):
    """`_filter_L_for_condition_pair` retains exactly the +1/-1 balanced columns."""
    L = r_cond_inputs["L_cond"]
    lp_cols = list(r_cond_inputs["lpmatrix"].columns)
    Lpair = _filter_L_for_condition_pair(L, lp_cols, (1, 2))
    # Each retained column must have signed sum == 0 and abs-sum != 0
    col_sum = Lpair.sum(axis=0)
    col_abs = np.abs(Lpair).sum(axis=0)
    assert np.allclose(col_sum, 0.0)
    assert np.all(col_abs > 0)
