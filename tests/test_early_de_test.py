"""Tests for early_de_test (R source: tradeSeq/R/earlyDETest.R).

``early_de_test`` and ``pattern_test`` share an implementation; this file
focuses on the *knot-range* behaviour that distinguishes the two — i.e. the
``knots`` argument selecting a sub-range of pseudotime.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from tradeseq.pattern_test import (
    _build_global_L,
    _build_pairwise_L,
    early_de_test,
    pattern_test,
)

REF_DIR = Path("/tmp")


def _require(name: str) -> Path:
    p = REF_DIR / name
    if not p.exists():
        pytest.skip(f"R reference CSV not found: {p}")
    return p


@pytest.fixture(scope="module")
def r_earlyde_inputs():
    _ = _require("r_earlyde.csv")
    _ = _require("r_L_earlyde.csv")
    lpmatrix = pd.read_csv(REF_DIR / "r_lpmatrix_pat.csv")
    dm = pd.read_csv(REF_DIR / "r_dm_pat.csv")
    knots = pd.read_csv(REF_DIR / "r_knots_pat.csv").iloc[:, 0].to_numpy()
    pseudotime = pd.read_csv(REF_DIR / "r_pseudotime_pat.csv").to_numpy()
    beta = pd.read_csv(REF_DIR / "r_beta_pat.csv", index_col=0)
    gene_names = list(beta.index)
    beta_arr = beta.to_numpy()
    n_genes = beta_arr.shape[0]
    sigma_arr = np.empty(n_genes, dtype=object)
    for g in range(n_genes):
        sigma_arr[g] = pd.read_csv(REF_DIR / f"r_sigma_pat_g{g + 1}.csv").to_numpy()
    L_earlyde = pd.read_csv(REF_DIR / "r_L_earlyde.csv", index_col=0).to_numpy()
    L_pattern = pd.read_csv(REF_DIR / "r_L_pattern.csv", index_col=0).to_numpy()
    earlyde_ref = pd.read_csv(REF_DIR / "r_earlyde.csv", index_col=0)
    pattern_ref = pd.read_csv(REF_DIR / "r_pattern.csv", index_col=0)
    return {
        "lpmatrix": lpmatrix,
        "dm": dm,
        "knots": knots,
        "pseudotime": pseudotime,
        "beta": beta_arr,
        "sigma": sigma_arr,
        "gene_names": gene_names,
        "L_earlyde": L_earlyde,
        "L_pattern": L_pattern,
        "earlyde_ref": earlyde_ref,
        "pattern_ref": pattern_ref,
    }


def _make_adata_from_r(inputs: dict, *, key: str = "tradeseq") -> ad.AnnData:
    n_cells = inputs["lpmatrix"].shape[0]
    n_genes = inputs["beta"].shape[0]
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.var_names = inputs["gene_names"]
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.obsm["pseudotime"] = inputs["pseudotime"]
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
        "conditions": None,
        "slingshot_coldata": None,
    }
    return adata


# -----------------------------------------------------------------------------
# Tier-1: knot-range L matrices
# -----------------------------------------------------------------------------


def test_early_de_test_L_matches_R_knots_1_2(r_earlyde_inputs):
    """Building L with knots=(1,2) must match R element-wise."""
    L = _build_global_L(
        dm=r_earlyde_inputs["dm"],
        lpmatrix_df=r_earlyde_inputs["lpmatrix"],
        pseudotime=r_earlyde_inputs["pseudotime"],
        knots=(1, 2),
        knot_points=r_earlyde_inputs["knots"],
        n_points=12,
        n_lineages=2,
        conditions=None,
    )
    np.testing.assert_allclose(L, r_earlyde_inputs["L_earlyde"], atol=1e-10)


def test_early_de_test_knots_endtoend(r_earlyde_inputs):
    """early_de_test(knots=(1,2)) replicates R Wald stat/df/pvalue/fcMedian."""
    adata = _make_adata_from_r(r_earlyde_inputs)
    res = early_de_test(
        adata, knots=(1, 2), global_=True, pairwise=False, n_points=12
    )
    ref = r_earlyde_inputs["earlyde_ref"]
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["df"].to_numpy(), ref["df"].to_numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        res["pvalue"].to_numpy(), ref["pvalue"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["fcMedian"].to_numpy(), ref["fcMedian"].to_numpy(), atol=1e-10
    )


def test_early_de_test_knots_none_equals_pattern(r_earlyde_inputs):
    """early_de_test(knots=None) must produce identical output to pattern_test."""
    adata = _make_adata_from_r(r_earlyde_inputs)
    r_pat = pattern_test(adata, global_=True, pairwise=False, n_points=12)
    r_ede = early_de_test(adata, knots=None, n_points=12)
    pd.testing.assert_frame_equal(r_pat, r_ede)


def test_early_de_test_l2fc_filter(r_earlyde_inputs):
    """Setting a large l2fc shrinks all estimates to 0 and pvalues to 1."""
    adata = _make_adata_from_r(r_earlyde_inputs)
    res = early_de_test(adata, knots=(1, 2), l2fc=20.0, n_points=12)
    assert np.all(res["waldStat"].to_numpy() <= 1e-8)
    assert np.allclose(res["pvalue"].to_numpy(), 1.0, atol=1e-10)


def test_early_de_test_knots_3to5(r_earlyde_inputs):
    """A different knot range yields a different L (smoke test, non-equality)."""
    adata = _make_adata_from_r(r_earlyde_inputs)
    r12 = early_de_test(adata, knots=(1, 2), n_points=12)
    r35 = early_de_test(adata, knots=(3, 5), n_points=12)
    assert not np.allclose(r12["waldStat"], r35["waldStat"])


def test_early_de_test_knots_invalid_length():
    """A non-length-2 knots argument must raise."""
    n_cells = 10
    n_genes = 1
    dm = pd.DataFrame({
        "U": np.ones(n_cells),
        "offset(offset)": np.zeros(n_cells),
        "t1": np.linspace(0, 1, n_cells),
        "l1": np.array([1.0] * 5 + [0.0] * 5),
        "t2": np.linspace(0, 1, n_cells),
        "l2": np.array([0.0] * 5 + [1.0] * 5),
    })
    lp_cols = ["U"] + [f"s(t1):l1.{i}" for i in range(1, 4)] + [f"s(t2):l2.{i}" for i in range(1, 4)]
    lpmat = np.zeros((n_cells, 7))
    lpmat[:5, 1:4] = 0.1
    lpmat[5:, 4:7] = 0.1
    lpmat[:, 0] = 1.0
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.obsm["pseudotime"] = np.stack([np.linspace(0, 1, n_cells)] * 2, axis=1)
    adata.varm["tradeseq_beta"] = np.zeros((n_genes, 7))
    sigma_arr = np.empty(n_genes, dtype=object)
    for g in range(n_genes):
        sigma_arr[g] = np.eye(7)
    adata.varm["tradeseq_Sigma"] = sigma_arr
    adata.var["tradeseq_converged"] = np.ones(n_genes, dtype=bool)
    adata.uns["tradeseq"] = {
        "design_matrix": dm,
        "lpmatrix": lpmat,
        "lpmatrix_columns": lp_cols,
        "knots": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "family": "nb",
        "conditions": None,
        "slingshot_coldata": None,
    }
    with pytest.raises(ValueError, match="length-2"):
        early_de_test(adata, knots=(1, 2, 3))


# -----------------------------------------------------------------------------
# Gap #17: SCE pairwise branch -> _get_eigen_stat_gam ("plain" flavour)
# Degenerate-Σ genes must NOT propagate NaN through the pairwise waldStat.
# -----------------------------------------------------------------------------


def test_early_de_test_pairwise_no_nan_for_degenerate_sigma(r_earlyde_inputs):
    """A gene with all-zero Σ used to leak NaN through the SCE pairwise branch.

    After gap #17, ``_get_eigen_stat_gam`` returns ``(0.0, 0)`` for the
    degenerate Σ instead of ``(NaN, NaN)``. We exercise the pairwise branch
    on a 3-lineage adata (so ``pairwise=True`` is not silently disabled) and
    inject a zero Σ for gene 0; the pairwise waldStat columns must come out
    finite for that gene.
    """
    n_cells = 60
    n_genes = 3
    rng = np.random.default_rng(0)
    n_lin = 3
    n_coefs = 1 + n_lin * 4  # U + n_lin * (4 basis cols)
    # Build a synthetic 3-lineage adata.
    dm = pd.DataFrame({
        "U": np.ones(n_cells),
        "offset(offset)": np.zeros(n_cells),
        "t1": np.linspace(0, 1, n_cells),
        "l1": np.array([1.0] * 20 + [0.0] * 40),
        "t2": np.linspace(0, 1, n_cells),
        "l2": np.array([0.0] * 20 + [1.0] * 20 + [0.0] * 20),
        "t3": np.linspace(0, 1, n_cells),
        "l3": np.array([0.0] * 40 + [1.0] * 20),
    })
    lp_cols = (
        ["U"]
        + [f"s(t1):l1.{i}" for i in range(1, 5)]
        + [f"s(t2):l2.{i}" for i in range(1, 5)]
        + [f"s(t3):l3.{i}" for i in range(1, 5)]
    )
    lpmat = np.zeros((n_cells, n_coefs))
    lpmat[:, 0] = 1.0
    lpmat[:20, 1:5] = 0.1
    lpmat[20:40, 5:9] = 0.1
    lpmat[40:, 9:13] = 0.1
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.obsm["pseudotime"] = np.stack(
        [np.linspace(0, 1, n_cells)] * n_lin, axis=1
    )
    beta = rng.normal(size=(n_genes, n_coefs))
    adata.varm["tradeseq_beta"] = beta
    sigma_arr = np.empty(n_genes, dtype=object)
    # Gene 0: all-zero Σ (degenerate)
    sigma_arr[0] = np.zeros((n_coefs, n_coefs))
    # Genes 1, 2: positive-definite Σ
    sigma_arr[1] = np.eye(n_coefs)
    sigma_arr[2] = np.eye(n_coefs)
    adata.varm["tradeseq_Sigma"] = sigma_arr
    adata.var["tradeseq_converged"] = np.ones(n_genes, dtype=bool)
    adata.uns["tradeseq"] = {
        "design_matrix": dm,
        "lpmatrix": lpmat,
        "lpmatrix_columns": lp_cols,
        "knots": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "family": "nb",
        "conditions": None,
        "slingshot_coldata": None,
    }
    res = early_de_test(adata, knots=None, global_=True, pairwise=True, n_points=6)
    # Three pairwise columns: 1vs2, 1vs3, 2vs3.
    for tag in ("1vs2", "1vs3", "2vs3"):
        col = f"waldStat_{tag}"
        # Gene 0 (degenerate Σ) must be finite (0.0), not NaN.
        assert not np.isnan(res[col].iloc[0]), (
            f"gene 0 with all-zero Σ produced NaN in {col}"
        )
        assert res[col].iloc[0] == 0.0
        # df likewise: gene 0 should have df = 0 in pairwise branch.
        df_col = f"df_{tag}"
        assert res[df_col].iloc[0] == 0.0
