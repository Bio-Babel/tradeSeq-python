"""Tests for pattern_test (R source: tradeSeq/R/patternTest.R + earlyDETest.R).

Two test tiers:

- Tier-1 (L-matrix-only): load R's β / Σ / X / dm / knots / pseudotime,
  build the contrast and run ``pattern_test`` machinery on the R-side fits,
  then assert ``atol=1e-10`` agreement with R's reported waldStat/df/pvalue.
- Tier-2 (end-to-end): fit on the Paul15 fixture in Python and assert the
  output structure (column names, shape, NaN propagation, etc.).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tradeseq
from tradeseq.pattern_test import (
    _build_global_L,
    _build_pairwise_L,
    _fc_median_per_gene,
    _get_eigen_stat_gam,
    _wald_per_gene,
    early_de_test,
    pattern_test,
)

REF_DIR = Path("/tmp")


def _require(name: str) -> Path:
    p = REF_DIR / name
    if not p.exists():
        pytest.skip(f"R reference CSV not found: {p}")
    return p


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_pattern_inputs():
    """Load the R-side fitted artefacts for the 2-lineage patternTest scenario."""
    _ = _require("r_pattern.csv")
    _ = _require("r_earlyde.csv")
    _ = _require("r_L_pattern.csv")
    _ = _require("r_L_earlyde.csv")
    _ = _require("r_dm_pat.csv")
    _ = _require("r_lpmatrix_pat.csv")
    _ = _require("r_beta_pat.csv")
    _ = _require("r_knots_pat.csv")
    _ = _require("r_pseudotime_pat.csv")

    lpmatrix = pd.read_csv(REF_DIR / "r_lpmatrix_pat.csv")
    dm = pd.read_csv(REF_DIR / "r_dm_pat.csv")
    knots = pd.read_csv(REF_DIR / "r_knots_pat.csv").iloc[:, 0].to_numpy()
    pseudotime = pd.read_csv(REF_DIR / "r_pseudotime_pat.csv").to_numpy()
    beta = pd.read_csv(REF_DIR / "r_beta_pat.csv", index_col=0)
    gene_names = list(beta.index)
    beta_arr = beta.to_numpy()
    n_genes = beta_arr.shape[0]
    sigma_list = [
        pd.read_csv(REF_DIR / f"r_sigma_pat_g{g + 1}.csv").to_numpy()
        for g in range(n_genes)
    ]
    sigma_arr = np.empty(n_genes, dtype=object)
    for g, s in enumerate(sigma_list):
        sigma_arr[g] = s
    L_pattern = pd.read_csv(REF_DIR / "r_L_pattern.csv", index_col=0).to_numpy()
    L_earlyde = pd.read_csv(REF_DIR / "r_L_earlyde.csv", index_col=0).to_numpy()
    pattern_ref = pd.read_csv(REF_DIR / "r_pattern.csv", index_col=0)
    earlyde_ref = pd.read_csv(REF_DIR / "r_earlyde.csv", index_col=0)
    return {
        "lpmatrix": lpmatrix,
        "dm": dm,
        "knots": knots,
        "pseudotime": pseudotime,
        "beta": beta_arr,
        "sigma": sigma_arr,
        "gene_names": gene_names,
        "L_pattern": L_pattern,
        "L_earlyde": L_earlyde,
        "pattern_ref": pattern_ref,
        "earlyde_ref": earlyde_ref,
    }


@pytest.fixture(scope="module")
def r_pattern_inputs_3lin():
    """3-lineage scenario for testing pairwise output."""
    _ = _require("r_pattern_3lin.csv")
    _ = _require("r_earlyde_3lin.csv")
    _ = _require("r_L_pattern_3lin.csv")
    _ = _require("r_L_earlyde_3lin.csv")

    lpmatrix = pd.read_csv(REF_DIR / "r_lpmatrix_3lin.csv")
    dm = pd.read_csv(REF_DIR / "r_dm_3lin.csv")
    knots = pd.read_csv(REF_DIR / "r_knots_3lin.csv").iloc[:, 0].to_numpy()
    pseudotime = pd.read_csv(REF_DIR / "r_pseudotime_3lin.csv").to_numpy()
    beta = pd.read_csv(REF_DIR / "r_beta_3lin.csv", index_col=0)
    gene_names = list(beta.index)
    beta_arr = beta.to_numpy()
    n_genes = beta_arr.shape[0]
    sigma_list = [
        pd.read_csv(REF_DIR / f"r_sigma_3lin_g{g + 1}.csv").to_numpy()
        for g in range(n_genes)
    ]
    sigma_arr = np.empty(n_genes, dtype=object)
    for g, s in enumerate(sigma_list):
        sigma_arr[g] = s
    L_global = pd.read_csv(REF_DIR / "r_L_pattern_3lin.csv", index_col=0).to_numpy()
    L_earlyde_global = pd.read_csv(
        REF_DIR / "r_L_earlyde_3lin.csv", index_col=0
    ).to_numpy()
    L_pair_1v2 = pd.read_csv(
        REF_DIR / "r_L_pattern_3lin_pair_1v2.csv", index_col=0
    ).to_numpy()
    L_pair_1v3 = pd.read_csv(
        REF_DIR / "r_L_pattern_3lin_pair_1v3.csv", index_col=0
    ).to_numpy()
    L_pair_2v3 = pd.read_csv(
        REF_DIR / "r_L_pattern_3lin_pair_2v3.csv", index_col=0
    ).to_numpy()
    pattern_ref = pd.read_csv(REF_DIR / "r_pattern_3lin.csv", index_col=0)
    earlyde_ref = pd.read_csv(REF_DIR / "r_earlyde_3lin.csv", index_col=0)
    return {
        "lpmatrix": lpmatrix,
        "dm": dm,
        "knots": knots,
        "pseudotime": pseudotime,
        "beta": beta_arr,
        "sigma": sigma_arr,
        "gene_names": gene_names,
        "L_global": L_global,
        "L_earlyde_global": L_earlyde_global,
        "L_pair_1v2": L_pair_1v2,
        "L_pair_1v3": L_pair_1v3,
        "L_pair_2v3": L_pair_2v3,
        "pattern_ref": pattern_ref,
        "earlyde_ref": earlyde_ref,
    }


def _make_adata_from_r(inputs: dict, *, key: str = "tradeseq") -> ad.AnnData:
    """Build an AnnData mirror of an R-fitted SCE for Tier-1 testing."""
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
# Tier-1: L-matrix-only parity vs R
# -----------------------------------------------------------------------------


def test_pattern_test_L_matches_R_2lineage(r_pattern_inputs):
    """The omnibus L for the 2-lineage pattern test must match R element-wise."""
    L_py = _build_global_L(
        dm=r_pattern_inputs["dm"],
        lpmatrix_df=r_pattern_inputs["lpmatrix"],
        pseudotime=r_pattern_inputs["pseudotime"],
        knots=None,
        knot_points=r_pattern_inputs["knots"],
        n_points=12,
        n_lineages=2,
        conditions=None,
    )
    np.testing.assert_allclose(
        L_py, r_pattern_inputs["L_pattern"], atol=1e-10
    )


def test_pattern_test_endtoend_R_fitted_2lineage(r_pattern_inputs):
    """Run pattern_test on an AnnData backed by R-fitted values; match R Wald output."""
    adata = _make_adata_from_r(r_pattern_inputs)
    res = pattern_test(adata, global_=True, pairwise=False, n_points=12)
    ref = r_pattern_inputs["pattern_ref"]
    # Use the gene names ordering from R reference
    assert list(res.index) == list(ref.index)
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


def test_early_de_test_L_matches_R_2lineage_knots(r_pattern_inputs):
    """earlyDETest with knots=c(1,2) builds an L matching R element-wise."""
    L_py = _build_global_L(
        dm=r_pattern_inputs["dm"],
        lpmatrix_df=r_pattern_inputs["lpmatrix"],
        pseudotime=r_pattern_inputs["pseudotime"],
        knots=(1, 2),
        knot_points=r_pattern_inputs["knots"],
        n_points=12,
        n_lineages=2,
        conditions=None,
    )
    np.testing.assert_allclose(
        L_py, r_pattern_inputs["L_earlyde"], atol=1e-10
    )


def test_early_de_test_endtoend_R_fitted_2lineage(r_pattern_inputs):
    """early_de_test(knots=(1,2)) matches R reference cell-for-cell."""
    adata = _make_adata_from_r(r_pattern_inputs)
    res = early_de_test(adata, knots=(1, 2), global_=True, pairwise=False, n_points=12)
    ref = r_pattern_inputs["earlyde_ref"]
    np.testing.assert_allclose(
        res["waldStat"].to_numpy(), ref["waldStat"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["pvalue"].to_numpy(), ref["pvalue"].to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["fcMedian"].to_numpy(), ref["fcMedian"].to_numpy(), atol=1e-10
    )


def test_early_de_test_with_knots_none_equals_pattern_test(r_pattern_inputs):
    """early_de_test with knots=None must produce identical output to pattern_test."""
    adata = _make_adata_from_r(r_pattern_inputs)
    r1 = pattern_test(adata, global_=True, pairwise=False, n_points=12)
    r2 = early_de_test(
        adata, knots=None, global_=True, pairwise=False, n_points=12
    )
    pd.testing.assert_frame_equal(r1, r2)


# -----------------------------------------------------------------------------
# Tier-1: 3-lineage scenarios (pairwise output exercise)
# -----------------------------------------------------------------------------


def test_pattern_test_L_matches_R_3lineage_global(r_pattern_inputs_3lin):
    """Global L for the 3-lineage scenario must match R element-wise."""
    L_py = _build_global_L(
        dm=r_pattern_inputs_3lin["dm"],
        lpmatrix_df=r_pattern_inputs_3lin["lpmatrix"],
        pseudotime=r_pattern_inputs_3lin["pseudotime"],
        knots=None,
        knot_points=r_pattern_inputs_3lin["knots"],
        n_points=12,
        n_lineages=3,
        conditions=None,
    )
    np.testing.assert_allclose(
        L_py, r_pattern_inputs_3lin["L_global"], atol=1e-10
    )


def test_pattern_test_L_pairwise_matches_R_3lineage(r_pattern_inputs_3lin):
    """Pairwise L for each curve pair must match R element-wise."""
    for (a, b), expected in zip(
        [(1, 2), (1, 3), (2, 3)],
        ["L_pair_1v2", "L_pair_1v3", "L_pair_2v3"],
    ):
        L_py = _build_pairwise_L(
            dm=r_pattern_inputs_3lin["dm"],
            lpmatrix_df=r_pattern_inputs_3lin["lpmatrix"],
            pseudotime=r_pattern_inputs_3lin["pseudotime"],
            curves=(a, b),
            knots=None,
            knot_points=r_pattern_inputs_3lin["knots"],
            n_points=12,
            conditions=None,
        )
        np.testing.assert_allclose(
            L_py, r_pattern_inputs_3lin[expected], atol=1e-10,
            err_msg=f"pair {(a, b)} mismatch",
        )


def test_pattern_test_endtoend_R_fitted_3lineage(r_pattern_inputs_3lin):
    """End-to-end pattern_test(global=True, pairwise=True) matches R for 3 lineages."""
    adata = _make_adata_from_r(r_pattern_inputs_3lin)
    res = pattern_test(adata, global_=True, pairwise=True, n_points=12)
    ref = r_pattern_inputs_3lin["pattern_ref"]
    # Required column structure
    expected_cols = [
        "waldStat", "df", "pvalue",
        "waldStat_1vs2", "df_1vs2", "pvalue_1vs2",
        "waldStat_1vs3", "df_1vs3", "pvalue_1vs3",
        "waldStat_2vs3", "df_2vs3", "pvalue_2vs3",
        "fcMedian",
    ]
    assert list(res.columns) == expected_cols
    for col in expected_cols:
        np.testing.assert_allclose(
            res[col].to_numpy(), ref[col].to_numpy(), atol=1e-10,
            err_msg=f"col {col} mismatch",
        )


def test_early_de_test_endtoend_R_fitted_3lineage(r_pattern_inputs_3lin):
    """End-to-end early_de_test(knots=(1,2), global=True, pairwise=True) for 3 lineages."""
    adata = _make_adata_from_r(r_pattern_inputs_3lin)
    res = early_de_test(
        adata, knots=(1, 2), global_=True, pairwise=True, n_points=12
    )
    ref = r_pattern_inputs_3lin["earlyde_ref"]
    expected_cols = [
        "waldStat", "df", "pvalue",
        "waldStat_1vs2", "df_1vs2", "pvalue_1vs2",
        "waldStat_1vs3", "df_1vs3", "pvalue_1vs3",
        "waldStat_2vs3", "df_2vs3", "pvalue_2vs3",
        "fcMedian",
    ]
    assert list(res.columns) == expected_cols
    for col in expected_cols:
        np.testing.assert_allclose(
            res[col].to_numpy(), ref[col].to_numpy(), atol=1e-10,
            err_msg=f"col {col} mismatch",
        )


# -----------------------------------------------------------------------------
# Branch / API tests
# -----------------------------------------------------------------------------


def test_pattern_test_two_lineage_pairwise_emits_warning(r_pattern_inputs):
    """Two lineages + pairwise=True must emit a warning and disable pairwise."""
    adata = _make_adata_from_r(r_pattern_inputs)
    with pytest.warns(UserWarning, match="Only two lineages"):
        res = pattern_test(adata, global_=True, pairwise=True, n_points=12)
    # Only global columns + fcMedian should be present (no pairwise blocks)
    assert "waldStat_1vs2" not in res.columns
    assert "waldStat" in res.columns


def test_pattern_test_npoints_default_is_2x_nknots(r_pattern_inputs):
    """When n_points=None, default to 2*nknots (12 here)."""
    adata = _make_adata_from_r(r_pattern_inputs)
    res_default = pattern_test(adata, global_=True, pairwise=False)
    res_explicit = pattern_test(adata, global_=True, pairwise=False, n_points=12)
    pd.testing.assert_frame_equal(res_default, res_explicit)


def test_pattern_test_one_lineage_raises():
    """Single-lineage AnnData must raise (R: ``stop()`` at earlyDETest.R:43)."""
    # Build a minimal AnnData with only one lineage in dm
    n_cells = 5
    n_genes = 2
    dm = pd.DataFrame({
        "U": np.ones(n_cells),
        "offset(offset)": np.zeros(n_cells),
        "t1": np.linspace(0, 1, n_cells),
        "l1": np.ones(n_cells),
    })
    lpmat = np.eye(n_cells, 6)
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float64)
    adata = ad.AnnData(X=X)
    adata.obsm["pseudotime"] = np.linspace(0, 1, n_cells).reshape(-1, 1)
    adata.varm["tradeseq_beta"] = np.zeros((n_genes, 6))
    sigma_arr = np.empty(n_genes, dtype=object)
    for g in range(n_genes):
        sigma_arr[g] = np.eye(6)
    adata.varm["tradeseq_Sigma"] = sigma_arr
    adata.var["tradeseq_converged"] = np.ones(n_genes, dtype=bool)
    adata.uns["tradeseq"] = {
        "design_matrix": dm,
        "lpmatrix": lpmat,
        "lpmatrix_columns": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "knots": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "family": "nb",
        "conditions": None,
        "slingshot_coldata": None,
    }
    with pytest.raises(ValueError, match="one lineage"):
        pattern_test(adata)


def test_pattern_test_nan_propagates(r_pattern_inputs):
    """NaN in β or Σ for one gene must yield NaN waldStat/df/pvalue for that gene."""
    adata = _make_adata_from_r(r_pattern_inputs)
    # Inject NaN into gene 0
    adata.varm["tradeseq_beta"] = adata.varm["tradeseq_beta"].copy()
    adata.varm["tradeseq_beta"][0, :] = np.nan
    res = pattern_test(adata, global_=True, pairwise=False, n_points=12)
    assert np.isnan(res.iloc[0]["waldStat"])
    assert np.isnan(res.iloc[0]["df"])
    assert np.isnan(res.iloc[0]["pvalue"])
    # Other genes still finite
    assert not np.isnan(res.iloc[1]["waldStat"])


def test_fc_median_helper_matches_R_fold_changes(r_pattern_inputs):
    """The internal fcMedian helper must reproduce R's matrixStats::rowMedians."""
    fc = _fc_median_per_gene(
        r_pattern_inputs["beta"], r_pattern_inputs["L_pattern"]
    )
    np.testing.assert_allclose(
        fc, r_pattern_inputs["pattern_ref"]["fcMedian"].to_numpy(), atol=1e-10
    )


def test_pattern_test_l2fc_zeros_when_threshold_exceeds_estimate(r_pattern_inputs):
    """A huge l2fc must shrink the estimate; stat == 0 and pvalue == 1."""
    adata = _make_adata_from_r(r_pattern_inputs)
    res = pattern_test(adata, global_=True, l2fc=20.0, n_points=12)
    assert np.all(res["waldStat"].to_numpy() <= 1e-8)
    assert np.allclose(res["pvalue"].to_numpy(), 1.0, atol=1e-10)


def test_wald_per_gene_handles_nan_sigma():
    """``_wald_per_gene`` must return NaN for genes whose Σ contains NaN."""
    n_genes = 3
    n_coefs = 4
    rng = np.random.default_rng(0)
    beta = rng.normal(size=(n_genes, n_coefs))
    sigma_arr = np.empty(n_genes, dtype=object)
    sigma_arr[0] = np.eye(n_coefs)
    sigma_arr[1] = np.full((n_coefs, n_coefs), np.nan)
    sigma_arr[2] = np.eye(n_coefs)
    L = np.array([[1.0, 0], [0, 1], [0, 0], [0, 0]])
    stat, df, pval = _wald_per_gene(beta, sigma_arr, L, l2fc=0, eigen_thresh=1e-2)
    assert not np.isnan(stat[0])
    assert np.isnan(stat[1])
    assert not np.isnan(stat[2])


# -----------------------------------------------------------------------------
# Gap #17: _get_eigen_stat_gam must NOT emit NaN sentinels for r == 0
# (mirrors R utils.R:394-407 which has no early-return NaN; the SCE pairwise
#  branch at earlyDETest.R:173 calls this helper)
# -----------------------------------------------------------------------------


def test_get_eigen_stat_gam_zero_sigma_returns_zero_not_nan():
    """All-zero Σ (degenerate covariance) must return ``(0.0, 0)``, not NaN.

    R's ``getEigenStatGAM`` (utils.R:394-407) has no early-return sentinel —
    when ``eSigma$values[1] == 0`` the function actually errors out at
    ``seq_len(NA)``. The Python port chose the graceful ``(0.0, 0)`` return so
    downstream Wald tables stay finite, matching the spirit of "no NaN".
    """
    beta = np.zeros(2)
    Sigma = np.zeros((2, 2))
    L = np.eye(2)
    stat, r = _get_eigen_stat_gam(beta, Sigma, L)
    assert stat == 0.0
    assert r == 0


def test_get_eigen_stat_gam_low_rank_nonzero_top_eigenvalue():
    """A rank-1 Σ with non-zero top eigenvalue must produce a finite stat.

    R reference (rank-1 Σ, ``getEigenStatGAM``):
        beta = c(1, 2); Sigma = matrix(c(1, 0.5, 0.5, 0.25), 2, 2); L = diag(2)
        getEigenStatGAM(beta, Sigma, L)  -> c(2.56, 1)
    """
    beta = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.5], [0.5, 0.25]])
    L = np.eye(2)
    stat, r = _get_eigen_stat_gam(beta, Sigma, L)
    assert stat == pytest.approx(2.56, abs=1e-10)
    assert r == 1.0


def test_get_eigen_stat_gam_identity_full_rank():
    """Identity Σ, beta=(1,2), L=I: stat=1+4=5, r=2 (exact form is ``β'β``)."""
    beta = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    L = np.eye(2)
    stat, r = _get_eigen_stat_gam(beta, Sigma, L)
    assert stat == pytest.approx(5.0, abs=1e-10)
    assert r == 2.0


def test_get_eigen_stat_gam_pairwise_branch_no_nan_propagation():
    """SCE pairwise branch (earlyDETest.R:173) uses ``getEigenStatGAM`` (no FC).

    A gene with a degenerate Σ in the pairwise branch used to yield NaN in
    the Python port; after gap #17 it must yield ``(0.0, 0)``, propagating to
    a Wald statistic of 0 and a chisq p-value of 1 (1 - pchisq(0, df=0) = 1).
    """
    beta = np.zeros(3)
    Sigma = np.zeros((3, 3))
    L = np.array([[1.0], [-1.0], [0.0]])  # 1-vs-2 contrast, n_coefs=3
    stat, r = _get_eigen_stat_gam(beta, Sigma, L)
    assert not np.isnan(stat)
    assert not np.isnan(r)
    assert stat == 0.0
    assert r == 0


def test_get_eigen_stat_gam_below_threshold_eigenvalues_truncated():
    """Eigen-values < 1e-8 × top must be dropped (r counts only above-threshold).

    With Σ = diag(1, 1e-10), the second eigenvalue ratio 1e-10 is below the
    1e-8 cutoff and r should be 1, not 2.
    """
    beta = np.array([1.0, 1.0])
    Sigma = np.diag([1.0, 1e-10])
    L = np.eye(2)
    stat, r = _get_eigen_stat_gam(beta, Sigma, L)
    assert r == 1.0
    assert stat == pytest.approx(1.0, abs=1e-10)


# -----------------------------------------------------------------------------
# Tier-2: Python-fit end-to-end smoke test
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Some library sizes are zero")
def test_pattern_test_endtoend_python_fit(paul15_small_adata):
    """pattern_test on a Python fit_gam must run and return the expected shape.

    The Python NB-GAM fitter does not reproduce mgcv exactly, so we only check
    the table shape, column names, and basic sanity (waldStat >= 0, pvalue in
    [0,1] for converged genes). The library-size warning emitted by
    ``compute_offset`` (slice 0) is unrelated to this test and ignored.
    """
    a = paul15_small_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    tradeseq.fit_gam(a, n_knots=6, verbose=False, _w_samp=w_samp)
    res = pattern_test(a, global_=True, pairwise=False)
    assert list(res.columns) == ["waldStat", "df", "pvalue", "fcMedian"]
    assert res.shape == (a.n_vars, 4)
    finite = res.dropna()
    if len(finite) > 0:
        assert (finite["waldStat"] >= 0).all()
        assert ((finite["pvalue"] >= 0) & (finite["pvalue"] <= 1)).all()
