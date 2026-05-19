"""Tests for tradeseq.cascade (R source: tradeSeq/R/cascade.R).

The R implementation contains upstream bugs (notably an unbound ``sce`` symbol
inside the S4 method body), so the R reference fixture is produced by applying
the minimal source-faithful fixes described in
``validation/_dump_cascade_reference.R``. The algorithm itself remains R-gold:
finite-difference first derivatives, pointwise positive-derivative testing,
peak-time ordering, and pheatmap rendering.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pheatmap import PHeatmap
from scipy import sparse

import tradeseq as ts
from tradeseq._slot_io import write_fit_results
from tradeseq.cascade import CascadeResult, cascade, plot_cascade

REF_DIR = Path(__file__).resolve().parents[1] / "validation" / "r_reference" / "cascade"


def _read_r_cascade_adata() -> ad.AnnData:
    dm = pd.read_csv(REF_DIR / "dm.csv")
    x_mat = pd.read_csv(REF_DIR / "X.csv")
    beta = pd.read_csv(REF_DIR / "beta.csv", index_col=0)
    pseudotime = pd.read_csv(REF_DIR / "pseudotime.csv").to_numpy()
    knots = pd.read_csv(REF_DIR / "knots.csv").to_numpy().reshape(-1)
    sigma = [
        pd.read_csv(REF_DIR / f"sigma_g{i + 1}.csv").to_numpy()
        for i in range(beta.shape[0])
    ]
    adata = ad.AnnData(
        X=sparse.csr_matrix((dm.shape[0], beta.shape[0]), dtype=float),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(dm.shape[0])]),
        var=pd.DataFrame(index=beta.index.astype(str)),
    )
    adata.obsm["pseudotime"] = pseudotime
    adata.obsm["cell_weights"] = np.ones_like(pseudotime)
    write_fit_results(
        adata,
        beta=beta.to_numpy(dtype=float),
        sigma_list=sigma,
        converged=np.ones(beta.shape[0], dtype=bool),
        design_matrix=dm,
        lpmatrix=x_mat.to_numpy(dtype=float),
        knots=knots,
        family="nb",
        conditions=None,
    )
    adata.uns["tradeseq"]["lpmatrix_columns"] = list(x_mat.columns)
    return adata


@pytest.fixture(scope="module")
def r_cascade_adata() -> ad.AnnData:
    return _read_r_cascade_adata()


@pytest.fixture(scope="module")
def r_cascade_params() -> dict[str, float | int]:
    row = pd.read_csv(REF_DIR / "params.csv").iloc[0]
    return {
        "n_points": int(row["nPoints"]),
        "derivative_threshold": float(row["derivativeThreshold"]),
        "der_pval_threshold": float(row["derPvalThreshold"]),
    }


@pytest.fixture(scope="module")
def r_cascade_result(r_cascade_adata, r_cascade_params) -> CascadeResult:
    return cascade(
        r_cascade_adata,
        lineage=1,
        genes=list(r_cascade_adata.var_names),
        **r_cascade_params,
    )


def test_cascade_matches_patched_r_reference(r_cascade_result):
    """Python cascade matches the source-faithful patched R reference."""
    peak_ref = pd.read_csv(REF_DIR / "peak_time.csv").set_index("gene")["peakTime"]
    pd.testing.assert_index_equal(
        r_cascade_result.peak_time.index, peak_ref.index, check_names=False
    )
    np.testing.assert_allclose(
        r_cascade_result.peak_time.to_numpy(), peak_ref.to_numpy(), atol=1e-12
    )

    comparisons = [
        ("derivatives", r_cascade_result.derivatives, "derivatives.csv", 1e-8),
        ("sd_derivatives", r_cascade_result.sd_derivatives, "sd_derivatives.csv", 1e-8),
        ("test_statistics", r_cascade_result.test_statistics, "test_statistics.csv", 1e-8),
        ("p_values", r_cascade_result.p_values, "p_values.csv", 1e-8),
        ("yhat", r_cascade_result.yhat, "yhat.csv", 1e-10),
    ]
    for _name, observed, filename, atol in comparisons:
        expected = pd.read_csv(REF_DIR / filename, index_col=0)
        assert observed.shape == expected.shape
        np.testing.assert_allclose(
            observed.to_numpy(dtype=float), expected.to_numpy(dtype=float), atol=atol
        )


def test_cascade_return_structure(r_cascade_result):
    """The compute export returns the split calculation object, not a plot."""
    assert isinstance(r_cascade_result, CascadeResult)
    assert r_cascade_result.lineage == 1
    assert r_cascade_result.grid.shape[0] == 20
    assert list(r_cascade_result.yhat.index) == list(r_cascade_result.peak_time.index)
    assert all(c.startswith("lineage1_") for c in r_cascade_result.yhat.columns)


def test_cascade_integer_gene_indices(r_cascade_adata, r_cascade_params, r_cascade_result):
    """Integer gene indices select the same genes as R's row-index path."""
    out = cascade(r_cascade_adata, lineage=1, genes=range(10), **r_cascade_params)
    pd.testing.assert_series_equal(out.peak_time, r_cascade_result.peak_time)
    np.testing.assert_allclose(out.yhat.to_numpy(), r_cascade_result.yhat.to_numpy())


def test_cascade_empty_result_is_valid_compute_output(r_cascade_adata):
    """No significant genes returns empty peak/yhat objects instead of failing."""
    out = cascade(
        r_cascade_adata,
        lineage=1,
        genes=list(r_cascade_adata.var_names),
        n_points=10,
        derivative_threshold=1e9,
        der_pval_threshold=1e-12,
    )
    assert out.peak_time.empty
    assert out.yhat.empty
    assert out.derivatives.shape == (r_cascade_adata.n_vars, 10)


def test_plot_cascade_returns_pheatmap(r_cascade_result):
    """plot_cascade renders with pheatmap-python and leaves columns unclustered."""
    ph = plot_cascade(r_cascade_result)
    assert isinstance(ph, PHeatmap)
    assert ph.gtable is not None
    assert ph.tree_col is None
    assert ph.tree_row is None


def test_plot_cascade_cluster_heatmap(r_cascade_result):
    """cluster_heatmap=True maps to pheatmap(cluster_rows=True)."""
    ph = plot_cascade(r_cascade_result, cluster_heatmap=True)
    assert isinstance(ph, PHeatmap)
    assert ph.tree_row is not None
    assert ph.tree_col is None


def test_plot_cascade_empty_raises(r_cascade_adata):
    """Plotting is intentionally separate and rejects an empty compute result."""
    out = cascade(
        r_cascade_adata,
        lineage=1,
        n_points=10,
        derivative_threshold=1e9,
        der_pval_threshold=1e-12,
    )
    with pytest.raises(ValueError, match="empty cascade result"):
        plot_cascade(out)


def test_plot_cascade_can_save_nonblank_png(r_cascade_result, tmp_path):
    """pheatmap output should render to a nonblank PNG."""
    path = tmp_path / "cascade.png"
    plot_cascade(r_cascade_result, filename=str(path), width=5, height=3)
    assert path.exists()
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    assert arr.std() > 0


def test_cascade_invalid_inputs(r_cascade_adata):
    """Input validation fails loudly."""
    with pytest.raises(ValueError, match="positive 1-based"):
        cascade(r_cascade_adata, lineage=0)
    with pytest.raises(ValueError, match="at least 2"):
        cascade(r_cascade_adata, n_points=1)
    with pytest.raises(ValueError, match="epsilon"):
        cascade(r_cascade_adata, epsilon=0)
    with pytest.raises(ValueError, match="Not all gene IDs"):
        cascade(r_cascade_adata, genes=["not_a_gene"])
    with pytest.raises(TypeError, match="AnnData"):
        cascade("not_anndata")  # type: ignore[arg-type]


def test_top_level_exports():
    """The split compute/plot surface is exported at package top level."""
    assert ts.cascade is cascade
    assert ts.plot_cascade is plot_cascade
    assert ts.CascadeResult is CascadeResult
