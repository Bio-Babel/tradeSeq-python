"""Tests for tradeseq._slot_io — AnnData slot read/write round-trips.

Each test constructs a small synthetic AnnData, writes a fitGAM-shaped payload,
reads it back, and asserts equality of every field. This is purely a storage
contract: no R reference required.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tradeseq._slot_io import (
    read_beta,
    read_conditions,
    read_converged,
    read_design_matrix,
    read_family,
    read_knots,
    read_lpmatrix,
    read_sigma,
    read_slingshot_coldata,
    write_fit_results,
)


@pytest.fixture()
def small_adata() -> ad.AnnData:
    """Empty AnnData of 20 cells × 5 genes — no payload, ready for write_fit_results."""
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.integers(0, 10, size=(20, 5)).astype(np.float32))
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_id": np.arange(20)}, index=[f"c{i}" for i in range(20)]),
        var=pd.DataFrame({"gene_id": np.arange(5)}, index=[f"g{i}" for i in range(5)]),
    )


@pytest.fixture()
def payload(small_adata: ad.AnnData) -> dict:
    """A complete fitGAM-shaped payload for round-trip testing."""
    n_cells = small_adata.n_obs
    n_genes = small_adata.n_vars
    n_coefs = 12
    n_knots = 6
    rng = np.random.default_rng(1)
    beta = rng.normal(size=(n_genes, n_coefs))
    sigma_list = [rng.normal(size=(n_coefs, n_coefs)) for _ in range(n_genes)]
    converged = np.array([True, True, False, True, True])
    dm_cols = ["U", "offset(offset)", "t1", "l1", "t2", "l2"]
    design_matrix = pd.DataFrame(
        {col: rng.normal(size=n_cells) for col in dm_cols},
        index=small_adata.obs_names.copy(),
    )
    lpmatrix = rng.normal(size=(n_cells, n_coefs))
    knots = np.linspace(0, 1.5, n_knots)
    family = "nb"
    conditions = pd.Categorical(
        np.repeat(["A", "B"], n_cells // 2), categories=["A", "B"]
    )
    slingshot_coldata = pd.DataFrame(
        {"pseudotime_1": rng.normal(size=n_cells)},
        index=small_adata.obs_names.copy(),
    )
    return dict(
        beta=beta,
        sigma_list=sigma_list,
        converged=converged,
        design_matrix=design_matrix,
        lpmatrix=lpmatrix,
        knots=knots,
        family=family,
        conditions=conditions,
        slingshot_coldata=slingshot_coldata,
    )


def test_write_and_read_beta(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_beta(small_adata)
    np.testing.assert_array_equal(out, payload["beta"])
    assert out.dtype == np.float64
    assert out.shape == (small_adata.n_vars, payload["beta"].shape[1])


def test_write_and_read_sigma(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_sigma(small_adata)
    assert out.shape == (small_adata.n_vars,)
    assert out.dtype == object
    for g in range(small_adata.n_vars):
        np.testing.assert_array_equal(out[g], payload["sigma_list"][g])
        assert out[g].dtype == np.float64


def test_write_and_read_converged(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_converged(small_adata)
    np.testing.assert_array_equal(out, payload["converged"])
    assert out.dtype == bool


def test_write_and_read_design_matrix(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_design_matrix(small_adata)
    pd.testing.assert_frame_equal(out, payload["design_matrix"])


def test_write_and_read_lpmatrix(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_lpmatrix(small_adata)
    np.testing.assert_array_equal(out, payload["lpmatrix"])
    assert out.dtype == np.float64


def test_write_and_read_knots(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_knots(small_adata)
    np.testing.assert_array_equal(out, payload["knots"])
    assert out.dtype == np.float64


def test_write_and_read_family(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    assert read_family(small_adata) == "nb"


def test_write_and_read_conditions(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, **payload)
    out = read_conditions(small_adata)
    assert isinstance(out, pd.Categorical)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(payload["conditions"]))
    assert list(out.categories) == ["A", "B"]


def test_write_and_read_conditions_none(small_adata: ad.AnnData, payload: dict) -> None:
    payload["conditions"] = None
    write_fit_results(small_adata, **payload)
    assert read_conditions(small_adata) is None


def test_write_and_read_slingshot_coldata(
    small_adata: ad.AnnData, payload: dict
) -> None:
    write_fit_results(small_adata, **payload)
    out = read_slingshot_coldata(small_adata)
    pd.testing.assert_frame_equal(out, payload["slingshot_coldata"])


def test_write_and_read_slingshot_coldata_none(
    small_adata: ad.AnnData, payload: dict
) -> None:
    payload["slingshot_coldata"] = None
    write_fit_results(small_adata, **payload)
    assert read_slingshot_coldata(small_adata) is None


def test_write_uses_custom_key(small_adata: ad.AnnData, payload: dict) -> None:
    write_fit_results(small_adata, key="myrun", **payload)
    # Default key should NOT be populated.
    assert "tradeseq" not in small_adata.uns
    assert "tradeseq_beta" not in small_adata.varm
    assert "tradeseq_converged" not in small_adata.var
    # Custom key should round-trip.
    np.testing.assert_array_equal(read_beta(small_adata, key="myrun"), payload["beta"])
    np.testing.assert_array_equal(
        read_converged(small_adata, key="myrun"), payload["converged"]
    )
    np.testing.assert_array_equal(read_knots(small_adata, key="myrun"), payload["knots"])
    assert read_family(small_adata, key="myrun") == "nb"
    pd.testing.assert_frame_equal(
        read_design_matrix(small_adata, key="myrun"), payload["design_matrix"]
    )


def test_write_validates_beta_shape(small_adata: ad.AnnData, payload: dict) -> None:
    payload["beta"] = payload["beta"][:-1]  # 4 rows but n_vars=5
    with pytest.raises(ValueError, match="rows but adata has"):
        write_fit_results(small_adata, **payload)


def test_write_validates_sigma_length(small_adata: ad.AnnData, payload: dict) -> None:
    payload["sigma_list"] = payload["sigma_list"][:-1]
    with pytest.raises(ValueError, match="sigma_list has length"):
        write_fit_results(small_adata, **payload)


def test_write_validates_converged_length(
    small_adata: ad.AnnData, payload: dict
) -> None:
    payload["converged"] = payload["converged"][:-1]
    with pytest.raises(ValueError, match="converged has length"):
        write_fit_results(small_adata, **payload)


def test_write_validates_lpmatrix_rows(small_adata: ad.AnnData, payload: dict) -> None:
    payload["lpmatrix"] = payload["lpmatrix"][:-1]
    with pytest.raises(ValueError, match="lpmatrix has"):
        write_fit_results(small_adata, **payload)


def test_write_validates_design_matrix_rows(
    small_adata: ad.AnnData, payload: dict
) -> None:
    payload["design_matrix"] = payload["design_matrix"].iloc[:-1]
    with pytest.raises(ValueError, match="design_matrix has"):
        write_fit_results(small_adata, **payload)


def test_read_raises_when_missing(small_adata: ad.AnnData) -> None:
    with pytest.raises(KeyError):
        read_beta(small_adata)
    with pytest.raises(KeyError):
        read_sigma(small_adata)
    with pytest.raises(KeyError):
        read_converged(small_adata)
    with pytest.raises(KeyError):
        read_design_matrix(small_adata)
    with pytest.raises(KeyError):
        read_lpmatrix(small_adata)
    with pytest.raises(KeyError):
        read_knots(small_adata)
    with pytest.raises(KeyError):
        read_family(small_adata)
    with pytest.raises(KeyError):
        read_conditions(small_adata)
    with pytest.raises(KeyError):
        read_slingshot_coldata(small_adata)


def test_h5ad_round_trip_non_object_slots(
    small_adata: ad.AnnData, payload: dict, tmp_path
) -> None:
    """Every non-object payload must survive an h5ad write/read cycle.

    The Sigma slot is an object array of ``(n_coefs, n_coefs)`` matrices and is not
    natively h5ad-serializable; it is intentionally excluded here so we can verify
    the slots that do serialize (beta, converged, design_matrix, lpmatrix, knots,
    family, conditions, slingshot_coldata).
    """
    write_fit_results(small_adata, **payload)
    # Drop the non-serializable object varm slot before writing h5ad.
    del small_adata.varm["tradeseq_Sigma"]
    path = tmp_path / "round_trip.h5ad"
    small_adata.write_h5ad(path)
    reloaded = ad.read_h5ad(path)

    np.testing.assert_array_equal(read_beta(reloaded), payload["beta"])
    np.testing.assert_array_equal(read_lpmatrix(reloaded), payload["lpmatrix"])
    np.testing.assert_array_equal(read_knots(reloaded), payload["knots"])
    np.testing.assert_array_equal(read_converged(reloaded), payload["converged"])
    assert read_family(reloaded) == "nb"
    pd.testing.assert_frame_equal(
        read_design_matrix(reloaded).reset_index(drop=True),
        payload["design_matrix"].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        read_slingshot_coldata(reloaded).reset_index(drop=True),
        payload["slingshot_coldata"].reset_index(drop=True),
        check_dtype=False,
    )
    conds_back = read_conditions(reloaded)
    np.testing.assert_array_equal(np.asarray(conds_back), np.asarray(payload["conditions"]))
