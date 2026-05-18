"""Tests for the bundled-dataset loader (R source: tradeSeq/R/data.R)."""

from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np
import pytest
from scipy.sparse import issparse

import tradeseq


EXPECTED_SHA256 = "bf11293b3c0a6f5e4100d08aa7ffb07cf48e76541fca03c45b8f8b756967a5fd"


def test_resource_sha256():
    path = resources.files("tradeseq.resources") / "paul15_tradeseq.h5ad"
    with resources.as_file(path) as p:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA256


def test_load_paul15_shape(paul15_adata):
    assert paul15_adata.n_obs == 2660
    assert paul15_adata.n_vars == 240


def test_load_paul15_layers(paul15_adata):
    assert "counts" in paul15_adata.layers
    counts = paul15_adata.layers["counts"]
    assert issparse(counts)
    assert counts.dtype == np.int32
    assert counts.shape == (2660, 240)


def test_load_paul15_obs(paul15_adata):
    assert "cluster" in paul15_adata.obs.columns
    assert "celltype" in paul15_adata.obs.columns
    assert str(paul15_adata.obs["cluster"].dtype) == "category"
    assert str(paul15_adata.obs["celltype"].dtype) == "category"


def test_load_paul15_obsm(paul15_adata):
    for key in ("X_umap", "pseudotime", "cell_weights"):
        assert key in paul15_adata.obsm
        assert paul15_adata.obsm[key].shape == (2660, 2)


def test_load_paul15_no_nan_in_pseudotime(paul15_adata):
    assert not np.isnan(paul15_adata.obsm["pseudotime"]).any(), (
        "load_paul15() must use na=FALSE — no NaN in pseudotime"
    )


def test_load_paul15_uns_slingshot(paul15_adata):
    s = paul15_adata.uns["slingshot"]
    assert set(s.keys()) >= {"lineages", "curves",
                              "pseudotime_columns", "cell_weights_columns"}
    assert set(s["curves"].keys()) == {"1", "2"}
    for lin_id, curve in s["curves"].items():
        assert set(curve.keys()) >= {"s", "ord", "lambda"}
        assert curve["s"].shape == (2660, 2)
        assert curve["ord"].shape == (2660,)
        assert curve["lambda"].shape == (2660,)


def test_load_paul15_provenance(paul15_adata):
    prov = paul15_adata.uns["provenance"]
    assert prov["source_package"] == "tradeSeq"
    assert "slingshot_version" in prov


def test_load_paul15_idempotent():
    """Calling twice must produce equivalent objects (read-only resource)."""
    a = tradeseq.load_paul15()
    b = tradeseq.load_paul15()
    assert a.shape == b.shape
    assert (a.layers["counts"] != b.layers["counts"]).nnz == 0
    np.testing.assert_array_equal(a.obsm["pseudotime"], b.obsm["pseudotime"])
