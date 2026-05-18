"""Shared pytest fixtures for tradeSeq-python."""

from __future__ import annotations

import numpy as np
import pytest

from tradeseq import load_paul15


@pytest.fixture(scope="session")
def paul15_adata():
    """Full bundled Paul-2015 trajectory fixture (2660 cells × 240 genes, 2 lineages)."""
    return load_paul15()


@pytest.fixture(scope="session")
def paul15_small_adata(paul15_adata):
    """A tiny slice of the Paul-2015 fixture suitable for fast unit tests.

    Five high-expression genes used in the R smoke test, all 2660 cells retained
    so that the trajectory geometry (`obsm["pseudotime"]`, `obsm["cell_weights"]`,
    `uns["slingshot"]`) is unchanged.
    """
    genes = ["Acin1", "Actb", "Ank", "Adam9", "Alas1"]
    keep = [g for g in genes if g in paul15_adata.var_names]
    if not keep:
        keep = list(paul15_adata.var_names[:5])
    return paul15_adata[:, keep].copy()


@pytest.fixture(scope="session")
def paul15_one_lineage_adata(paul15_adata):
    """Single-lineage projection of the Paul-2015 fixture.

    Retains cells with `cell_weights[:, 0] > 0.5`; keeps only lineage 0 of the
    `pseudotime` and `cell_weights` matrices. Mirrors the R-side
    `testOneLineage.R` setup.
    """
    cw = paul15_adata.obsm["cell_weights"]
    keep_cells = cw[:, 0] > 0.5
    sub = paul15_adata[keep_cells].copy()
    sub.obsm["pseudotime"] = sub.obsm["pseudotime"][:, :1]
    sub.obsm["cell_weights"] = np.ones_like(sub.obsm["pseudotime"])
    sub.uns["slingshot"] = {
        **sub.uns["slingshot"],
        "pseudotime_columns": ["lineage_1"],
        "cell_weights_columns": ["lineage_1"],
        "lineages": {"1": sub.uns["slingshot"]["lineages"]["1"]},
        "curves": {"1": sub.uns["slingshot"]["curves"]["1"]},
    }
    return sub
