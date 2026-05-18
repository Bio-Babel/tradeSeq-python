"""Bundled-dataset loaders (R source: tradeSeq/R/data.R)."""

from __future__ import annotations

from importlib import resources

import anndata as ad

__all__ = ["load_paul15"]


def load_paul15() -> ad.AnnData:
    """Load the Paul-2015 myeloid trajectory fixture bundled with tradeseq.

    This is the AnnData repacking of the three R datasets `tradeSeq::countMatrix`,
    `tradeSeq::crv`, and `tradeSeq::celltype`. The trajectory was computed once
    on the R side with slingshot 2.8.0 and frozen into the h5ad so the Python
    package depends on no trajectory-inference library at runtime.

    Returns
    -------
    anndata.AnnData
        Shape ``n_obs × n_vars = 2660 × 240`` with:

        - ``X`` and ``layers["counts"]``: CSR int32 raw counts.
        - ``obs["cluster"]``: categorical slingshot cluster id (string).
        - ``obs["celltype"]``: categorical Paul-2015 cell type.
        - ``obsm["X_umap"]``: ``(n_obs, 2)`` UMAP.
        - ``obsm["pseudotime"]``: ``(n_obs, n_lineages)`` ``slingPseudotime(crv, na=FALSE)``.
        - ``obsm["cell_weights"]``: ``(n_obs, n_lineages)`` ``slingCurveWeights(crv)``.
        - ``uns["slingshot"]["lineages"]``: dict cluster sequence per lineage.
        - ``uns["slingshot"]["curves"]``: per-lineage ``{s, ord, lambda}`` arrays.
        - ``uns["provenance"]``: source-package / tool-version metadata.
    """
    resource = resources.files("tradeseq.resources") / "paul15_tradeseq.h5ad"
    with resources.as_file(resource) as path:
        return ad.read_h5ad(path)
