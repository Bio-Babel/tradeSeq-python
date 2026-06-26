"""Recipe: canonical tradeSeq trajectory differential expression on an AnnData.

Mirrors workflows/trajectory_de.yaml step-by-step. Requires an AnnData that
already carries a trajectory (``obsm['pseudotime']`` + ``obsm['cell_weights']``);
tradeSeq consumes a trajectory, it does not infer one. ``ts.load_paul15()``
returns a ready-made two-lineage example.

Knot selection (``evaluate_k`` / ``plot_evaluatek_results``) is intentionally
left out of ``main`` because it refits many throwaway models; pick ``n_knots``
with it once, then pass the chosen value here (the R vignette uses 6).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

import tradeseq as ts


def main(
    adata: ad.AnnData,
    *,
    n_knots: int = 6,
    layer: str = "counts",
    out_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit the NB-GAM and run the Wald battery; return the result tables.

    Parameters
    ----------
    adata : AnnData
        Cells × genes with raw counts in ``adata.layers[layer]`` and a trajectory
        in ``adata.obsm['pseudotime']`` / ``adata.obsm['cell_weights']``.
    n_knots : int, default 6
        Knot count for the cr-spline basis (choose with ``ts.evaluate_k`` first).
    layer : str, default "counts"
        Raw-count layer name.
    out_path : Path, optional
        If given, write the fitted AnnData to this path as h5ad.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``association`` / ``start_vs_end`` (within-lineage) and, when the
        trajectory has ≥ 2 lineages, ``diff_end`` / ``pattern`` (between-lineage)
        result tables, each indexed by gene.
    """
    # 1. Fit the NB-GAM in place: writes var.tradeseq_converged,
    #    varm.tradeseq_beta, varm.tradeseq_Sigma, uns.tradeseq.
    ts.fit_gam(adata, layer=layer, n_knots=n_knots)

    results: dict[str, pd.DataFrame] = {}

    # 2. Within-lineage differential expression.
    results["association"] = ts.association_test(adata)
    results["start_vs_end"] = ts.start_vs_end_test(adata)

    # 3. Between-lineage differential expression (needs a branching trajectory).
    pt = adata.obsm["pseudotime"]
    n_lineages = pt.shape[1] if getattr(pt, "ndim", 1) == 2 else 1
    if n_lineages >= 2:
        results["diff_end"] = ts.diff_end_test(adata)
        results["pattern"] = ts.pattern_test(adata)

    if out_path is not None:
        adata.write_h5ad(out_path)
    return results


if __name__ == "__main__":
    # Smoke entry: the bundled Paul15 fixture ships counts + a 2-lineage trajectory.
    a = ts.load_paul15()
    tables = main(a, n_knots=6)
    for name, df in tables.items():
        top_gene = df["waldStat"].sort_values(ascending=False).index[0]
        print(f"{name}: top gene = {top_gene}")
