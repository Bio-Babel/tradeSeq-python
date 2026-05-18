"""Start-vs-end test (R source: tradeSeq/R/startVsEndTest.R:1-269).

Port of ``startVsEndTest`` for AnnData populated by :func:`fit_gam`
(mirroring startVsEndTest.R:55-103). Builds a one-column-per-lineage contrast comparing
the lpmatrix row at the lineage end versus the start (or two user-specified
pseudotime points) and runs a multivariate Wald test per gene.
"""

from __future__ import annotations

from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._test_common import (
    build_sve_contrast,
    fetch_test_arrays,
    get_fold_changes,
    n_curves_from_dm,
    run_wald_table,
)

__all__ = ["start_vs_end_test"]


def start_vs_end_test(
    adata: ad.AnnData,
    *,
    global_: bool = True,
    lineages: bool = False,
    pseudotime_values: Optional[Sequence[float]] = None,
    l2fc: float = 0.0,
    key: str = "tradeseq",
) -> pd.DataFrame:
    """Test differential average expression between start and end of every lineage.

    Python port of ``tradeSeq::startVsEndTest`` for AnnData inputs
    (mirroring startVsEndTest.R:233-249). Builds a ``(p, n_lineages)`` contrast
    matrix where each column compares ``X_end - X_start`` for one lineage.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`tradeseq.fit_gam`.
    global_ : bool, default True
        If True, return the omnibus statistic across all lineages.
    lineages : bool, default False
        If True, return per-lineage statistics.
    pseudotime_values : sequence of float or None, default None
        Optional ``(t0, t1)`` pair of pseudotime values to compare across all
        lineages. ``None`` uses the natural lineage start/end points.
    l2fc : float, default 0.0
        TREAT-style log2 fold-change threshold.
    key : str, default "tradeseq"
        Namespace under which :func:`fit_gam` wrote its outputs.

    Returns
    -------
    pandas.DataFrame
        Per-gene test statistics indexed by gene name.

        * ``global_=True, lineages=False`` → ``(waldStat, df, pvalue,
          logFClineage1, ..., logFClineageL)``.
        * ``global_=False, lineages=True`` →
          ``(waldStat_lineage1, df_lineage1, pvalue_lineage1, ...,
          logFClineage1, ...)``.
        * ``global_=True, lineages=True`` → both blocks concatenated.
    """
    if not (global_ or lineages):
        raise ValueError("at least one of `global_` or `lineages` must be True")
    if pseudotime_values is not None and len(pseudotime_values) != 2:
        raise ValueError("`pseudotime_values` must be a length-2 sequence")

    beta_all, sigma_all, dm, lpmatrix, pseudotime, conditions = fetch_test_arrays(
        adata, key=key
    )

    # R uses `length(grep("t[1-9]"))` to count lineages here (startVsEndTest.R:60).
    n_lineages = n_curves_from_dm(dm)
    if n_lineages == 0:
        raise ValueError(
            "Could not detect any lineage columns 't1, t2, ...' in the design matrix."
        )

    L = build_sve_contrast(
        lpmatrix=lpmatrix,
        dm=dm,
        pseudotime=pseudotime,
        conditions=conditions,
        n_lineages=n_lineages,
        pseudotime_values=pseudotime_values,
    )
    # ``inverse`` defaults to "QR" (waldTestFC default) — startVsEndTest never
    # overrides the ``inverse`` argument (cf. startVsEndTest.R:124).
    inverse = "QR"

    # Global Wald
    if global_:
        wald_global = run_wald_table(
            beta_all, sigma_all, L, l2fc=l2fc, inverse=inverse
        )

    # Per-lineage Wald
    if lineages:
        per_lineage = np.empty((beta_all.shape[0], 3 * n_lineages), dtype=float)
        for jj in range(n_lineages):
            L_jj = L[:, [jj]]  # keep 2-D
            block = run_wald_table(
                beta_all, sigma_all, L_jj, l2fc=l2fc, inverse=inverse
            )
            per_lineage[:, 3 * jj : 3 * jj + 3] = block

    # Fold changes (no abs).
    # R's idiom (startVsEndTest.R:189-194):
    #   fcAll <- apply(betaAll, 1, function(b) .getFoldChanges(b, L))
    #   colnames(fcAll) <- paste0("logFC", colnames(L))
    # i.e. one signed FC column per lineage contrast.
    fc_all = np.empty((beta_all.shape[0], n_lineages), dtype=float)
    for ii in range(beta_all.shape[0]):
        if np.isnan(beta_all[ii, :]).any():
            fc_all[ii, :] = np.nan
        else:
            fc_all[ii, :] = get_fold_changes(beta_all[ii, :], L)

    # Assemble output, matching R's column order.
    cols: dict[str, np.ndarray] = {}
    if global_:
        cols["waldStat"] = wald_global[:, 0]
        cols["df"] = wald_global[:, 1]
        cols["pvalue"] = wald_global[:, 2]
    if lineages:
        for jj in range(1, n_lineages + 1):
            cols[f"waldStat_lineage{jj}"] = per_lineage[:, 3 * (jj - 1) + 0]
            cols[f"df_lineage{jj}"] = per_lineage[:, 3 * (jj - 1) + 1]
            cols[f"pvalue_lineage{jj}"] = per_lineage[:, 3 * (jj - 1) + 2]
    for jj in range(1, n_lineages + 1):
        cols[f"logFClineage{jj}"] = fc_all[:, jj - 1]

    return pd.DataFrame(cols, index=adata.var_names)
