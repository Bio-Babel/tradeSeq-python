"""tradeSeq-python — trajectory-based differential expression analysis (NB-GAM).

Python port of the R/Bioconductor package tradeSeq 1.13.12 at upstream commit
02a9050. The container is :class:`anndata.AnnData`; see :doc:`docs/index.md`
for the slot schema.
"""

from __future__ import annotations

__version__ = "1.13.12"
__r_commit__ = "02a9050"

from .association_test import association_test
from .cascade import CascadeResult, cascade, plot_cascade
from .cluster import cluster_expression_patterns
from .condition_test import condition_test
from .datasets import load_paul15
from .diff_end_test import diff_end_test
from .evaluate_k import evaluate_k, evaluate_k2, plot_evaluatek_results
from .fit_gam import FittedGam, fit_gam, nknots
from .get_smoother import get_smoother_pvalues, get_smoother_test_stats
from .pattern_test import early_de_test, pattern_test
from .plot_gene_count import plot_gene_count
from .plot_smoothers import plot_smoothers
from .predict_cells import predict_cells
from .predict_smooth import predict_smooth
from .start_vs_end_test import start_vs_end_test

__all__ = [
    "load_paul15",
    "fit_gam",
    "nknots",
    "FittedGam",
    "CascadeResult",
    "evaluate_k",
    "evaluate_k2",
    "plot_evaluatek_results",
    "association_test",
    "start_vs_end_test",
    "diff_end_test",
    "pattern_test",
    "early_de_test",
    "condition_test",
    "predict_cells",
    "predict_smooth",
    "get_smoother_pvalues",
    "get_smoother_test_stats",
    "plot_smoothers",
    "plot_gene_count",
    "plot_cascade",
    "cluster_expression_patterns",
    "cascade",
]
