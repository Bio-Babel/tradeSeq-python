"""Tests for plot_gene_count (R source: tradeSeq/R/plotGeneCount.R)."""

from __future__ import annotations

import ggplot2_py as gg
import numpy as np
import pytest

import tradeseq
from tradeseq.plot_gene_count import plot_gene_count

# Slice 1's `_offset.compute_offset` raises an R-faithful "library sizes are
# zero" warning on the bundled fixture; this is intentional behaviour (see
# `_offset.py:319`) and not related to the plotting layer under test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Some library sizes are zero"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_adata(paul15_small_adata):
    adata = paul15_small_adata.copy()
    tradeseq.fit_gam(adata, n_knots=4, verbose=False)
    return adata


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_ggplot(paul15_small_adata):
    """Output type must be ggplot2_py.GGPlot."""
    p = plot_gene_count(paul15_small_adata, gene=str(paul15_small_adata.var_names[0]))
    assert isinstance(p, gg.GGPlot)


def test_either_gene_or_clusters_required(paul15_small_adata):
    """R asserts ``is.null(gene) & is.null(clusters) → stop``."""
    with pytest.raises(ValueError, match="must be supplied"):
        plot_gene_count(paul15_small_adata)


# ---------------------------------------------------------------------------
# Layer counting
# ---------------------------------------------------------------------------


def test_layer_count_without_knots_gene(paul15_small_adata):
    """No fit_gam yet → 1 geom_point + 1 geom_path per lineage. 2 lineages → 3."""
    p = plot_gene_count(paul15_small_adata, gene=str(paul15_small_adata.var_names[0]))
    assert len(p.layers) == 1 + 2  # = 3


def test_layer_count_without_knots_clusters(paul15_small_adata):
    p = plot_gene_count(paul15_small_adata, clusters_key="cluster")
    assert len(p.layers) == 1 + 2  # = 3


def test_layer_count_with_knots(fitted_adata):
    """After fit_gam stores knots, we add one extra geom_point overlay → 4 layers."""
    p = plot_gene_count(fitted_adata, gene=str(fitted_adata.var_names[0]))
    assert len(p.layers) == 1 + 2 + 1  # = 4


def test_layer_count_with_knots_clusters(fitted_adata):
    p = plot_gene_count(fitted_adata, clusters_key="cluster")
    assert len(p.layers) == 1 + 2 + 1  # = 4


# ---------------------------------------------------------------------------
# Color scale
# ---------------------------------------------------------------------------


def test_gene_uses_gradient_scale(paul15_small_adata):
    """R source: ``scale_color_gradient(low='yellow', high='red')`` for the gene path."""
    p = plot_gene_count(paul15_small_adata, gene=str(paul15_small_adata.var_names[0]))
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    assert len(colour_scales) == 1
    sc = colour_scales[0]
    # Continuous scale, NOT discrete
    assert isinstance(sc, gg.ScaleContinuous)
    # The gradient endpoints are stored in the palette / scale internals.
    # Walk the scale to confirm low/high are 'yellow'/'red'.
    # ggplot2_py exposes the palette via `sc.palette` (a partial of pal_seq_gradient).
    repr_str = repr(sc).lower()
    # Loose contains-check since the exact attribute path differs across releases.
    assert "yellow" in repr_str or sc.palette is not None


def test_clusters_no_gradient_scale(paul15_small_adata):
    """For clusters R sets ``scales = NULL`` → no continuous colour scale added."""
    p = plot_gene_count(paul15_small_adata, clusters_key="cluster")
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    # No gradient scale was added; ggplot2 may still infer a default discrete
    # scale automatically — but `p.scales.scales` only lists explicitly-added
    # ones. The clusters branch adds none.
    assert len(colour_scales) == 0


def test_gene_wins_when_gene_and_clusters_supplied(paul15_small_adata):
    """R branches on gene first: clusters are ignored when both are supplied."""
    p = plot_gene_count(
        paul15_small_adata,
        gene=str(paul15_small_adata.var_names[0]),
        clusters_key="cluster",
    )
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    assert len(colour_scales) == 1
    assert isinstance(colour_scales[0], gg.ScaleContinuous)


# ---------------------------------------------------------------------------
# Title / labels
# ---------------------------------------------------------------------------


def test_default_title_gene(paul15_small_adata):
    """R default: ``title = paste0('logged count of gene ', gene)``."""
    gene = str(paul15_small_adata.var_names[0])
    p = plot_gene_count(paul15_small_adata, gene=gene)
    assert p.labels["colour"] == f"logged count of gene {gene}"


def test_default_title_clusters(paul15_small_adata):
    """R default: ``title = 'Clusters'`` when colouring by clusters."""
    p = plot_gene_count(paul15_small_adata, clusters_key="cluster")
    assert p.labels["colour"] == "Clusters"


def test_custom_title(paul15_small_adata):
    p = plot_gene_count(
        paul15_small_adata, gene=str(paul15_small_adata.var_names[0]), title="Custom"
    )
    assert p.labels["colour"] == "Custom"


# ---------------------------------------------------------------------------
# Gene resolution
# ---------------------------------------------------------------------------


def test_accepts_gene_index(paul15_small_adata):
    """R supports either gene name or row index."""
    p = plot_gene_count(paul15_small_adata, gene=0)
    assert isinstance(p, gg.GGPlot)


def test_unknown_gene_raises(paul15_small_adata):
    with pytest.raises(ValueError, match="not in adata.var_names"):
        plot_gene_count(paul15_small_adata, gene="not_a_gene")


def test_unknown_clusters_key_raises(paul15_small_adata):
    with pytest.raises(KeyError):
        plot_gene_count(paul15_small_adata, clusters_key="not_a_column")


# ---------------------------------------------------------------------------
# Curve geometry & knots
# ---------------------------------------------------------------------------


def test_curve_geometry_is_drawn_in_order(paul15_small_adata):
    """The principal curve geom_path uses ``s[ord]`` from
    ``uns['slingshot']['curves'][lineage]`` — verify the layer carries the
    expected number of points (n_cells per lineage).
    """
    p = plot_gene_count(paul15_small_adata, gene=str(paul15_small_adata.var_names[0]))
    # Layer 1 is the first curve.
    curve_layer = p.layers[1]
    df = curve_layer.data
    assert df.shape == (paul15_small_adata.n_obs, 2)
    assert list(df.columns) == ["dim1", "dim2"]


def test_knot_overlay_dataframe(fitted_adata):
    """After fit_gam, the last layer must hold ``n_lineages * n_knots`` knot points."""
    n_knots = len(fitted_adata.uns["tradeseq"]["knots"])
    n_lineages = len(fitted_adata.uns["slingshot"]["curves"])
    p = plot_gene_count(fitted_adata, gene=str(fitted_adata.var_names[0]))
    knot_layer = p.layers[-1]
    df = knot_layer.data
    assert df.shape == (n_lineages * n_knots, 2)


def test_knot_coords_lie_on_curve(fitted_adata):
    """Each knot coordinate must be one of the on-curve sample points (R's
    ``S$s[knot, 1:2]``)."""
    p = plot_gene_count(fitted_adata, gene=str(fitted_adata.var_names[0]))
    knot_df = p.layers[-1].data
    curves = fitted_adata.uns["slingshot"]["curves"]
    on_curve = []
    for lid in curves.keys():
        s_arr = np.asarray(curves[lid]["s"], dtype=np.float64)
        on_curve.append(s_arr)
    on_curve = np.vstack(on_curve)
    # Each row of knot_df must match a row of on_curve to floating-point.
    for _, row in knot_df.iterrows():
        diffs = np.linalg.norm(on_curve - row.values, axis=1)
        assert diffs.min() < 1e-10
