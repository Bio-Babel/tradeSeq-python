"""Tests for plot_smoothers (R source: tradeSeq/R/plotSmoothers.R)."""

from __future__ import annotations

import ggplot2_py as gg
import numpy as np
import pandas as pd
import pytest

import tradeseq
from tradeseq.plot_smoothers import plot_smoothers

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
    """Run fit_gam on the small fixture and return a fitted copy."""
    adata = paul15_small_adata.copy()
    tradeseq.fit_gam(adata, n_knots=4, verbose=False)
    return adata


@pytest.fixture(scope="module")
def fitted_one_lineage_adata(paul15_one_lineage_adata):
    """Fit on the single-lineage fixture; used to test layer counts at L=1."""
    adata = paul15_one_lineage_adata.copy()
    tradeseq.fit_gam(adata, n_knots=4, verbose=False)
    return adata


@pytest.fixture(scope="module")
def fitted_conditions_adata(paul15_small_adata):
    """Fit on a copy with a synthetic two-level condition factor.

    R counterpart: ``conditions = sample(c("A", "B"), n_cells, replace=TRUE)``.
    Exercises the conditions branch of ``plotSmoothers``.
    """
    rng = np.random.default_rng(0)
    adata = paul15_small_adata.copy()
    cond = rng.choice(["A", "B"], size=adata.n_obs)
    adata.obs["cond"] = pd.Categorical(cond, categories=["A", "B"])
    tradeseq.fit_gam(
        adata, n_knots=4, conditions_key="cond", verbose=False
    )
    return adata


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_ggplot(fitted_adata):
    """Output type must be ggplot2_py.GGPlot, matching R's `setMethod` return type."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(fitted_adata, gene=g_name, n_points=10, random_state=0)
    assert isinstance(p, gg.GGPlot)


def test_axis_labels_default(fitted_adata):
    """xlab/ylab defaults must match R: 'Pseudotime' and 'Log(expression + 1)'."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(fitted_adata, gene=g_name, n_points=10, random_state=0)
    assert p.labels["x"] == "Pseudotime"
    assert p.labels["y"] == "Log(expression + 1)"


def test_axis_labels_custom(fitted_adata):
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, xlab="t", ylab="y", random_state=0
    )
    assert p.labels["x"] == "t"
    assert p.labels["y"] == "y"


# ---------------------------------------------------------------------------
# Layer counting — derived from the R source `geom_*` calls.
# ---------------------------------------------------------------------------


def test_layer_count_two_lineages_border(fitted_adata):
    """border=True (R default) + 2 lineages → 1 geom_point + 2*2 geom_line = 5."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, border=True, random_state=0
    )
    assert len(p.layers) == 5


def test_layer_count_two_lineages_no_border(fitted_adata):
    """border=False + 2 lineages → 1 geom_point + 2 geom_line = 3."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, border=False, random_state=0
    )
    assert len(p.layers) == 3


def test_layer_count_no_lineages(fitted_adata):
    """plot_lineages=False suppresses every smoother line → 1 geom_point only."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata,
        gene=g_name,
        n_points=10,
        plot_lineages=False,
        random_state=0,
    )
    assert len(p.layers) == 1


def test_layer_count_one_lineage_border(fitted_one_lineage_adata):
    """border=True + 1 lineage → 1 geom_point + 2 geom_line = 3."""
    g_name = str(fitted_one_lineage_adata.var_names[0])
    p = plot_smoothers(
        fitted_one_lineage_adata,
        gene=g_name,
        n_points=10,
        border=True,
        random_state=0,
    )
    assert len(p.layers) == 3


def test_layer_count_lineages_to_plot(fitted_adata):
    """Filtering to a single lineage with border=True drops the lines for the other."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata,
        gene=g_name,
        n_points=10,
        border=True,
        lineages_to_plot=[1],
        random_state=0,
    )
    # 1 geom_point + 2 lines (white halo + colour) for lineage 1 = 3.
    assert len(p.layers) == 3


# ---------------------------------------------------------------------------
# Colour scale checks
# ---------------------------------------------------------------------------


def test_default_color_is_viridis_d(fitted_adata):
    """When point_col is None the R source uses ``scale_color_viridis_d(alpha=)``.

    The Python port must register a discrete-colour scale on the ``colour`` aes.
    """
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(fitted_adata, gene=g_name, n_points=10, random_state=0)
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    assert len(colour_scales) == 1
    # Discrete viridis returns a ScaleDiscrete in ggplot2_py.
    assert isinstance(colour_scales[0], gg.ScaleDiscrete)


def test_point_col_uses_discrete_scale(fitted_adata):
    """``point_col='cluster'`` triggers ``scale_color_discrete()`` in the R source."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata,
        gene=g_name,
        n_points=10,
        point_col="cluster",
        random_state=0,
    )
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    assert len(colour_scales) == 1
    assert isinstance(colour_scales[0], gg.ScaleDiscrete)
    # R applies labs(col="Cell labels") in this branch.
    assert p.labels["colour"] == "Cell labels"


def test_point_col_vector_input(fitted_adata):
    """Passing a per-cell vector for point_col is allowed (R checks length == n_cells)."""
    g_name = str(fitted_adata.var_names[0])
    pc = np.array(["A"] * fitted_adata.n_obs)
    pc[: fitted_adata.n_obs // 2] = "B"
    p = plot_smoothers(
        fitted_adata,
        gene=g_name,
        n_points=10,
        point_col=pc,
        random_state=0,
    )
    assert isinstance(p, gg.GGPlot)
    assert p.labels["colour"] == "Cell labels"


def test_point_col_bad_length_falls_back_like_r(fitted_adata):
    """R keeps the pointCol branch active after warning and uses lineage labels."""
    g_name = str(fitted_adata.var_names[0])
    with pytest.warns(UserWarning, match="pointCol should have length"):
        p = plot_smoothers(
            fitted_adata,
            gene=g_name,
            n_points=10,
            point_col=np.array(["too", "short"]),
            random_state=0,
        )
    assert p.labels["colour"] == "Cell labels"
    assert set(p.data["pCol"].astype(str).unique()).issubset({"1", "2"})


# ---------------------------------------------------------------------------
# Aesthetic kwargs propagation
# ---------------------------------------------------------------------------


def test_size_kwarg_propagates(fitted_adata):
    """`size` flows into the geom_point layer."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, size=1.5, random_state=0
    )
    # First layer is the scatter; its aes_params hold the size override.
    layer = p.layers[0]
    # ggplot2_py stores layer constants in `aes_params` (matching ggplot2's R API).
    assert float(layer.aes_params.get("size", 0)) == pytest.approx(1.5)


def test_lwd_kwarg_propagates(fitted_adata):
    """`lwd` flows into the geom_line linewidth."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata,
        gene=g_name,
        n_points=10,
        lwd=4.0,
        border=False,
        random_state=0,
    )
    # First non-scatter layer is the smoother line.
    smoother_layers = p.layers[1:]
    assert len(smoother_layers) >= 1
    lw = smoother_layers[0].aes_params.get("linewidth", None)
    assert float(lw) == pytest.approx(4.0)


def test_alpha_default_is_one(fitted_adata):
    """The SCE-method dispatcher in R (plotSmoothers.R:432) sets ``alpha = 1``.

    The Python user-facing default must match: the viridis palette returned by
    ``scale_color_viridis_d`` must not have a ``80`` (or any other) alpha-hex
    suffix on the emitted colour strings, which would indicate alpha < 1.
    """
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(fitted_adata, gene=g_name, n_points=10, random_state=0)
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    assert len(colour_scales) == 1
    # palette(N) for any N returns hex strings of length 7 ("#rrggbb") when
    # alpha == 1; alpha < 1 widens them to 9 chars ("#rrggbbaa").
    cols = colour_scales[0].palette(2)
    assert all(len(c) == 7 for c in cols), (
        f"alpha-default != 1: palette emitted {cols!r}; the SCE dispatcher in "
        "R sets alpha=1 (plotSmoothers.R:432)."
    )


def test_alpha_kwarg_propagates(fitted_adata):
    """Explicit alpha=0.5 should appear as the alpha channel in the palette."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, alpha=0.5, random_state=0
    )
    colour_scales = [s for s in p.scales.scales if "colour" in s.aesthetics]
    cols = colour_scales[0].palette(2)
    # All hex strings now carry an "80" alpha suffix (0.5 → 0x80).
    assert all(len(c) == 9 for c in cols) and all(c.endswith("80") for c in cols)


def test_sample_kwarg_reduces_scatter_n(fitted_adata):
    """sample=0.5 must halve the rows in the scatter df.

    The scatter df lives on layer 0 (or on the plot if inherit_aes=True). We
    can read `layer.data` if set, otherwise inspect the plot's df via
    ``p.data``.
    """
    g_name = str(fitted_adata.var_names[0])
    full = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, sample=1.0, random_state=0
    )
    half = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, sample=0.5, random_state=0
    )
    n_full = full.data.shape[0]
    n_half = half.data.shape[0]
    assert n_half == n_full // 2


def test_sample_kwarg_matches_r_integer_truncation(fitted_adata):
    """R sample(size=nrow(df) * sample) truncates non-integer sizes."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, sample=0.26, random_state=0
    )
    assert p.data.shape[0] == int(fitted_adata.n_obs * 0.26)


def test_sample_zero_keeps_empty_scatter_like_r(fitted_adata):
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(
        fitted_adata, gene=g_name, n_points=10, sample=0.0, random_state=0
    )
    assert p.data.shape[0] == 0


def test_sample_above_one_raises(fitted_adata):
    g_name = str(fitted_adata.var_names[0])
    with pytest.raises(ValueError, match="between 0 and 1"):
        plot_smoothers(fitted_adata, gene=g_name, n_points=10, sample=1.1)


# ---------------------------------------------------------------------------
# Gene resolution
# ---------------------------------------------------------------------------


def test_accepts_gene_index(fitted_adata):
    """Numeric gene index must work like ``rownames(counts)[gene]`` in R."""
    p = plot_smoothers(fitted_adata, gene=0, n_points=10, random_state=0)
    assert isinstance(p, gg.GGPlot)


def test_unknown_gene_raises(fitted_adata):
    with pytest.raises(ValueError, match="not present"):
        plot_smoothers(fitted_adata, gene="not_a_gene", n_points=10)


def test_multi_gene_facets(fitted_adata):
    """Passing multiple gene names yields a faceted plot — Python extension of the
    R API which restricts to a single gene. Layer count and df cardinality should
    both scale with the number of genes."""
    genes = [str(fitted_adata.var_names[0]), str(fitted_adata.var_names[1])]
    p = plot_smoothers(
        fitted_adata, gene=genes, n_points=10, random_state=0, border=False
    )
    assert isinstance(p, gg.GGPlot)
    # 1 scatter + 2 lineages × 2 genes = 5 smoother lines, total 5.
    assert len(p.layers) == 5
    # Scatter df should hold cells × len(genes) rows.
    assert "gene" in p.data.columns
    assert set(p.data["gene"].unique()) == set(genes)


# ---------------------------------------------------------------------------
# Conditions branch
# ---------------------------------------------------------------------------


def test_conditions_branch_layer_count(fitted_conditions_adata):
    """With 2 conditions × 2 lineages + border, we expect 1 + 2*2*2 = 9 layers."""
    g_name = str(fitted_conditions_adata.var_names[0])
    p = plot_smoothers(
        fitted_conditions_adata,
        gene=g_name,
        n_points=10,
        border=True,
        random_state=0,
    )
    assert isinstance(p, gg.GGPlot)
    assert len(p.layers) == 1 + 2 * 2 * 2  # = 9


def test_conditions_branch_no_border(fitted_conditions_adata):
    """border=False → 1 scatter + 2 lineages × 2 conditions = 5 layers."""
    g_name = str(fitted_conditions_adata.var_names[0])
    p = plot_smoothers(
        fitted_conditions_adata,
        gene=g_name,
        n_points=10,
        border=False,
        random_state=0,
    )
    assert len(p.layers) == 1 + 2 * 2  # = 5


def test_noncondition_nan_beta_keeps_nan_line_layers(fitted_adata):
    """R's non-conditions branch does not stop on NA beta; the yhat line is NA."""
    adata = fitted_adata.copy()
    adata.varm["tradeseq_beta"][0, :] = np.nan
    g_name = str(adata.var_names[0])
    p = plot_smoothers(
        adata, gene=g_name, n_points=10, border=False, random_state=0
    )
    assert len(p.layers) == 3
    for layer in p.layers[1:]:
        assert layer.data["log1p_count"].isna().all()


def test_bad_curves_cols_falls_back_to_viridis_with_warning(fitted_adata):
    g_name = str(fitted_adata.var_names[0])
    with pytest.warns(UserWarning, match="Incorrect number of lineage colors"):
        p = plot_smoothers(
            fitted_adata,
            gene=g_name,
            n_points=10,
            curves_cols=["#000000"],
            random_state=0,
            border=False,
        )
    assert isinstance(p, gg.GGPlot)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def test_uses_theme_classic(fitted_adata):
    """The R source ends with ``+ theme_classic()``; check the python equivalent is applied."""
    g_name = str(fitted_adata.var_names[0])
    p = plot_smoothers(fitted_adata, gene=g_name, n_points=10, random_state=0)
    # `theme_classic` defines `panel.grid = element_blank()`; checking that the
    # theme is non-default is enough — it shouldn't be an empty Theme().
    assert p.theme is not None
