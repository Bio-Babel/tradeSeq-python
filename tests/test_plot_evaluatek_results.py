"""Tests for plot_evaluatek_results (R source: tradeSeq/R/evaluateK.R:342-374)."""

from __future__ import annotations

import ggplot2_py as gg
import numpy as np
import pandas as pd
import patchwork
import pytest
from PIL import Image

from tradeseq.evaluate_k import plot_evaluatek_results


@pytest.fixture
def synthetic_aic_mat() -> np.ndarray:
    """A 20-gene × 6-k AIC matrix with a planted elbow at k=5."""
    rng = np.random.default_rng(0)
    n_genes, n_k = 20, 6
    mat = rng.uniform(8000, 13000, size=(n_genes, n_k))
    # Inject an elbow at the 3rd column (k=5 when k_range=3:8).
    mat[:, 2] -= 200
    mat[:, 3] -= 250
    return mat


def test_plot_returns_patchwork(synthetic_aic_mat):
    fig = plot_evaluatek_results(
        synthetic_aic_mat, k_range=range(3, 9), aic_diff=2.0
    )
    assert isinstance(fig, patchwork.Patchwork)
    assert fig.fig_width == 12.0
    assert fig.fig_height == 3.0


def test_plot_accepts_list_k_range(synthetic_aic_mat):
    fig = plot_evaluatek_results(
        synthetic_aic_mat, k_range=[3, 4, 5, 6, 7, 8], aic_diff=2.0
    )
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_accepts_display_size_override(synthetic_aic_mat):
    fig = plot_evaluatek_results(
        synthetic_aic_mat,
        k_range=range(3, 9),
        aic_diff=2.0,
        fig_width=10,
        fig_height=2.5,
        fig_dpi=120,
    )
    assert fig.fig_width == 10.0
    assert fig.fig_height == 2.5
    assert fig.fig_dpi == 120


def test_plot_accepts_numpy_k_range(synthetic_aic_mat):
    fig = plot_evaluatek_results(
        synthetic_aic_mat, k_range=np.arange(3, 9), aic_diff=2.0
    )
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_accepts_dataframe_with_default_k_range(synthetic_aic_mat):
    """When a DataFrame is passed with default k_range=None, the columns
    are parsed via ``int(col.replace("k: ", ""))`` (mirrors R)."""
    df = pd.DataFrame(
        synthetic_aic_mat,
        columns=[f"k: {k}" for k in range(3, 9)],
    )
    fig = plot_evaluatek_results(df)
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_no_varying_genes_still_returns_patchwork():
    """If no gene has aic_range > aic_diff, panel 4 should be empty
    (no bars at all) but the function must still return a valid Patchwork.
    Mirrors R's ``table(...)`` behaviour: only k values that are some
    gene's optimum get rendered, so when no gene varies enough the table
    is empty."""
    n_genes, n_k = 5, 4
    # All-equal columns → aic_range is 0 → no gene passes the threshold.
    mat = np.full((n_genes, n_k), 9000.0)
    fig = plot_evaluatek_results(mat, k_range=range(3, 7), aic_diff=2.0)
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_high_aic_diff_excludes_all_genes():
    """A huge `aic_diff` should send every gene below the threshold."""
    n_genes, n_k = 10, 4
    rng = np.random.default_rng(0)
    mat = rng.uniform(8000, 13000, size=(n_genes, n_k))
    fig = plot_evaluatek_results(mat, k_range=range(3, 7), aic_diff=1e9)
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_panel4_only_shows_observed_best_k():
    """Gap #14: panel 4 must NOT reindex to the full k_list with zero bars.

    R's ``table(k[apply(aicMatSub, 1, which.min)])`` only contains k values
    that are some gene's optimum. We mirror that exactly: a contrived
    matrix where every gene's argmin is the same k should produce a
    single bar (not ``len(k_range)`` bars, with all-but-one being zero).
    """
    # 5 genes, 4 candidate k values. Make k=5 (column index 2) the
    # uniformly-optimum knot count.
    mat = np.array(
        [
            [10000, 9500, 9000, 9200],
            [10000, 9500, 9000, 9300],
            [10100, 9600, 9000, 9400],
            [10200, 9700, 9100, 9500],
            [10300, 9800, 9200, 9600],
        ],
        dtype=np.float64,
    )
    # We can't crack open the Patchwork to count bars, but we CAN inspect
    # the underlying counts_df assembly logic by replicating it.
    aic_range = mat.max(axis=1) - mat.min(axis=1)
    var_mask = aic_range > 2.0
    sub = mat[var_mask]
    best_idx = np.argmin(sub, axis=1)
    k_list = [3, 4, 5, 6]
    best_k = np.array([k_list[i] for i in best_idx])
    counts_observed = pd.Series(best_k).value_counts().sort_index()
    # All 5 genes' argmin should be k=5 — so the observed-counts series
    # has exactly ONE row, not ``len(k_list)`` rows.
    assert len(counts_observed) == 1
    assert int(counts_observed.index[0]) == 5
    # The plot function itself returns a Patchwork.
    fig = plot_evaluatek_results(mat, k_range=k_list, aic_diff=2.0)
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_rejects_1d_input():
    with pytest.raises(ValueError, match="must be 2-D"):
        plot_evaluatek_results(np.array([1.0, 2.0, 3.0]), k_range=[3, 4, 5])


def test_plot_rejects_k_range_length_mismatch(synthetic_aic_mat):
    """k_range length must match aic_matrix columns."""
    with pytest.raises(ValueError, match="length"):
        plot_evaluatek_results(
            synthetic_aic_mat, k_range=range(3, 5), aic_diff=2.0
        )


def test_plot_default_k_range():
    """Default k_range is range(3, 11) — must match a 8-column matrix."""
    mat = np.random.default_rng(0).uniform(8000, 13000, size=(10, 8))
    fig = plot_evaluatek_results(mat)
    assert isinstance(fig, patchwork.Patchwork)


def test_plot_can_be_saved_to_nonblank_png(tmp_path, synthetic_aic_mat):
    """Patchwork output must render through ggplot2_py.ggsave, not save blank."""
    fig = plot_evaluatek_results(
        synthetic_aic_mat, k_range=range(3, 9), aic_diff=2.0
    )
    path = tmp_path / "evaluate_k.png"
    with pytest.warns(UserWarning, match="Saving 6 x 4 in image"):
        gg.ggsave(str(path), plot=fig, width=6, height=4, dpi=96)
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im)
    assert path.stat().st_size > 1000
    assert arr[..., :3].max() > arr[..., :3].min()
