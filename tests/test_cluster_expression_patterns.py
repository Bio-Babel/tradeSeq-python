"""Tests for tradeseq.cluster_expression_patterns (R source: clusterExpressionPatterns.R).

Tier-3 deviation: the R implementation runs
``clusterExperiment::RSEC`` (consensus clustering with internal PCA), which
has no Python equivalent. The Python port delegates to
``sklearn.cluster.AgglomerativeClustering`` with Ward linkage and uses
``sklearn.metrics.silhouette_score`` to pick ``n_clusters`` when not
supplied.

Return-structure parity: R returns ``list(rsec=<ClusterExperiment>,
yhatScaled=<matrix>)`` (clusterExpressionPatterns.R:64, 133). The Python port
returns a :class:`tradeseq.cluster.ClusterResult` NamedTuple with
``cluster_labels``, ``yhat_scaled``, and a ``long_df`` view.

These tests therefore validate:

* The :class:`ClusterResult` NamedTuple has the expected fields and shapes
  (``yhat_scaled`` mirrors R's ``yhatScaled`` matrix, ``cluster_labels`` is a
  per-gene Series).
* Column labels on ``yhat_scaled`` follow R's ``l{ii}:t{t}`` (no conditions)
  or ``l{ii}_{kk}:t{t}`` (conditions) convention
  (clusterExpressionPatterns.R:50, 118).
* The silhouette-selected ``k`` lies in ``[2, n_genes]``.
* Explicit ``n_clusters`` overrides the search.
* String gene labels and integer gene indices are both accepted.
* The output is bit-deterministic given a fixed ``random_state``.
* The list-mode branch (dict[str, FittedGam], clusterExpressionPatterns.R:
  225-251) and the conditions branch (clusterExpressionPatterns.R:67-135) both
  produce output matrices of the expected shape and column-name pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tradeseq
from tradeseq.cluster import ClusterResult, cluster_expression_patterns

GENES = [
    "Acin1",
    "Actb",
    "Alas1",
    "Anapc5",
    "Apoe",
    "Ank",
    "Adam9",
    "Ak2",
    "Alad",
    "Aldoa",
]


@pytest.fixture(scope="module")
def fitted_paul15_cluster(paul15_adata):
    """Fit GAMs on a 10-gene panel of the bundled Paul-2015 fixture."""
    a = paul15_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    keep = [g for g in GENES if g in a.var_names]
    out = tradeseq.fit_gam(
        a, genes=keep, n_knots=6, verbose=False, _w_samp=w_samp, copy=True,
    )
    return out, keep


@pytest.fixture(scope="module")
def fitted_paul15_cluster_list(paul15_adata):
    """List-mode fit_gam(sce=False) over the same 10-gene panel.

    Used to exercise the list-mode branch
    (clusterExpressionPatterns.R:225-251).
    """
    a = paul15_adata.copy()
    rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = rng.multinomial(1, probs[i])
    keep = [g for g in GENES if g in a.var_names]
    fits = tradeseq.fit_gam(
        a, genes=keep, n_knots=6, verbose=False, _w_samp=w_samp, sce=False,
    )
    return fits, keep


@pytest.fixture(scope="module")
def fitted_paul15_cluster_conditions(paul15_adata):
    """Fit GAMs with a synthetic 2-level condition factor.

    Exercises the conditions branch
    (clusterExpressionPatterns.R:67-135). The synthetic ``cond`` column is
    fixed-seeded so the test is deterministic.
    """
    a = paul15_adata.copy()
    rng = np.random.default_rng(0)
    cond = rng.choice(["A", "B"], size=a.n_obs)
    a.obs["batch"] = pd.Categorical(cond, categories=["A", "B"])
    w_rng = np.random.default_rng(7)
    w_samp = np.zeros_like(a.obsm["cell_weights"], dtype=np.int64)
    probs = a.obsm["cell_weights"] / a.obsm["cell_weights"].sum(axis=1, keepdims=True)
    for i in range(a.n_obs):
        w_samp[i] = w_rng.multinomial(1, probs[i])
    keep = [g for g in GENES if g in a.var_names]
    out = tradeseq.fit_gam(
        a, genes=keep, n_knots=4, verbose=False, _w_samp=w_samp,
        conditions_key="batch", copy=True,
    )
    return out, keep


# ---------------------------------------------------------------------------
# Return-type structure (Gap #23)
# ---------------------------------------------------------------------------


def test_return_is_cluster_result(fitted_paul15_cluster):
    """The return value is a ClusterResult NamedTuple with the three fields."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(adata, n_points=12, genes=genes)
    assert isinstance(res, ClusterResult)
    # NamedTuple fields
    assert res._fields == ("cluster_labels", "yhat_scaled", "long_df")
    # And each field is the expected type
    assert isinstance(res.cluster_labels, pd.Series)
    assert isinstance(res.yhat_scaled, pd.DataFrame)
    assert isinstance(res.long_df, pd.DataFrame)


def test_cluster_labels_series_indexed_by_gene(fitted_paul15_cluster):
    """``cluster_labels`` is a Series indexed by gene name with int values."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(adata, n_points=12, genes=genes)
    cl = res.cluster_labels
    # Index is the gene labels (preserving order)
    assert list(cl.index) == list(genes)
    # All ints
    assert cl.dtype.kind in "iu"
    assert len(cl) == len(genes)


def test_yhat_scaled_shape_matches_r_grid(fitted_paul15_cluster):
    """``yhat_scaled`` has shape ``(n_genes, n_lineages * n_points)``."""
    adata, genes = fitted_paul15_cluster
    n_points = 12
    n_lineages = 2  # Paul-2015 has two slingshot lineages
    res = cluster_expression_patterns(adata, n_points=n_points, genes=genes)
    assert res.yhat_scaled.shape == (len(genes), n_lineages * n_points)
    assert list(res.yhat_scaled.index) == list(genes)


def test_yhat_scaled_columns_follow_r_pattern_no_conditions(fitted_paul15_cluster):
    """Column labels follow R's ``l{ii}:t{t}`` (clusterExpressionPatterns.R:50)."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(adata, n_points=12, genes=genes)
    cols = list(res.yhat_scaled.columns)
    # 24 distinct labels (2 lineages * 12 points)
    assert len(cols) == 24
    starts = {c.split(":", 1)[0] for c in cols}
    assert starts == {"l1", "l2"}
    # All labels match ``l{ii}:t{t}`` with t numeric
    for c in cols:
        head, tail = c.split(":", 1)
        assert head.startswith("l")
        assert tail.startswith("t")
        # convertible to float once t prefix is stripped
        float(tail[1:])


def test_long_df_columns(fitted_paul15_cluster):
    """``long_df`` exposes the legacy 4-column tidy view."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(adata, n_points=12, genes=genes)
    df = res.long_df
    assert sorted(df.columns.tolist()) == ["cluster", "gene", "profile", "time_point"]


def test_long_df_shape(fitted_paul15_cluster):
    """``long_df`` row count = n_genes * (n_lineages * n_points)."""
    adata, genes = fitted_paul15_cluster
    n_points = 12
    n_lineages = 2
    res = cluster_expression_patterns(adata, n_points=n_points, genes=genes)
    assert res.long_df.shape[0] == len(genes) * n_lineages * n_points
    per_gene_counts = res.long_df.groupby("gene", observed=True).size()
    assert (per_gene_counts == n_lineages * n_points).all()


def test_cluster_labels_consistent_with_long_df(fitted_paul15_cluster):
    """The Series ``cluster_labels`` matches the ``cluster`` values in ``long_df``."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=3
    )
    # For each gene, the cluster value in long_df must equal cluster_labels[gene]
    df = res.long_df
    for g in genes:
        per_gene = df.loc[df["gene"] == g, "cluster"].unique()
        assert len(per_gene) == 1
        assert int(per_gene[0]) == int(res.cluster_labels.loc[g])


def test_yhat_scaled_consistent_with_long_df(fitted_paul15_cluster):
    """``yhat_scaled.loc[g, col]`` equals the matching ``long_df`` profile value."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=3
    )
    df = res.long_df
    wide = res.yhat_scaled
    # Check a handful of (gene, time_point) cells
    for g in genes[:3]:
        for tp in wide.columns[:5]:
            wide_val = float(wide.at[g, tp])
            long_val = float(
                df.loc[(df["gene"] == g) & (df["time_point"] == tp), "profile"].iloc[0]
            )
            assert abs(wide_val - long_val) < 1e-12


# ---------------------------------------------------------------------------
# Behavioural tests (silhouette, validation, determinism)
# ---------------------------------------------------------------------------


def test_silhouette_selects_valid_k(fitted_paul15_cluster):
    """Silhouette-selected ``k`` lies in ``[2, n_genes]``."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(adata, n_points=12, genes=genes)
    n_clusters = res.cluster_labels.nunique()
    assert 2 <= n_clusters <= len(genes)


def test_explicit_n_clusters(fitted_paul15_cluster):
    """Explicit ``n_clusters`` overrides the silhouette sweep."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=4
    )
    assert res.cluster_labels.nunique() == 4


def test_integer_gene_indices(fitted_paul15_cluster):
    """Passing genes as 0-based integer indices yields identical labels."""
    adata, genes = fitted_paul15_cluster
    indices = [adata.var_names.get_loc(g) for g in genes]
    res_str = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=3
    )
    res_int = cluster_expression_patterns(
        adata, n_points=12, genes=indices, n_clusters=3
    )
    # Same gene order in cluster_labels
    pd.testing.assert_index_equal(
        res_str.cluster_labels.index, res_int.cluster_labels.index
    )
    np.testing.assert_array_equal(
        res_str.cluster_labels.to_numpy(),
        res_int.cluster_labels.to_numpy(),
    )


def test_profile_row_standardised(fitted_paul15_cluster):
    """Each gene's row in ``yhat_scaled`` has mean ~ 0 and SD ~ 1 (R's scale)."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=3
    )
    for g in genes:
        row = res.yhat_scaled.loc[g].to_numpy()
        assert abs(row.mean()) < 1e-8
        assert abs(row.std(ddof=1) - 1.0) < 1e-8


def test_deterministic_given_random_state(fitted_paul15_cluster):
    """Two calls with the same ``random_state`` produce identical output."""
    adata, genes = fitted_paul15_cluster
    res1 = cluster_expression_patterns(
        adata, n_points=12, genes=genes, random_state=176201
    )
    res2 = cluster_expression_patterns(
        adata, n_points=12, genes=genes, random_state=176201
    )
    pd.testing.assert_series_equal(res1.cluster_labels, res2.cluster_labels)
    pd.testing.assert_frame_equal(res1.yhat_scaled, res2.yhat_scaled)
    pd.testing.assert_frame_equal(res1.long_df, res2.long_df)


def test_unknown_gene_raises(fitted_paul15_cluster):
    """A string gene label not present in ``adata.var_names`` raises ``ValueError``."""
    adata, genes = fitted_paul15_cluster
    with pytest.raises(ValueError, match="Not all gene IDs"):
        cluster_expression_patterns(
            adata, n_points=12, genes=[genes[0], "NotAGeneXYZ"]
        )


def test_n_clusters_validation(fitted_paul15_cluster):
    """``n_clusters`` outside ``[2, n_genes]`` raises."""
    adata, genes = fitted_paul15_cluster
    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        cluster_expression_patterns(
            adata, n_points=12, genes=genes, n_clusters=1
        )
    with pytest.raises(ValueError, match="cannot exceed n_genes"):
        cluster_expression_patterns(
            adata, n_points=12, genes=genes, n_clusters=len(genes) + 1
        )


def test_n_points_validation(fitted_paul15_cluster):
    """``n_points < 2`` raises early."""
    adata, genes = fitted_paul15_cluster
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        cluster_expression_patterns(adata, n_points=1, genes=genes)


def test_too_few_genes_raises(fitted_paul15_cluster):
    """Clustering with only one gene is meaningless and raises."""
    adata, genes = fitted_paul15_cluster
    with pytest.raises(ValueError, match="at least 2 genes"):
        cluster_expression_patterns(adata, n_points=12, genes=[genes[0]])


def test_profile_values_finite(fitted_paul15_cluster):
    """No NaN/Inf in the ``yhat_scaled`` matrix for a well-converged fit."""
    adata, genes = fitted_paul15_cluster
    res = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=3
    )
    assert np.isfinite(res.yhat_scaled.to_numpy()).all()


def test_n_points_changes_grid(fitted_paul15_cluster):
    """Different ``n_points`` changes ``yhat_scaled`` column count proportionally."""
    adata, genes = fitted_paul15_cluster
    res_small = cluster_expression_patterns(
        adata, n_points=12, genes=genes, n_clusters=3
    )
    res_large = cluster_expression_patterns(
        adata, n_points=24, genes=genes, n_clusters=3
    )
    assert res_large.yhat_scaled.shape[1] == 2 * res_small.yhat_scaled.shape[1]


# ---------------------------------------------------------------------------
# Gap #21 -- list-mode branch (clusterExpressionPatterns.R:225-251)
# ---------------------------------------------------------------------------


def test_list_mode_returns_cluster_result(fitted_paul15_cluster_list):
    """Passing a dict[str, FittedGam] dispatches to the list branch."""
    fits, genes = fitted_paul15_cluster_list
    res = cluster_expression_patterns(fits, n_points=8, genes=genes, n_clusters=3)
    assert isinstance(res, ClusterResult)
    # cluster_labels indexed by gene
    assert list(res.cluster_labels.index) == list(genes)
    # yhat_scaled has the same gene index and n_lineages * n_points columns
    assert res.yhat_scaled.shape == (len(genes), 2 * 8)


def test_list_mode_skips_failed_reference(fitted_paul15_cluster_list):
    """A failed first fit must not prevent using the next successful reference."""
    fits, genes = fitted_paul15_cluster_list
    fits = dict(fits)
    keys = list(fits.keys())
    fits[keys[0]] = None
    usable_genes = [g for g in genes if g != keys[0]]
    res = cluster_expression_patterns(
        fits, n_points=8, genes=usable_genes, n_clusters=3
    )
    assert isinstance(res, ClusterResult)
    assert list(res.cluster_labels.index) == usable_genes


def test_list_mode_failed_selected_gene_raises(fitted_paul15_cluster_list):
    """Failed selected fits should not be converted to flat zero profiles."""
    fits, genes = fitted_paul15_cluster_list
    fits = dict(fits)
    fits[genes[0]] = None
    with pytest.raises(ValueError, match="non-finite fitted expression"):
        cluster_expression_patterns(
            fits, n_points=8, genes=genes, n_clusters=3
        )


def test_list_mode_column_labels(fitted_paul15_cluster_list):
    """List-mode column labels follow ``l{ii}:t{t}`` like the AnnData branch."""
    fits, genes = fitted_paul15_cluster_list
    res = cluster_expression_patterns(fits, n_points=8, genes=genes, n_clusters=3)
    cols = list(res.yhat_scaled.columns)
    starts = {c.split(":", 1)[0] for c in cols}
    assert starts == {"l1", "l2"}
    for c in cols:
        head, tail = c.split(":", 1)
        assert head.startswith("l")
        assert tail.startswith("t")


def test_list_mode_int_indices(fitted_paul15_cluster_list):
    """Integer indices into ``names(models)`` are accepted (R: id <- genes)."""
    fits, genes = fitted_paul15_cluster_list
    keys = list(fits.keys())
    indices = [keys.index(g) for g in genes]
    res_str = cluster_expression_patterns(
        fits, n_points=8, genes=genes, n_clusters=3
    )
    res_int = cluster_expression_patterns(
        fits, n_points=8, genes=indices, n_clusters=3
    )
    pd.testing.assert_series_equal(
        res_str.cluster_labels.reset_index(drop=True),
        res_int.cluster_labels.reset_index(drop=True),
    )


def test_list_mode_unknown_gene_raises(fitted_paul15_cluster_list):
    """An unknown character gene key raises (R: stop('Not all gene IDs ...'))."""
    fits, genes = fitted_paul15_cluster_list
    with pytest.raises(ValueError, match="Not all gene IDs"):
        cluster_expression_patterns(
            fits, n_points=8, genes=[genes[0], "NoSuchGene"], n_clusters=3
        )


# ---------------------------------------------------------------------------
# Gap #22 -- conditions branch (clusterExpressionPatterns.R:67-135)
# ---------------------------------------------------------------------------


def test_conditions_branch_runs(fitted_paul15_cluster_conditions):
    """The conditions branch returns a ClusterResult with proper shape."""
    adata, genes = fitted_paul15_cluster_conditions
    res = cluster_expression_patterns(
        adata, n_points=6, genes=genes, n_clusters=3
    )
    assert isinstance(res, ClusterResult)
    # Expected width: n_lineages (2) * n_conditions (2) * n_points (6) = 24
    n_lineages = 2
    n_conditions = 2
    n_points = 6
    assert res.yhat_scaled.shape == (len(genes), n_lineages * n_conditions * n_points)


def test_conditions_branch_column_labels(fitted_paul15_cluster_conditions):
    """Conditions column labels follow R's ``l{ii}_{kk}:t{t}`` pattern.

    R source: clusterExpressionPatterns.R:118
    ``colnames(y) <- paste0("l", ii, "_", kk, ":t", df[, paste0("t", ii)])``.
    """
    adata, genes = fitted_paul15_cluster_conditions
    res = cluster_expression_patterns(
        adata, n_points=6, genes=genes, n_clusters=3
    )
    cols = list(res.yhat_scaled.columns)
    # Each label is "l{ii}_{kk}:t{t}"
    starts = {c.split(":", 1)[0] for c in cols}
    assert starts == {"l1_1", "l1_2", "l2_1", "l2_2"}


def test_conditions_branch_long_df_aligned(fitted_paul15_cluster_conditions):
    """``long_df.time_point`` matches ``yhat_scaled.columns`` for conditions branch."""
    adata, genes = fitted_paul15_cluster_conditions
    res = cluster_expression_patterns(
        adata, n_points=6, genes=genes, n_clusters=3
    )
    unique_tp = list(res.long_df["time_point"].drop_duplicates())
    assert unique_tp == list(res.yhat_scaled.columns)


# ---------------------------------------------------------------------------
# Type dispatch
# ---------------------------------------------------------------------------


def test_invalid_models_type_raises():
    """Non-AnnData / non-dict input raises ``TypeError``."""
    with pytest.raises(TypeError, match="unsupported models type"):
        cluster_expression_patterns(
            models=123, n_points=5, genes=["a", "b"]
        )
