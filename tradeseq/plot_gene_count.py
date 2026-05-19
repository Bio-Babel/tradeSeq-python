"""plot_gene_count (R source: tradeSeq/R/plotGeneCount.R).

Renders the reduced-dimension cell embedding coloured by gene expression or
categorical clusters, overlaid with the principal-curve geometry and (when
``fit_gam`` results are available) the knot points along each lineage.

The R source dispatches on several trajectory-container signatures; the Python
public container is always ``AnnData``, so this single entrypoint serves all
paths. The
principal-curve geometry is read from
``adata.uns["slingshot"]["curves"][lineage_id]["s"][ord]``; the per-knot 2D
coordinates are resolved by nearest-``lambda`` lookup against
``adata.uns["slingshot"]["curves"][lineage_id]["lambda"]`` — see the
``tradeseq_porting_essential_suggestions.md`` §3 NOTE for why
``princurve::project_to_curve`` degenerates to ``argmin(|lambda - knot|)`` in
this code path.
"""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import ggplot2_py as gg
import numpy as np
import pandas as pd

__all__ = ["plot_gene_count"]


def _knot_coordinates(curves: dict, knots: np.ndarray) -> np.ndarray:
    """Resolve knot 2D coordinates by nearest-``lambda`` lookup per lineage.

    Mirrors ``plotGeneCount.R:45-56``: for every (lineage, knot) pair, locate
    the on-curve sample whose pseudotime ``lambda`` is closest to the knot
    location, then return its 2D position from the curve's ``s`` array sorted
    by ``ord``. The R source calls
    ``princurve::project_to_curve(x = slingCurves[[ii]]$s, s = ordered_s)`` and
    pulls ``S$lambda`` and ``S$s`` from the result, but since ``x`` is the
    curve itself the projection collapses to identity (each on-curve point's
    closest curve point is itself), so we can read ``lambda`` and ``s`` from
    the stored fixture directly.

    Parameters
    ----------
    curves : dict
        ``adata.uns["slingshot"]["curves"]`` — mapping from string lineage id to
        a dict with keys ``s`` ``(n_cells, 2)``, ``ord`` ``(n_cells,)``,
        ``lambda`` ``(n_cells,)``.
    knots : numpy.ndarray
        Knot vector from ``adata.uns[key]["knots"]``.

    Returns
    -------
    numpy.ndarray
        ``(n_lineages * len(knots), 2)`` coordinate matrix in row-major
        (lineage_outer, knot_inner) order — same ordering as R's
        ``knots_dim[jj + (ii-1)*length(knots), ]``.
    """
    knots = np.asarray(knots, dtype=np.float64)
    n_knots = knots.size
    lineage_ids = _lineage_ids(curves)
    n_lineages = len(lineage_ids)
    out = np.empty((n_lineages * n_knots, 2), dtype=np.float64)
    for ii, lid in enumerate(lineage_ids):
        curve = curves[lid]
        s_arr = np.asarray(curve["s"], dtype=np.float64)
        # R uses 1-based ord; subtract 1 for Python.
        ord_arr = np.asarray(curve["ord"]).astype(np.int64)
        if ord_arr.min() == 1:
            ord_arr = ord_arr - 1
        lambda_arr = np.asarray(curve["lambda"], dtype=np.float64)
        # Sort by ord so points walk the curve.
        s_ord = s_arr[ord_arr]
        lambda_ord = lambda_arr[ord_arr]
        for jj, kn in enumerate(knots):
            # argmin |lambda - kn| over the on-curve points (post-ord).
            idx = int(np.argmin(np.abs(lambda_ord - kn)))
            out[jj + ii * n_knots, :] = s_ord[idx, :2]
    return out


def _curve_2d(curve: dict) -> np.ndarray:
    """Return the principal-curve geometry as a ``(n_cells, 2)`` array sorted by ``ord``."""
    s_arr = np.asarray(curve["s"], dtype=np.float64)
    ord_arr = np.asarray(curve["ord"]).astype(np.int64)
    if ord_arr.min() == 1:
        ord_arr = ord_arr - 1
    return s_arr[ord_arr, :2]


def _lineage_ids(curves: dict) -> list:
    """Return lineage ids in R ``seq_along(slingCurves(...))`` order."""

    def key_fn(x):
        s = str(x)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(curves.keys(), key=key_fn)


def plot_gene_count(
    adata: ad.AnnData,
    *,
    gene: Optional[Union[str, int]] = None,
    clusters_key: Optional[str] = None,
    layer: str = "counts",
    key: str = "tradeseq",
    title: Optional[str] = None,
    embedding_key: str = "X_umap",
    slingshot_key: str = "slingshot",
) -> gg.GGPlot:
    """Render the per-cell embedding overlaid with the principal curves and knots.

    Port of ``tradeSeq::plotGeneCount`` (R source: ``tradeSeq/R/plotGeneCount.R``).
    A scatter of cells on ``adata.obsm[embedding_key]`` is coloured either by
    the log1p expression of ``gene`` (continuous yellow→red gradient, R
    ``scale_color_gradient(low="yellow", high="red")``) or by the categorical
    cluster label at ``adata.obs[clusters_key]`` (default discrete palette).
    Each lineage's principal curve is drawn as a black ``geom_path``; when the
    AnnData has been processed by :func:`tradeseq.fit_gam`, the per-lineage
    knots are also drawn as black ``geom_point`` markers.

    Parameters
    ----------
    adata : anndata.AnnData
        Container with the embedding at ``adata.obsm[embedding_key]`` and the
        per-lineage principal-curve geometry at
        ``adata.uns[slingshot_key]["curves"]``.
    gene : str or int, optional
        Gene name or 0-based row index. If both ``gene`` and ``clusters_key`` are
        supplied, the gene-expression branch wins, matching R's
        ``if (!is.null(gene)) ... else ...``.
    clusters_key : str, optional
        Name of an ``adata.obs`` column carrying the per-cell cluster label.
    layer : str, default "counts"
        Layer holding the raw counts (used only when ``gene`` is supplied).
    key : str, default "tradeseq"
        Namespace prefix for :func:`tradeseq.fit_gam` slots. When
        ``adata.uns[key]["knots"]`` is present the knots are overlaid;
        otherwise the knot layer is skipped (mirrors the R ``is.null(models)``
        guard at ``plotGeneCount.R:32``).
    title : str, optional
        Override the colour-bar / legend title. Defaults to
        ``"logged count of gene {gene}"`` or ``"Clusters"`` matching R.
    embedding_key : str, default "X_umap"
        ``adata.obsm`` key for the 2-D embedding.
    slingshot_key : str, default "slingshot"
        ``adata.uns`` key holding the principal-curve dict.

    Returns
    -------
    ggplot2_py.GGPlot
        The composed ggplot2 figure.
    """
    if gene is None and clusters_key is None:
        raise ValueError(
            "Either gene and counts, or clusters argument must be supplied"
        )

    if embedding_key not in adata.obsm:
        raise KeyError(f"adata.obsm has no key {embedding_key!r}")
    rd = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    if rd.ndim != 2 or rd.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{embedding_key!r}] must be 2-D with >=2 columns; got "
            f"shape {rd.shape}"
        )

    if slingshot_key not in adata.uns:
        raise KeyError(f"adata.uns has no key {slingshot_key!r}")
    sling = adata.uns[slingshot_key]
    if "curves" not in sling:
        raise KeyError(f"adata.uns[{slingshot_key!r}] has no 'curves' entry")
    curves = sling["curves"]

    # Build the per-cell scatter dataframe.
    df = pd.DataFrame({"dim1": rd[:, 0], "dim2": rd[:, 1]})
    if gene is not None:
        if layer not in adata.layers:
            raise KeyError(f"adata.layers has no key {layer!r}")
        if isinstance(gene, str):
            if gene not in adata.var_names:
                raise ValueError(f"gene {gene!r} not in adata.var_names")
            g_idx = int(adata.var_names.get_loc(gene))
            g_name = gene
        else:
            g_idx = int(gene)
            if g_idx < 0 or g_idx >= adata.n_vars:
                raise ValueError(f"gene index {g_idx} out of bounds")
            g_name = str(adata.var_names[g_idx])
        counts_layer = adata.layers[layer]
        if hasattr(counts_layer, "toarray"):
            col = counts_layer[:, g_idx].toarray().ravel()
        else:
            col = np.asarray(counts_layer[:, g_idx]).ravel()
        df["col"] = np.log1p(col.astype(np.float64))
        legend_title = title if title is not None else f"logged count of gene {g_name}"
        use_gradient = True
    else:
        if clusters_key not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {clusters_key!r}")
        df["col"] = adata.obs[clusters_key].astype(str).values
        legend_title = title if title is not None else "Clusters"
        use_gradient = False

    p = (
        gg.ggplot(df, gg.aes(x="dim1", y="dim2", colour="col"))
        + gg.geom_point(size=1)
        + gg.theme_classic()
        + gg.labs(colour=legend_title)
    )
    if use_gradient:
        p = p + gg.scale_color_gradient(low="yellow", high="red")

    # Add the per-lineage principal curves.
    for lid in _lineage_ids(curves):
        curve_xy = _curve_2d(curves[lid])
        curve_df = pd.DataFrame(curve_xy, columns=["dim1", "dim2"])
        p = p + gg.geom_path(
            data=curve_df,
            mapping=gg.aes(x="dim1", y="dim2"),
            inherit_aes=False,
            colour="black",
            linewidth=1,
        )

    # Add the knots, if fit_gam has been run.
    if key in adata.uns and "knots" in adata.uns[key]:
        knots = np.asarray(adata.uns[key]["knots"], dtype=np.float64)
        knots_xy = _knot_coordinates(curves, knots)
        knots_df = pd.DataFrame(knots_xy, columns=["dim1", "dim2"])
        p = p + gg.geom_point(
            data=knots_df,
            mapping=gg.aes(x="dim1", y="dim2"),
            inherit_aes=False,
            colour="black",
            size=2,
        )

    return p
