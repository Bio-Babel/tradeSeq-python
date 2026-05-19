"""plot_smoothers (R source: tradeSeq/R/plotSmoothers.R).

Port of the AnnData branch corresponding to ``plotSmoothers``. Renders the per-cell
expression scatter overlaid with the fitted per-lineage smoothers produced by
:func:`tradeseq.fit_gam`.

The R source dispatches on fitted model, fitted-container, and conditions
branches. In Python the container is always ``AnnData`` so the single
:func:`plot_smoothers` entrypoint covers both the fitted-trajectory and
conditions paths — they are selected automatically by reading
``adata.uns[key]["conditions"]``.
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable, Optional, Sequence, Union

import anndata as ad
import ggplot2_py as gg
import numpy as np
import pandas as pd

from ._predict import predict_gam

__all__ = ["plot_smoothers"]


def _get_predict_range_df(
    dm: pd.DataFrame,
    lineage_id: int,
    *,
    condition_id: Optional[int] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """Build the prediction grid for a single lineage (R: ``.getPredictRangeDf``).

    Mirrors ``tradeSeq/R/utils.R:108-154``. Sets all ``tK`` and ``lK`` columns to
    zero, then fills the lineage-of-interest column(s) with ``1 / n_matching``
    and the ``tK`` column with an evenly spaced grid covering the empirical
    pseudotime range of cells assigned to that lineage.

    Parameters
    ----------
    dm : pd.DataFrame
        Per-cell design matrix from ``adata.uns[key]["design_matrix"]``.
    lineage_id : int
        1-based lineage index.
    condition_id : int or None
        If not ``None``, restrict the matching ``lK_C`` column. 1-based.
    n_points : int
        Length of the output grid.

    Returns
    -------
    pd.DataFrame
        One-row-per-grid-point design frame with the same columns as ``dm``
        (sans the ``y`` column if present; R also strips it). The ``offset``
        column is renamed back from ``offset(offset)`` to the inner name.
    """
    vars_df = dm.iloc[[0]].copy()
    cols = list(vars_df.columns)
    if "y" in cols:
        vars_df = vars_df.drop(columns=["y"])
        cols = list(vars_df.columns)

    # Find the offset column and rename it from `offset(offset)` -> `offset`.
    offset_pat = re.compile(r"offset")
    offset_idxs = [i for i, c in enumerate(cols) if offset_pat.search(c)]
    if not offset_idxs:
        raise KeyError("design matrix has no offset column")
    offset_idx = offset_idxs[0]
    offset_col_name = cols[offset_idx]
    # R: substr(offsetName, start = 8, stop = nchar(offsetName) - 1) → strips "offset(" ... ")"
    if offset_col_name.startswith("offset(") and offset_col_name.endswith(")"):
        inner_offset_name = offset_col_name[len("offset(") : -1]
    else:
        inner_offset_name = offset_col_name
    vars_df = vars_df.rename(columns={offset_col_name: inner_offset_name})

    # Set all tK / lK columns to zero
    t_re = re.compile(r"^t[0-9]+$")
    l_re = re.compile(r"^l[0-9]+(_[0-9]+)?$")
    t_cols = [c for c in vars_df.columns if t_re.match(c)]
    l_cols = [c for c in vars_df.columns if l_re.match(c)]
    for c in t_cols:
        vars_df[c] = 0.0
    for c in l_cols:
        vars_df[c] = 0.0

    # Replicate to n_points rows.
    vars_df = pd.concat([vars_df] * n_points, ignore_index=True)

    # Identify the lineage columns to fill.
    if condition_id is None:
        # match pattern paste0("l", lineageId, "($|_)") — R regex meaning lI or lI_*
        match_re = re.compile(rf"^l{lineage_id}($|_)")
    else:
        match_re = re.compile(rf"^l{lineage_id}_{condition_id}$")
    lineage_ids_cols = [c for c in vars_df.columns if match_re.match(c)]
    if not lineage_ids_cols:
        raise KeyError(
            f"design matrix has no lineage column for lineage_id={lineage_id} "
            f"condition_id={condition_id}"
        )

    # Compute training range for the lineage from rows where the lineage column
    # set sums to 1 in the original dm.
    if len(lineage_ids_cols) == 1:
        mask = dm[lineage_ids_cols[0]].values == 1
    else:
        mask = dm[lineage_ids_cols].sum(axis=1).values == 1
    t_col_name = f"t{lineage_id}"
    if t_col_name not in dm.columns:
        raise KeyError(f"design matrix has no column {t_col_name!r}")
    lineage_data = dm.loc[mask, t_col_name].values.astype(float)
    if lineage_data.size == 0:
        raise ValueError(
            f"No cells are assigned to lineage_id={lineage_id} in the design matrix"
        )
    # R: if(min(lineageData)/max(lineageData) < .01) lineageData[which.min(lineageData)] <- 0
    lin_min = float(lineage_data.min())
    lin_max = float(lineage_data.max())
    if lin_max != 0 and (lin_min / lin_max) < 0.01:
        idx = int(np.argmin(lineage_data))
        lineage_data[idx] = 0.0
        lin_min = float(lineage_data.min())

    # Fill: 1/n in lineage cols, pseudotime grid in tK col.
    for c in lineage_ids_cols:
        vars_df[c] = 1.0 / len(lineage_ids_cols)
    vars_df[t_col_name] = np.linspace(lin_min, lin_max, n_points)

    # Offset column = mean(dm.offset)
    mean_offset = float(dm[offset_col_name].mean())
    vars_df[inner_offset_name] = mean_offset
    return vars_df


def _resolve_gene_index(adata: ad.AnnData, gene: Union[str, int]) -> tuple[str, int]:
    """Resolve ``gene`` (string name or 0-based index) to a (name, index) pair."""
    if isinstance(gene, (str, np.str_)):
        if gene not in adata.var_names:
            raise ValueError(f"The gene ID is not present in the models object: {gene!r}")
        return str(gene), int(adata.var_names.get_loc(str(gene)))
    if isinstance(gene, (int, np.integer)):
        idx = int(gene)
        if idx < 0 or idx >= adata.n_vars:
            raise ValueError(f"gene index {idx} out of bounds for n_vars={adata.n_vars}")
        return str(adata.var_names[idx]), idx
    raise TypeError(f"gene must be str or int, got {type(gene).__name__}")


def plot_smoothers(
    adata: ad.AnnData,
    *,
    gene: Union[str, int, Sequence[Union[str, int]]],
    n_points: int = 100,
    lwd: float = 2.0,
    size: float = 2.0 / 3.0,
    xlab: str = "Pseudotime",
    ylab: str = "Log(expression + 1)",
    border: bool = True,
    alpha: float = 1.0,
    sample: float = 1.0,
    point_col: Optional[Union[str, Sequence]] = None,
    curves_cols: Optional[Sequence[str]] = None,
    plot_lineages: bool = True,
    lineages_to_plot: Optional[Iterable[int]] = None,
    layer: str = "counts",
    pseudotime_key: str = "pseudotime",
    key: str = "tradeseq",
    random_state: Optional[int] = None,
) -> gg.GGPlot:
    """Plot the smoothers estimated by :func:`tradeseq.fit_gam`.

    Port of ``tradeSeq::plotSmoothers`` (R source: ``tradeSeq/R/plotSmoothers.R``)
    — specifically the R method that branches on the presence/absence of
    ``conditions``. Each lineage's predicted smoother
    is rendered as a ``geom_line`` over a per-cell ``geom_point`` scatter of
    ``log1p(count)`` versus pseudotime. With ``border=True`` (R default) every
    smoother gets a thicker white halo drawn underneath. The R-side defaults
    ``lwd=2``, ``size=2/3``, ``alpha=1`` (R: ``plotSmoothers.R:432``),
    ``sample=1`` are preserved.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`tradeseq.fit_gam`. Must carry
        the design matrix at ``adata.uns[key]["design_matrix"]``, the linear
        predictor at ``adata.uns[key]["lpmatrix"]`` /
        ``adata.uns[key]["lpmatrix_columns"]``, the knots at
        ``adata.uns[key]["knots"]``, the optional conditions categorical at
        ``adata.uns[key]["conditions"]``, the per-gene coefficient matrix at
        ``adata.varm[f"{key}_beta"]``, and a count layer at
        ``adata.layers[layer]``.
    gene : str or int or sequence
        Gene name (string) or 0-based row index, or a list of either to plot
        multiple genes side-by-side via ``facet_wrap``. The R source asserts
        ``length(gene) == 1`` but Python supports the natural extension.
    n_points : int, default 100
        Number of grid points per lineage for the predicted smoother.
    lwd : float, default 2.0
        Smoother line width (passed to ``geom_line(linewidth=)``).
    size : float, default 2/3
        Point size for the scatter layer.
    xlab, ylab : str
        Axis labels. R defaults: ``"Pseudotime"`` and ``"Log(expression + 1)"``.
    border : bool, default True
        Add a white halo behind each smoother line.
    alpha : float, default 1.0
        Point transparency (passed to ``scale_color_viridis_d(alpha=)``). The
        R user-facing dispatcher (plotSmoothers.R:432) sets ``alpha = 1``,
        overriding the inner helper's ``alpha = 2/3`` default.
    sample : float, default 1.0
        Fraction of cells to retain in the scatter layer. R uses
        ``sample(seq_len(nrow(df)), nrow(df) * sample, replace = FALSE)``.
    point_col : str or sequence or None
        If a string, name of an ``adata.obs`` column to colour points by.
        If a sequence, per-cell colour labels of length ``adata.n_obs``.
        ``None`` reverts to the default lineage colouring.
    curves_cols : sequence of str or None
        Per-lineage smoother colours. Must have length equal to the number of
        lineages (or lineages × conditions). Falls back to the viridis palette
        when ``None`` or when the length is wrong (matching R).
    plot_lineages : bool, default True
        Whether to draw the smoother lines on top of the scatter.
    lineages_to_plot : iterable of int or None
        1-based lineage indices to plot. ``None`` plots all lineages.
    layer : str, default "counts"
        Layer holding the raw counts.
    pseudotime_key : str, default "pseudotime"
        ``adata.obsm`` key for the per-cell, per-lineage pseudotime matrix.
    key : str, default "tradeseq"
        Namespace prefix for the ``fit_gam`` slots.
    random_state : int or None
        Seed for the ``sample`` subsampling.

    Returns
    -------
    ggplot2_py.GGPlot
        The composed ggplot2 figure.
    """
    # Validate AnnData payload.
    if key not in adata.uns:
        raise KeyError(f"adata.uns has no key {key!r}; run fit_gam first")
    uns = adata.uns[key]
    if "design_matrix" not in uns:
        raise KeyError(f"adata.uns[{key!r}] has no key 'design_matrix'")
    dm: pd.DataFrame = uns["design_matrix"]
    lp_arr = np.asarray(uns["lpmatrix"], dtype=np.float64)
    lp_cols = list(uns["lpmatrix_columns"])
    lpmatrix = pd.DataFrame(lp_arr, columns=lp_cols)
    beta = np.asarray(adata.varm[f"{key}_beta"], dtype=np.float64)
    conditions = uns.get("conditions")
    cond_present = conditions is not None

    if pseudotime_key not in adata.obsm:
        raise KeyError(f"adata.obsm has no key {pseudotime_key!r}")
    pseudotime = np.asarray(adata.obsm[pseudotime_key], dtype=np.float64)
    if pseudotime.ndim == 1:
        pseudotime = pseudotime[:, None]

    if layer not in adata.layers:
        raise KeyError(f"adata.layers has no key {layer!r}")
    counts_layer = adata.layers[layer]
    if hasattr(counts_layer, "toarray"):
        counts_layer = counts_layer.toarray()
    counts = np.asarray(counts_layer, dtype=np.float64)  # cells × genes

    # Multi-gene faceting support.
    if isinstance(gene, (str, int, np.integer, np.str_)):
        gene_list: list[Union[str, int]] = [gene]
        facet = False
    else:
        gene_list = list(gene)
        if len(gene_list) == 0:
            raise ValueError("`gene` is empty")
        facet = len(gene_list) > 1

    # Number of lineages and conditions.
    t_re = re.compile(r"^t[0-9]+$")
    n_curves = sum(1 for c in dm.columns if t_re.match(c))
    if cond_present:
        cond_cat = pd.Categorical(conditions)
        cond_levels = list(cond_cat.categories)
        n_conditions = len(cond_levels)
    else:
        cond_cat = None
        cond_levels = None
        n_conditions = 1

    if lineages_to_plot is None:
        lineages_active = list(range(1, n_curves + 1))
    else:
        lineages_active = [int(x) for x in lineages_to_plot]

    # --- Construct per-cell scatter df (shared across genes, then merged via
    # cross product if facet=True).
    n_cells = adata.n_obs
    rng = np.random.default_rng(random_state)

    # Build the per-cell time and lineage-label vectors.
    time_all = np.zeros(n_cells, dtype=np.float64)
    if not cond_present:
        lcol = np.zeros(n_cells, dtype=np.int64)
        for jj in range(1, n_curves + 1):
            l_col_name = f"l{jj}"
            if l_col_name not in dm.columns:
                continue
            assigned = dm[l_col_name].values == 1
            time_all[assigned] = dm.loc[assigned, f"t{jj}"].values
            lcol[assigned] = jj
        lineage_label = lcol.astype(str)
    else:
        lineage_label_list: list[str] = ["0"] * n_cells
        for jj in range(1, n_curves + 1):
            for kk in range(1, n_conditions + 1):
                l_col_name = f"l{jj}_{kk}"
                if l_col_name not in dm.columns:
                    continue
                assigned = dm[l_col_name].values == 1
                idxs = np.where(assigned)[0]
                t_jj = dm[f"t{jj}"].values
                cond_str = str(cond_levels[kk - 1])
                for i in idxs:
                    time_all[i] = t_jj[i]
                    lineage_label_list[i] = f"Lineage {jj}_{cond_str}"
        lineage_label = np.asarray(lineage_label_list, dtype=object)
        # Reorder the lineage factor as R does.
        combs = []
        for jj in range(1, n_curves + 1):
            for kk in range(1, n_conditions + 1):
                combs.append(f"Lineage {jj}_{cond_levels[kk - 1]}")
        # We keep ordering implicit via a Categorical with the desired levels.

    # Determine the colour column for points.
    point_col_supplied = point_col is not None
    if point_col is None:
        point_col_vec = None
    elif isinstance(point_col, str):
        if point_col not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {point_col!r}")
        point_col_vec = adata.obs[point_col].astype(str).values
    else:
        arr = np.asarray(point_col)
        if arr.ndim == 0 or arr.shape[0] != n_cells:
            msg = (
                "pointCol should have length of either 1 or the number of cells, "
                "reverting to default color scheme"
            )
            if cond_present:
                msg += ", by lineages and conditions."
            else:
                msg += "."
            warnings.warn(msg, UserWarning, stacklevel=2)
            # R keeps the pointCol branch active after this fallback, using the
            # lineage labels as pCol and a discrete "Cell labels" legend.
            point_col_vec = lineage_label.astype(str)
        else:
            point_col_vec = arr.astype(str)

    # --- Subsample cells.
    sample_float = float(sample)
    if sample_float < 0 or sample_float > 1:
        raise ValueError("sample must be between 0 and 1")
    # R's sample() truncates a non-integer size via as.integer(). Preserve that
    # behavior, including sample=0 yielding an empty scatter layer.
    n_sample = int(n_cells * sample_float)
    rows = rng.choice(n_cells, size=n_sample, replace=False)

    # --- Assemble scatter dataframe.
    scatter_rows: list[pd.DataFrame] = []
    for g_obj in gene_list:
        g_name, g_idx = _resolve_gene_index(adata, g_obj)
        y = counts[:, g_idx]
        df = pd.DataFrame(
            {
                "time": time_all,
                "gene_count": y,
                "lineage": lineage_label,
                "gene": g_name,
            }
        )
        if point_col_vec is not None:
            df["pCol"] = point_col_vec
        else:
            df["pCol"] = lineage_label
        df = df.iloc[rows].copy()
        scatter_rows.append(df)

    df_scatter = pd.concat(scatter_rows, axis=0, ignore_index=True)
    df_scatter["log1p_count"] = np.log1p(df_scatter["gene_count"].values)
    # Restrict scatter to lineages_to_plot for the non-conditions branch
    # (matches R block at `plotSmoothers.R:136-138`).
    if not cond_present and lineages_to_plot is not None:
        keep_labels = set(str(x) for x in lineages_active)
        df_scatter = df_scatter[df_scatter["lineage"].astype(str).isin(keep_labels)].copy()

    # --- Build the smoother dataframe (one row per (gene, lineage_active,
    #     condition, n_points)).
    smoother_rows: list[pd.DataFrame] = []
    for g_obj in gene_list:
        g_name, g_idx = _resolve_gene_index(adata, g_obj)
        beta_g = beta[g_idx]
        if np.any(np.isnan(beta_g)):
            if cond_present:
                raise ValueError(
                    f"Some coefficients for gene {g_name!r} are NA. Cannot plot this gene."
                )
        if not plot_lineages:
            continue
        if not cond_present:
            for jj in lineages_active:
                grid_df = _get_predict_range_df(dm, jj, n_points=n_points)
                xdf = predict_gam(
                    lpmatrix=lpmatrix,
                    df=grid_df,
                    pseudotime=pseudotime,
                )
                eta = xdf.values @ beta_g + grid_df["offset"].values
                yhat = np.exp(eta)
                lineage_str = str(jj)
                rec = pd.DataFrame(
                    {
                        "time": grid_df[f"t{jj}"].values,
                        "smoother_yhat": yhat,
                        "lineage": lineage_str,
                        "pCol": lineage_str,
                        "gene": g_name,
                    }
                )
                smoother_rows.append(rec)
        else:
            for jj in lineages_active:
                for kk in range(1, n_conditions + 1):
                    grid_df = _get_predict_range_df(
                        dm, jj, condition_id=kk, n_points=n_points
                    )
                    xdf = predict_gam(
                        lpmatrix=lpmatrix,
                        df=grid_df,
                        pseudotime=pseudotime,
                        conditions=conditions,
                    )
                    eta = xdf.values @ beta_g + grid_df["offset"].values
                    yhat = np.exp(eta)
                    lineage_str = f"{jj}_{kk}"
                    rec = pd.DataFrame(
                        {
                            "time": grid_df[f"t{jj}"].values,
                            "smoother_yhat": yhat,
                            "lineage": lineage_str,
                            "pCol": lineage_str,
                            "gene": g_name,
                        }
                    )
                    smoother_rows.append(rec)

    if smoother_rows:
        df_smoother = pd.concat(smoother_rows, axis=0, ignore_index=True)
        df_smoother["log1p_count"] = np.log1p(df_smoother["smoother_yhat"].values)
    else:
        df_smoother = pd.DataFrame(
            columns=["time", "smoother_yhat", "lineage", "pCol", "gene", "log1p_count"]
        )

    # --- Compose ggplot.
    p = gg.ggplot(df_scatter, gg.aes(x="time", y="log1p_count")) + gg.labs(
        x=xlab, y=ylab
    ) + gg.theme_classic()
    # Scatter layer.
    if not point_col_supplied:
        p = p + gg.geom_point(mapping=gg.aes(colour="lineage"), size=size)
        p = p + gg.scale_color_viridis_d(alpha=alpha)
    else:
        p = p + gg.geom_point(mapping=gg.aes(colour="pCol"), size=size, alpha=alpha)
        p = p + gg.scale_color_discrete()
        p = p + gg.labs(colour="Cell labels")

    # Smoother lines.
    if plot_lineages and not df_smoother.empty:
        # Resolve per-lineage colour palette.
        expected_curves_cols = n_curves * n_conditions if cond_present else n_curves
        invalid_curves_cols = curves_cols is not None and len(curves_cols) != expected_curves_cols
        if curves_cols is None or invalid_curves_cols:
            if invalid_curves_cols:
                warnings.warn(
                    "Incorrect number of lineage colors. Default to viridis",
                    UserWarning,
                    stacklevel=2,
                )
            curves_cols_resolved: list[str] = [
                _rgb_from_viridis(i, n_curves * n_conditions)
                for i in range(n_curves * n_conditions)
            ]
        else:
            curves_cols_resolved = list(curves_cols)

        for _, grp in df_smoother.groupby(["gene", "lineage"], sort=False):
            lineage_str = str(grp["lineage"].iloc[0])
            # R indexing: curvesCols[jj * nConditions - (nConditions - kk)] in the
            # conditional case; otherwise curvesCols[jj].
            if not cond_present:
                jj = int(lineage_str)
                col_idx = jj - 1
            else:
                # lineage is "jj_kk".
                jj_str, kk_str = lineage_str.split("_")
                jj = int(jj_str)
                kk = int(kk_str)
                col_idx = jj * n_conditions - (n_conditions - kk) - 1
            colour = curves_cols_resolved[col_idx]
            if border:
                p = p + gg.geom_line(
                    data=grp,
                    mapping=gg.aes(x="time", y="log1p_count"),
                    inherit_aes=False,
                    linewidth=lwd + 1,
                    colour="white",
                )
            p = p + gg.geom_line(
                data=grp,
                mapping=gg.aes(x="time", y="log1p_count"),
                inherit_aes=False,
                linewidth=lwd,
                colour=colour,
            )

    if facet:
        p = p + gg.facet_wrap("gene")

    return p


def _rgb_from_viridis(i: int, n: int) -> str:
    """Return the i-th of ``n`` viridis colours as a ``#rrggbb`` string.

    Equivalent to ``viridis::viridis(n)[i+1]`` in R. Uses ``scales.pal_viridis``
    so that the palette matches the visualization-stack contract §6 mapping.
    """
    import scales as _scales

    pal = _scales.pal_viridis()
    cols = pal(n)
    return str(cols[i])
