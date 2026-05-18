"""AnnData slot I/O helpers (storage schema: port_reports/tradeSeq/05_design.md)."""

from __future__ import annotations

from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

__all__ = [
    "write_fit_results",
    "read_beta",
    "read_sigma",
    "read_design_matrix",
    "read_lpmatrix",
    "read_knots",
    "read_converged",
    "read_conditions",
    "read_family",
    "read_slingshot_coldata",
]


def write_fit_results(
    adata: ad.AnnData,
    *,
    beta: np.ndarray,
    sigma_list: Sequence[np.ndarray],
    converged: np.ndarray,
    design_matrix: pd.DataFrame,
    lpmatrix: np.ndarray,
    knots: np.ndarray,
    family: str,
    conditions: Optional[pd.Categorical] = None,
    slingshot_coldata: Optional[pd.DataFrame] = None,
    key: str = "tradeseq",
) -> None:
    """Write the full fitGAM payload into the AnnData container.

    Parameters
    ----------
    adata : anndata.AnnData
        Container to mutate in place.
    beta : numpy.ndarray
        Per-gene coefficient matrix, shape ``(n_genes, n_coefs)``, dtype float64.
        Stored at ``adata.varm[f"{key}_beta"]``.
    sigma_list : sequence of numpy.ndarray
        Per-gene covariance matrices, each of shape ``(n_coefs, n_coefs)``. The
        sequence is length ``n_genes``. Stored as a dtype-``object`` array at
        ``adata.varm[f"{key}_Sigma"]``.
    converged : numpy.ndarray
        Boolean array of length ``n_genes`` indicating per-gene convergence.
        Stored at ``adata.var[f"{key}_converged"]``.
    design_matrix : pandas.DataFrame
        Per-cell design matrix. Stored at ``adata.uns[key]["design_matrix"]``.
    lpmatrix : numpy.ndarray
        Linear-predictor model matrix, shape ``(n_cells, n_coefs)``. Stored at
        ``adata.uns[key]["lpmatrix"]``.
    knots : numpy.ndarray
        Knot vector, shape ``(n_knots,)``. Stored at ``adata.uns[key]["knots"]``.
    family : str
        Family name, ``"nb"`` or ``"gaussian"``. Stored at
        ``adata.uns[key]["family"]``.
    conditions : pandas.Categorical, optional
        Per-cell condition assignments, length ``n_cells``. Stored at
        ``adata.uns[key]["conditions"]`` when provided; otherwise the slot is
        explicitly set to ``None`` so the key always exists.
    slingshot_coldata : pandas.DataFrame, optional
        The slingshot-side colData snapshot. Stored at
        ``adata.uns[key]["slingshot_coldata"]`` when provided.
    key : str, default ``"tradeseq"``
        Namespace prefix. ``adata.uns[key]``, ``adata.varm[f"{key}_beta"]``,
        ``adata.varm[f"{key}_Sigma"]``, ``adata.var[f"{key}_converged"]``.
    """
    n_genes = adata.n_vars
    n_cells = adata.n_obs

    beta_arr = np.asarray(beta, dtype=np.float64)
    if beta_arr.shape[0] != n_genes:
        raise ValueError(
            f"beta has {beta_arr.shape[0]} rows but adata has {n_genes} genes"
        )
    adata.varm[f"{key}_beta"] = beta_arr

    sigma_arr = np.empty(n_genes, dtype=object)
    if len(sigma_list) != n_genes:
        raise ValueError(
            f"sigma_list has length {len(sigma_list)} but adata has {n_genes} genes"
        )
    for g, sig in enumerate(sigma_list):
        sigma_arr[g] = np.asarray(sig, dtype=np.float64)
    adata.varm[f"{key}_Sigma"] = sigma_arr

    converged_arr = np.asarray(converged, dtype=bool)
    if converged_arr.shape[0] != n_genes:
        raise ValueError(
            f"converged has length {converged_arr.shape[0]} but adata has "
            f"{n_genes} genes"
        )
    adata.var[f"{key}_converged"] = converged_arr

    lp_arr = np.asarray(lpmatrix, dtype=np.float64)
    if lp_arr.shape[0] != n_cells:
        raise ValueError(
            f"lpmatrix has {lp_arr.shape[0]} rows but adata has {n_cells} cells"
        )

    knots_arr = np.asarray(knots, dtype=np.float64)

    if design_matrix.shape[0] != n_cells:
        raise ValueError(
            f"design_matrix has {design_matrix.shape[0]} rows but adata has "
            f"{n_cells} cells"
        )

    uns_payload: dict = {
        "design_matrix": design_matrix,
        "lpmatrix": lp_arr,
        "knots": knots_arr,
        "family": str(family),
        "conditions": conditions,
        "slingshot_coldata": slingshot_coldata,
    }
    adata.uns[key] = uns_payload


def _require_uns(adata: ad.AnnData, key: str) -> dict:
    """Return ``adata.uns[key]`` or raise ``KeyError`` with a helpful message."""
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns has no key {key!r}; call write_fit_results(..., key={key!r}) first"
        )
    return adata.uns[key]


def read_beta(adata: ad.AnnData, *, key: str = "tradeseq") -> np.ndarray:
    """Return the per-gene coefficient matrix, shape ``(n_genes, n_coefs)``."""
    slot = f"{key}_beta"
    if slot not in adata.varm:
        raise KeyError(f"adata.varm has no key {slot!r}")
    return np.asarray(adata.varm[slot])


def read_sigma(adata: ad.AnnData, *, key: str = "tradeseq") -> np.ndarray:
    """Return the per-gene covariance object array, length ``n_genes``.

    AnnData stores ``varm`` slots as 2D ``(n_genes, k)``. For an object array of
    ``(n_coefs, n_coefs)`` matrices, ``k == 1``; we flatten to ``(n_genes,)`` to
    match the schema.
    """
    slot = f"{key}_Sigma"
    if slot not in adata.varm:
        raise KeyError(f"adata.varm has no key {slot!r}")
    arr = np.asarray(adata.varm[slot])
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def read_design_matrix(adata: ad.AnnData, *, key: str = "tradeseq") -> pd.DataFrame:
    """Return the per-cell design matrix DataFrame from ``adata.uns[key]``."""
    return _require_uns(adata, key)["design_matrix"]


def read_lpmatrix(adata: ad.AnnData, *, key: str = "tradeseq") -> np.ndarray:
    """Return the linear-predictor model matrix, shape ``(n_cells, n_coefs)``."""
    return np.asarray(_require_uns(adata, key)["lpmatrix"])


def read_knots(adata: ad.AnnData, *, key: str = "tradeseq") -> np.ndarray:
    """Return the knot vector, shape ``(n_knots,)``."""
    return np.asarray(_require_uns(adata, key)["knots"])


def read_converged(adata: ad.AnnData, *, key: str = "tradeseq") -> np.ndarray:
    """Return the per-gene convergence flag vector, length ``n_genes``."""
    slot = f"{key}_converged"
    if slot not in adata.var:
        raise KeyError(f"adata.var has no key {slot!r}")
    return np.asarray(adata.var[slot], dtype=bool)


def read_conditions(
    adata: ad.AnnData, *, key: str = "tradeseq"
) -> Optional[pd.Categorical]:
    """Return the conditions categorical from ``adata.uns[key]``, or ``None``."""
    return _require_uns(adata, key)["conditions"]


def read_family(adata: ad.AnnData, *, key: str = "tradeseq") -> str:
    """Return the family name (``"nb"`` or ``"gaussian"``)."""
    return str(_require_uns(adata, key)["family"])


def read_slingshot_coldata(
    adata: ad.AnnData, *, key: str = "tradeseq"
) -> Optional[pd.DataFrame]:
    """Return the slingshot colData snapshot from ``adata.uns[key]``, or ``None``."""
    return _require_uns(adata, key)["slingshot_coldata"]
