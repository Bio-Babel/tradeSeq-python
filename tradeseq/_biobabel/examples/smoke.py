"""Smoke test for tradeseq._biobabel.

Exercises the minimal contract path: load the bundled fixture, fit the NB-GAM on
a small gene subset, and run one within-lineage Wald test. Intended for producer
maintainers / CI — it imports and *runs* the package, so the scientific deps must
be installed. biobabel itself never executes this.

    python tradeseq/_biobabel/examples/smoke.py
"""

from __future__ import annotations


def main() -> None:
    import tradeseq as ts

    # 1. Bundled Paul15 fixture: raw counts + a frozen 2-lineage trajectory.
    adata = ts.load_paul15()
    assert "counts" in adata.layers
    assert "pseudotime" in adata.obsm and "cell_weights" in adata.obsm

    # 2. Fit on a small subset so the smoke stays fast.
    sub = adata[:, adata.var_names[:8]].copy()
    ts.fit_gam(sub, n_knots=6, verbose=False)

    # fit_gam must have written its namespace.
    assert "tradeseq" in sub.uns
    assert "tradeseq_beta" in sub.varm and "tradeseq_Sigma" in sub.varm
    assert "tradeseq_converged" in sub.var
    assert ts.nknots(sub) == 6

    # 3. One Wald test reads those slots and returns a per-gene table.
    res = ts.association_test(sub)
    assert {"waldStat", "df", "pvalue", "meanLogFC"}.issubset(res.columns)
    assert len(res) == sub.n_vars

    print(f"tradeseq smoke OK: fitted {sub.n_vars} genes, association_test -> {res.shape}")


if __name__ == "__main__":
    main()
