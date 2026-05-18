"""Tests for tradeseq.cascade (R source: cascade.R).

The Python ``cascade`` is a deferral stub per
``tradeseq_porting_essential_suggestions.md`` §8: the upstream R algorithm
has a latent bug at ``cascade.R:122`` and the port is deferred to v2. The
stub MUST raise ``NotImplementedError`` with a specific message regardless of
arguments and must carry a NumPy-style docstring.
"""

from __future__ import annotations

import inspect

import pytest

from tradeseq.cascade import cascade


def test_cascade_raises_not_implemented(paul15_small_adata):
    """A bare call raises ``NotImplementedError`` with the documented message."""
    with pytest.raises(NotImplementedError) as excinfo:
        cascade(paul15_small_adata)
    msg = str(excinfo.value)
    assert "cascade is deferred to v2" in msg
    assert "port_reports/tradeSeq/08_validation.md" in msg


def test_cascade_raises_with_kwargs(paul15_small_adata):
    """Stub ignores extra positional/keyword args and still raises the same error."""
    with pytest.raises(NotImplementedError, match="deferred to v2"):
        cascade(
            paul15_small_adata,
            lineage=1,
            genes=list(paul15_small_adata.var_names),
            n_points=100,
            epsilon=1e-6,
            derivative_threshold=0.1,
            plot_heatmap=False,
        )


def test_cascade_raises_with_positional_args(paul15_small_adata):
    """Extra positional args are accepted without ``TypeError``."""
    with pytest.raises(NotImplementedError, match="deferred to v2"):
        cascade(paul15_small_adata, 1, ["Acin1"], 100)


def test_cascade_has_docstring():
    """Stub carries a NumPy-style docstring (referenced by validation)."""
    doc = cascade.__doc__
    assert doc is not None
    assert "cascade" in doc.lower()
    # NumPy-style section markers we care about for downstream sphinx parsing.
    assert "Parameters" in doc
    assert "Raises" in doc


def test_cascade_signature_accepts_var_args():
    """Public signature accepts ``*args, **kwargs`` so v2 can add params without breakage."""
    sig = inspect.signature(cascade)
    kinds = [p.kind for p in sig.parameters.values()]
    assert inspect.Parameter.VAR_POSITIONAL in kinds
    assert inspect.Parameter.VAR_KEYWORD in kinds


def test_cascade_returns_none_annotation():
    """Return annotation is ``None`` — the stub never returns a value.

    ``from __future__ import annotations`` turns the annotation into the
    string ``"None"``, so we accept either the literal value or its
    stringified form.
    """
    sig = inspect.signature(cascade)
    ann = sig.return_annotation
    assert ann is None or ann is type(None) or ann == "None"


def test_cascade_exact_error_message(paul15_small_adata):
    """Error message text is exactly what the port plan prescribes."""
    expected = (
        "cascade is deferred to v2; "
        "see port_reports/tradeSeq/08_validation.md for context"
    )
    with pytest.raises(NotImplementedError) as excinfo:
        cascade(paul15_small_adata)
    assert str(excinfo.value) == expected
