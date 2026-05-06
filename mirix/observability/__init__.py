"""Observability and tracing utilities for Mirix.

This package surfaces an ispy-pii-backed masker (:mod:`pii_mask`) for use
as the ``mask=`` callback on the Langfuse client. The mask scrubs PII
tokens out of conversation/query/completion content as Langfuse exports
spans, so live-eval consumers downstream see redacted but
structurally-intact traces.

The Langfuse client itself lives in ``mirix.observability.langfuse_client``
in the published ``jl-ecms-server`` wheel; the source of that file is
expected to land in this repo via a separate change.
"""

from mirix.observability.pii_mask import build_langfuse_mask, ispy_pii_mask

__all__ = ["build_langfuse_mask", "ispy_pii_mask"]
