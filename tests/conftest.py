"""Pytest session bootstrap.

VEPAGE-983: pytest has no Intuit network session, so the ispy-pii REST
endpoint is unreachable. Force masking off for the test session — synthetic
test fixtures don't carry real PII, and unreachable masking would otherwise
replace log/span content with the redaction placeholder and break tests that
assert on log text or span attribute values. Production code defaults this
to "true".
"""

import os

os.environ.setdefault("MIRIX_ISPY_PII_ENABLED", "false")
