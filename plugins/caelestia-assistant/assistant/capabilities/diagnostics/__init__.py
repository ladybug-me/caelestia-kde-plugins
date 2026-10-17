"""Layer 1: deterministic rule-based diagnostics (no ML, no execution).

Loads rules from rules.d/*.json, matches user text + pasted logs against
error signatures drawn from docs/TROUBLESHOOTING.md and the resolved issue
history, and prints a fix plan whose commands are inert strings.
"""

from . import engine  # noqa: F401
from . import risk  # noqa: F401
from . import schema_lint  # noqa: F401

__all__ = ["engine", "risk", "schema_lint"]
