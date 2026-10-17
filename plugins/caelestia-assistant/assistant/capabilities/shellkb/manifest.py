"""Manifest for the shellkb capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="shellkb",
    permissions=("shell-docs:read-only",),
    risk_tier="read_only",
    latency_budget_ms=300,
    memory_budget_mb=40,
)
