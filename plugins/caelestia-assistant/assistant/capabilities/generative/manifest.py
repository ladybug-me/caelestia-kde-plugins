"""Manifest for the generative capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="generative",
    permissions=("net:loopback-only", "fs:read",),
    risk_tier="read_only",
    latency_budget_ms=4000,
    memory_budget_mb=40,
)
