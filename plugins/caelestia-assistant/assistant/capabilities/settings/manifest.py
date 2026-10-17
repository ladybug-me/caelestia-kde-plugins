"""Manifest for the settings capability: permissions, risk tier, budgets."""
from assistant.capabilities._registry import CapManifest

MANIFEST = CapManifest(
    name="settings",
    permissions=("settings:read", "settings:write-behind-confirm",),
    risk_tier="confirm",
    latency_budget_ms=600,
    memory_budget_mb=40,
)
