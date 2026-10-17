"""CLI shim: python -m assistant.capabilities.diagnostics.cli <cmd>

Delegates to engine.main so the command surface stays in one place.
"""

from .engine import main

if __name__ == "__main__":
    raise SystemExit(main())
