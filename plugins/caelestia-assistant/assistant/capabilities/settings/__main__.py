"""CLI shim: python3 -m assistant.capabilities.settings "<text>" [--apply] [--file PATH]

Delegates to cli.main so the command surface stays in one place. Dry-run is
the default: without --apply this prints the validated plan and writes
nothing at all.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
