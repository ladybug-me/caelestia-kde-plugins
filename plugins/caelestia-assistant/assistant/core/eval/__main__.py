"""``python3 -m assistant.core.eval [suite] [--json] [--sealed]`` — arena CLI."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
