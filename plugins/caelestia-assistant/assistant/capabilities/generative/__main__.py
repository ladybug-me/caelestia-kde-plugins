"""CLI shim: python3 -m assistant.capabilities.generative "problem" [--generative] [-k N]

Delegates to rag.main so the command surface stays in one place. The
generative layer is OFF unless --generative is passed AND a loopback Ollama
server answers; without it this prints the disabled notice plus the Layer 2
retrieval hits.
"""

from .rag import main

if __name__ == "__main__":
    raise SystemExit(main())
