"""Layer 3 transport: a minimal, loopback-only HTTP client for a local Ollama.

This file is the ONLY network surface in the entire assistant. Guarantees:

- Loopback only: the target URL is parsed and validated BEFORE any connection
  is opened. Any host other than localhost / 127.0.0.1 / [::1] raises
  LoopbackViolation — including values read from CAELESTIA_ASSISTANT_OLLAMA_URL.
  Only the PORT is configurable (via the URL); the host is not.
- Single attempt: one TCP connect for is_available(), one POST for generate().
  No retries, no backoff, no pools, no background work; short timeout (10s).
- stdlib http.client only. No urllib (URL parsing is done by a tiny local
  parser so that neither urllib nor socket is ever imported), no subprocess,
  no file writes. The loopback-only entry is pinned by the generative tests
  as the dotted name "http.client"; bare "http" stays forbidden.
- Injectable: both entry points accept a connection factory so tests can
  supply fakes and never open a real socket.
- Silent by construction: nothing is written to disk, nothing is executed;
  responses are returned to the caller as plain strings.
"""

from __future__ import annotations

import http.client
import json
import os
from typing import Any, Callable, Optional, Tuple

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
DEFAULT_MODEL = "llama3"  # matches shell/plugin/src/Caelestia/Config/aiconfig.hpp defaultOllamaModel
DEFAULT_TIMEOUT_S = 10.0

ENV_URL = "CAELESTIA_ASSISTANT_OLLAMA_URL"
ENV_MODEL = "CAELESTIA_ASSISTANT_OLLAMA_MODEL"

# The complete set of acceptable hosts. Deliberately NOT "any 127.x.y.z" or
# "anything that resolves to loopback": exact string matching keeps the guard
# total, offline, and free of DNS/DNS-rebinding considerations.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

GENERATE_PATH = "/api/generate"

# factory(host, port, timeout) -> object with .request()/.getresponse()/.close()
# (and .connect() for the availability probe). http.client.HTTPConnection
# satisfies this protocol; tests substitute fakes.
ConnectionFactory = Callable[[str, int, float], Any]


class LoopbackViolation(ValueError):
    """Raised before any connection when a URL target is not loopback."""


class GenerativeError(RuntimeError):
    """Raised when the local Ollama server answers unusably (HTTP/JSON)."""


def resolve_url(url: Optional[str] = None) -> str:
    """Explicit argument > CAELESTIA_ASSISTANT_OLLAMA_URL > default loopback."""
    if url and url.strip():
        return url.strip()
    env_value = os.environ.get(ENV_URL, "").strip()
    return env_value or DEFAULT_URL


def resolve_model(model: Optional[str] = None) -> str:
    """Explicit argument > CAELESTIA_ASSISTANT_OLLAMA_MODEL > "llama3"."""
    if model and model.strip():
        return model.strip()
    return os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL


def parse_loopback(url: str) -> Tuple[str, int]:
    """Validate a URL as loopback-only; return (host, port) or raise.

    Called before EVERY connection attempt, so a non-loopback host can never
    reach the socket layer. Accepts only "http://<loopback-host>[:<port>]":
    no other scheme, no userinfo, no funny grammars. The error message names
    the rejected host only; it never needs to echo credentials.
    """
    text = url.strip()
    if not text.lower().startswith("http://"):
        raise LoopbackViolation(f"only plain-http loopback URLs are allowed: {text[:60]!r}")
    rest = text[len("http://"):]

    # Cut off path / query / fragment; whatever remains must be host[:port].
    for sep in ("/", "?", "#"):
        rest = rest.split(sep, 1)[0]
    if not rest:
        raise LoopbackViolation("URL has an empty host")
    if "@" in rest:
        raise LoopbackViolation("userinfo in URL is not accepted")

    port_text = ""
    if rest.startswith("["):  # IPv6 literal, e.g. [::1]:11434
        closing = rest.find("]")
        if closing == -1:
            raise LoopbackViolation("unterminated IPv6 literal in URL")
        host, port_text = rest[1:closing], rest[closing + 1:]
    elif ":" in rest:
        host, _, port_text = rest.rpartition(":")
    else:
        host = rest

    if port_text:
        if port_text.startswith(":"):
            port_text = port_text[1:]
        if not port_text.isdigit():
            raise LoopbackViolation(f"invalid port in URL: {port_text[:20]!r}")
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise LoopbackViolation(f"port out of range: {port}")
    else:
        port = DEFAULT_PORT

    host = host.strip("[]").lower()
    if host not in LOOPBACK_HOSTS:
        raise LoopbackViolation(
            f"non-loopback host rejected (allowed: localhost, 127.0.0.1, [::1]): {host!r}"
        )
    return host, port


def _default_connection_factory(host: str, port: int, timeout: float) -> Any:
    """One fresh http.client.HTTPConnection; nothing is pooled or reused."""
    return http.client.HTTPConnection(host, port=port, timeout=timeout)


def is_available(
    url: Optional[str] = None,
    conn_factory: Optional[ConnectionFactory] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> bool:
    """True iff one TCP connect to the (validated) loopback target succeeds.

    Exactly one attempt; any transport failure (refused, timeout, factory
    error) simply returns False. Never raises for an unreachable server;
    a non-loopback URL returns False without ever attempting a connection.
    """
    try:
        host, port = parse_loopback(resolve_url(url))
    except LoopbackViolation:
        return False
    factory = conn_factory or _default_connection_factory
    try:
        conn = factory(host, port, timeout)
        conn.connect()
        conn.close()
        return True
    except Exception:  # any failure at all (refused/timeout/factory) => unavailable
        return False


def generate(
    prompt: str,
    model: Optional[str] = None,
    url: Optional[str] = None,
    conn_factory: Optional[ConnectionFactory] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """POST one /api/generate request; return the model's raw text output.

    Single attempt, no streaming, no retries. Transport errors (the OSError
    family, including timeouts) propagate to the caller — the RAG layer turns
    them into a graceful "unavailable" result, never into a retry.
    """
    host, port = parse_loopback(resolve_url(url))
    factory = conn_factory or _default_connection_factory
    payload = json.dumps({"model": resolve_model(model), "prompt": prompt, "stream": False})
    body = payload.encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    conn = factory(host, port, timeout)
    try:
        # Exactly one request. There is deliberately no loop around this.
        conn.request("POST", GENERATE_PATH, body=body, headers=headers)
        response = conn.getresponse()
        status = response.status
        raw = response.read()
    finally:
        conn.close()

    if status != 200:
        head = raw[:200].decode("utf-8", "replace")
        raise GenerativeError(f"ollama answered HTTP {status}: {head}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenerativeError(f"ollama returned a non-JSON body: {exc}") from exc
    text = data.get("response") if isinstance(data, dict) else None
    if not isinstance(text, str):
        raise GenerativeError("ollama JSON body is missing the 'response' text field")
    return text
