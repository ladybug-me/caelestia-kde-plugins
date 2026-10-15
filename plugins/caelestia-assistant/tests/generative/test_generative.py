"""Tests for the Layer 3 generative package. OFFLINE ONLY.

No test here opens a real socket: every network touch goes through an
injected fake connection factory. Covered:

- loopback guard: accepts only localhost / 127.0.0.1 / [::1]; rejects
  example.com, LAN addresses, other schemes, userinfo — before connecting;
- post-processing: SUGGESTED_NOT_EXECUTED prefix + risk labels; DESTRUCTIVE
  suggestions withheld and replaced by the blocked note; banner prefixed;
- suggest(): disabled path (no network), ungrounded refusal, graceful
  unavailable when the (fake) transport fails, grounded happy path;
- prompt: user text + retrieval doc titles + safety preamble present;
- policy: the import allow-list scan stays clean and only gains the exact
  dotted entry "http.client" (bare "http" stays forbidden).
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, List
from unittest import mock

from assistant.capabilities.diagnostics import risk, schema_lint
from assistant.capabilities.generative import client, rag
from assistant.capabilities.retrieval import search as retrieval_search

PROBLEM = "quickshell errors after updating to dev branch"


class _FakeResponse:
    def __init__(self, status: int = 200, body: bytes = b'{"response": "ok"}') -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body


class _FakeConnection:
    """Duck-typed http.client.HTTPConnection; records requests, never dials."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: List[dict] = []
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def request(self, method: str, path: str, body: Any = None, headers: Any = None) -> None:
        self.requests.append({"method": method, "path": path, "body": body, "headers": headers})

    def getresponse(self) -> _FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class _BrokenConnection(_FakeConnection):
    def connect(self) -> None:
        raise TimeoutError("connect timed out")


class _CountingFactory:
    """Factory that must NEVER be called in the paths under test."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, host: str, port: int, timeout: float) -> Any:
        self.calls += 1
        raise AssertionError(f"connection factory must not be called (got {host}:{port})")


def _raising_factory(host: str, port: int, timeout: float) -> Any:
    raise OSError("connection refused")


class _StdoutRecorder:
    def __init__(self) -> None:
        self.chunks: List[str] = []

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        return "".join(self.chunks)


class LoopbackGuardTests(unittest.TestCase):
    def test_accepts_loopback_urls(self) -> None:
        self.assertEqual(client.parse_loopback("http://localhost:11434"), ("localhost", 11434))
        self.assertEqual(client.parse_loopback("http://127.0.0.1:11434"), ("127.0.0.1", 11434))
        self.assertEqual(client.parse_loopback("http://[::1]:11434"), ("::1", 11434))
        self.assertEqual(client.parse_loopback("http://localhost"), ("localhost", client.DEFAULT_PORT))
        self.assertEqual(client.parse_loopback("http://127.0.0.1:9999"), ("127.0.0.1", 9999))

    def test_rejects_non_loopback_hosts(self) -> None:
        for bad in (
            "http://example.com:11434",
            "http://192.168.1.5:11434",
            "http://10.9.8.7:11434",
            "http://0.0.0.0:11434",
            "http://[fd00::1]:11434",
            "https://127.0.0.1:11434",
            "http://user@127.0.0.1:11434",
            "http://",
            "ftp://127.0.0.1",
        ):
            with self.assertRaises(client.LoopbackViolation, msg=bad):
                client.parse_loopback(bad)

    def test_rejects_before_any_connection_attempt(self) -> None:
        factory = _CountingFactory()
        with self.assertRaises(client.LoopbackViolation):
            client.generate("prompt", url="http://example.com:11434", conn_factory=factory)
        self.assertEqual(factory.calls, 0)
        self.assertFalse(client.is_available(url="http://example.com:11434", conn_factory=factory))
        self.assertEqual(factory.calls, 0)

    def test_env_url_override_is_validated_too(self) -> None:
        factory = _CountingFactory()
        with mock.patch.dict(os.environ, {client.ENV_URL: "http://192.168.1.5:11434"}):
            self.assertEqual(client.resolve_url(), "http://192.168.1.5:11434")
            with self.assertRaises(client.LoopbackViolation):
                client.parse_loopback(client.resolve_url())
            result = rag.suggest(PROBLEM, enable=True, conn_factory=factory)
        self.assertIsNone(result["suggestion"])
        self.assertIn("non-loopback host rejected", result["reason"])
        self.assertEqual(factory.calls, 0)

    def test_env_model_and_defaults(self) -> None:
        self.assertEqual(client.resolve_model(), "llama3")
        self.assertEqual(client.resolve_model("qwen2.5:3b"), "qwen2.5:3b")
        with mock.patch.dict(os.environ, {client.ENV_MODEL: "qwen2.5:3b"}):
            self.assertEqual(client.resolve_model(), "qwen2.5:3b")
        self.assertEqual(client.resolve_url(), client.DEFAULT_URL)


class ClientTransportTests(unittest.TestCase):
    def _ok_conn(self, body: bytes = b'{"response": "did the thing"}') -> _FakeConnection:
        return _FakeConnection(_FakeResponse(200, body))

    def test_generate_posts_exactly_once(self) -> None:
        conn = self._ok_conn()
        out = client.generate("hello prompt", conn_factory=lambda *args: conn)
        self.assertEqual(out, "did the thing")
        self.assertEqual(len(conn.requests), 1)  # single attempt, no retries
        req = conn.requests[0]
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["path"], client.GENERATE_PATH)
        payload = json.loads(req["body"].decode("utf-8"))
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["prompt"], "hello prompt")
        self.assertIs(payload["stream"], False)
        self.assertTrue(conn.closed)

    def test_generate_raises_on_http_error(self) -> None:
        conn = _FakeConnection(_FakeResponse(404, b'{"error": "model not found"}'))
        with self.assertRaises(client.GenerativeError):
            client.generate("x", conn_factory=lambda *args: conn)

    def test_generate_raises_on_non_json_body(self) -> None:
        conn = _FakeConnection(_FakeResponse(200, b"<html>not json</html>"))
        with self.assertRaises(client.GenerativeError):
            client.generate("x", conn_factory=lambda *args: conn)

    def test_is_available_true_with_reachable_fake(self) -> None:
        conn = self._ok_conn()
        self.assertTrue(client.is_available(conn_factory=lambda *args: conn))
        self.assertTrue(conn.connected)
        self.assertTrue(conn.closed)

    def test_is_available_false_on_transport_failure(self) -> None:
        self.assertFalse(client.is_available(conn_factory=lambda *args: _BrokenConnection(_FakeResponse())))
        self.assertFalse(client.is_available(conn_factory=_raising_factory))


class PostProcessTests(unittest.TestCase):
    RAW_OUTPUT = (
        "Try these steps:\n"
        "1. Restart the shell service:\n"
        "systemctl --user restart caelestia-shell.service\n"
        "2. Look at the logs:\n"
        "journalctl --user -u caelestia-shell --no-pager -n 50\n"
        "3. Resync the database:\n"
        "$ sudo pacman -Syu\n"
    )

    def test_banner_is_prefixed_exactly_once(self) -> None:
        text = rag.sanitize_suggestion(self.RAW_OUTPUT)
        self.assertTrue(text.startswith(rag.GENERATIVE_BANNER))
        self.assertEqual(text.count(rag.GENERATIVE_BANNER), 1)

    def test_command_lines_are_labeled_with_risk_tiers(self) -> None:
        text = rag.sanitize_suggestion(self.RAW_OUTPUT)
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} systemctl --user restart caelestia-shell.service", text)
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} journalctl --user -u caelestia-shell --no-pager -n 50", text)
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} sudo pacman -Syu", text)
        self.assertIn(f"[risk: {risk.classify('systemctl --user restart caelestia-shell.service')}]", text)
        self.assertIn("[risk: STATE_CHANGING]", text)
        self.assertIn("[risk: READ_ONLY]", text)
        self.assertIn("[risk: PRIVILEGED]", text)

    def test_prose_lines_are_left_untouched(self) -> None:
        text = rag.sanitize_suggestion(self.RAW_OUTPUT)
        for prose in ("Try these steps:", "1. Restart the shell service:", "2. Look at the logs:"):
            self.assertIn(prose, text)
            self.assertNotIn(f"{rag.NOT_EXECUTED_PREFIX} {prose}", text)

    def test_destructive_suggestion_is_withheld(self) -> None:
        raw = (
            "Steps:\n"
            "rm -rf ~/.config/caelestia\n"
            "mkfs.ext4 /dev/sda1\n"
            "journalctl --user -u caelestia-shell -n 20\n"
        )
        text = rag.sanitize_suggestion(raw)
        self.assertEqual(risk.classify("rm -rf ~/.config/caelestia"), "DESTRUCTIVE")
        self.assertEqual(risk.classify("mkfs.ext4 /dev/sda1"), "DESTRUCTIVE")
        self.assertIn(rag.BLOCKED_STEP_NOTE, text)
        self.assertEqual(text.count(rag.BLOCKED_STEP_NOTE), 2)
        self.assertNotIn("rm -rf", text)
        self.assertNotIn("mkfs", text)
        self.assertNotIn(f"{rag.NOT_EXECUTED_PREFIX} rm", text)
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} journalctl --user -u caelestia-shell -n 20", text)

    def test_prompt_marker_line_is_labeled_without_marker(self) -> None:
        text = rag.sanitize_suggestion("$ kwriteconfig6 --file kwinrc --group Plugins --key k v")
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} kwriteconfig6 --file kwinrc --group Plugins --key k v", text)
        self.assertIn("[risk: STATE_CHANGING]", text)

    def test_markdown_fence_markers_survive_inner_command_labeled(self) -> None:
        raw = "```bash\nsystemctl --user restart caelestia-shell.service\n```"
        text = rag.sanitize_suggestion(raw)
        self.assertIn("```bash", text)
        self.assertIn("```", text)
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} systemctl --user restart caelestia-shell.service", text)

    def test_markdown_heading_is_not_mistaken_for_a_prompt(self) -> None:
        text = rag.sanitize_suggestion("# Troubleshooting steps\nnothing risky here")
        self.assertIn("# Troubleshooting steps", text)
        self.assertNotIn(rag.NOT_EXECUTED_PREFIX, text)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_user_text_titles_and_preamble(self) -> None:
        hits = retrieval_search.search(PROBLEM, k=4)
        self.assertTrue(hits, "the offline index should ground this known problem")
        prompt = rag.build_rag_prompt(PROBLEM, hits)
        self.assertIn(rag.SAFETY_PREAMBLE, prompt)
        self.assertIn("grounded ONLY in the provided context excerpts", prompt)
        self.assertIn("never claim to execute anything", prompt)
        self.assertIn(PROBLEM, prompt)
        self.assertIn(hits[0]["doc_id"], prompt)
        self.assertIn(hits[0]["title"], prompt)

    def test_prompt_without_hits_says_none(self) -> None:
        prompt = rag.build_rag_prompt("anything", [])
        self.assertIn("(none)", prompt)


class SuggestTests(unittest.TestCase):
    def test_disabled_by_default_touches_no_network(self) -> None:
        factory = _CountingFactory()
        result = rag.suggest(PROBLEM, k=3, enable=False, conn_factory=factory)
        self.assertIs(result["enabled"], False)
        self.assertIs(result["available"], False)
        self.assertIsNone(result["suggestion"])
        self.assertIn("disabled", result["reason"])
        self.assertEqual(result["retrieval_hits"], retrieval_search.search(PROBLEM, k=3))
        self.assertTrue(result["retrieval_hits"])
        self.assertEqual(factory.calls, 0)

    def test_enabled_without_grounding_refuses_before_network(self) -> None:
        factory = _CountingFactory()
        result = rag.suggest("qwqzwzqxqzqwqz gibberish", k=4, enable=True, conn_factory=factory)
        self.assertEqual(result["retrieval_hits"], [], "precondition: no grounding for gibberish")
        self.assertIsNone(result["suggestion"])
        self.assertIn("refuses", result["reason"])
        self.assertEqual(factory.calls, 0)

    def test_enabled_with_failing_transport_is_graceful(self) -> None:
        result = rag.suggest(PROBLEM, k=4, enable=True, conn_factory=_raising_factory)
        self.assertIs(result["enabled"], True)
        self.assertIs(result["available"], False)
        self.assertIsNone(result["suggestion"])
        self.assertIn("unavailable", result["reason"])
        self.assertIn("no Ollama server answered", result["reason"])
        self.assertEqual(result["retrieval_hits"], retrieval_search.search(PROBLEM, k=4))
        self.assertTrue(result["retrieval_hits"])

    def test_enabled_with_non_loopback_url_is_rejected_before_connect(self) -> None:
        factory = _CountingFactory()
        result = rag.suggest(PROBLEM, enable=True, url="http://192.168.1.5:11434", conn_factory=factory)
        self.assertIsNone(result["suggestion"])
        self.assertIn("non-loopback host rejected", result["reason"])
        self.assertEqual(factory.calls, 0)

    def test_enabled_happy_path_is_grounded_and_sanitized(self) -> None:
        raw_model = (
            "Try these steps:\n"
            "1. Restart the shell service:\n"
            "systemctl --user restart caelestia-shell.service\n"
            "2. Look at the logs:\n"
            "journalctl --user -u caelestia-shell --no-pager -n 50\n"
            "3. Remove the stale config:\n"
            "sudo rm -rf /etc\n"
        )
        conns: List[_FakeConnection] = []

        def factory(host: str, port: int, timeout: float) -> _FakeConnection:
            conn = _FakeConnection(_FakeResponse(200, json.dumps({"response": raw_model}).encode("utf-8")))
            conns.append(conn)
            return conn

        result = rag.suggest(PROBLEM, k=4, enable=True, conn_factory=factory)
        self.assertIs(result["available"], True)
        self.assertTrue(result["retrieval_hits"])
        suggestion = result["suggestion"] or ""
        self.assertEqual(suggestion.count(rag.GENERATIVE_BANNER), 1)
        self.assertIn(f"{rag.NOT_EXECUTED_PREFIX} systemctl --user restart caelestia-shell.service", suggestion)
        self.assertIn("[risk: STATE_CHANGING]", suggestion)
        self.assertIn("[risk: READ_ONLY]", suggestion)
        self.assertIn(rag.BLOCKED_STEP_NOTE, suggestion)
        self.assertNotIn("rm -rf", suggestion)
        self.assertEqual(len(conns), 2)  # one availability probe + one generation POST, nothing else
        posting = [conn for conn in conns if conn.requests]
        self.assertEqual(len(posting), 1)
        self.assertEqual(posting[0].requests[0]["path"], client.GENERATE_PATH)

    def test_result_dict_has_exactly_the_documented_keys(self) -> None:
        result = rag.suggest(PROBLEM, enable=False)
        self.assertEqual(set(result.keys()), {"enabled", "available", "suggestion", "retrieval_hits", "reason"})


class PolicyTests(unittest.TestCase):
    def test_generative_client_imports_no_forbidden_module(self) -> None:
        # the import allow-list is gone; the behavioral pin is on the
        # client's own source: loopback http.client only, never a
        # general network or execution surface
        import inspect

        src = inspect.getsource(client)
        self.assertIn("http.client", src)
        for forbidden in ("import socket", "import urllib", "import subprocess",
                          "import requests", "import asyncio"):
            self.assertNotIn(forbidden, src)


class CliTests(unittest.TestCase):
    def _run_main(self, argv: List[str]) -> Any:
        recorder = _StdoutRecorder()
        original = sys.stdout
        sys.stdout = recorder
        try:
            code = rag.main(argv)
        finally:
            sys.stdout = original
        return code, recorder.text()

    def test_cli_without_flag_prints_disabled_notice_and_hits(self) -> None:
        code, output = self._run_main([PROBLEM])
        self.assertEqual(code, 0)
        self.assertIn("DISABLED", output)
        self.assertIn("disabled", output)
        self.assertIn("Layer 2 retrieval hits:", output)
        self.assertNotIn(rag.GENERATIVE_BANNER, output)

    def test_cli_with_flag_but_no_server_prints_unavailable_notice_and_hits(self) -> None:
        with mock.patch.object(client, "is_available", lambda url=None, conn_factory=None, timeout=None: False):
            code, output = self._run_main([PROBLEM, "--generative"])
        self.assertEqual(code, 0)
        self.assertIn("UNAVAILABLE", output)
        self.assertIn("no Ollama server answered", output)
        self.assertIn("Layer 2 retrieval hits:", output)

    def test_cli_json_output_is_the_documented_dict(self) -> None:
        code, output = self._run_main([PROBLEM, "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(set(payload.keys()), {"enabled", "available", "suggestion", "retrieval_hits", "reason"})
        self.assertIs(payload["enabled"], False)


if __name__ == "__main__":
    unittest.main()


class TestSanitizerBypassRegression(unittest.TestCase):
    """Regression: a prefix word must not dodge destructive withholding."""

    def test_prefix_word_cannot_hide_destructive_command(self) -> None:
        sample = "Now then rm -rf /usr/share/caelestia to reset it"
        out = rag.sanitize_suggestion(sample)
        self.assertNotIn("rm -rf /usr/share/caelestia", out)
        self.assertIn("blocked by safety review", out)

    def test_prose_mention_is_still_labeled_conservatively(self) -> None:
        out = rag.sanitize_suggestion("you could try rm -f /tmp/whatever.txt (harmless)")
        self.assertIn("SUGGESTED_NOT_EXECUTED", out)

    def test_backslash_continuation_is_merged_and_withheld(self) -> None:
        # "rm \" + newline + "-rf /": per-physical-line classification left
        # the "-rf /" half unlabeled and copy-pasteable as a shell continuation.
        out = rag.sanitize_suggestion("rm \\\n-rf /")
        self.assertNotIn("-rf", out)
        self.assertNotIn("rm \\", out)
        self.assertIn("blocked by safety review", out)

    def test_backtick_wrapped_destructive_is_withheld(self) -> None:
        out = rag.sanitize_suggestion("Run `rm -rf /` to reset.")
        self.assertNotIn("rm -rf", out)
        self.assertIn("blocked by safety review", out)

    def test_command_substitution_destructive_is_withheld(self) -> None:
        out = rag.sanitize_suggestion("Now run $(rm -rf /etc)")
        self.assertNotIn("rm -rf", out)
        self.assertIn("blocked by safety review", out)

    def test_ansi_escapes_are_stripped_and_destructive_withheld(self) -> None:
        out = rag.sanitize_suggestion("rm \x1b[31m-rf\x1b[0m /")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("-rf", out)
        self.assertIn("blocked by safety review", out)

    def test_uppercase_destructive_is_withheld(self) -> None:
        out = rag.sanitize_suggestion("RM -RF /")
        self.assertNotIn("RM -RF", out)
        self.assertIn("blocked by safety review", out)

    def test_capital_R_recursive_flag_is_destructive(self) -> None:
        # -R is recursive just like -r; it must not downgrade to STATE_CHANGING.
        self.assertEqual(risk.classify("rm -Rf ~/.config/caelestia"), "DESTRUCTIVE")
        out = rag.sanitize_suggestion("rm -Rf ~/.config/caelestia")
        self.assertNotIn("-Rf", out)
        self.assertIn("blocked by safety review", out)

    def test_find_delete_is_never_read_only(self) -> None:
        self.assertEqual(risk.classify("find ~/ -name '*.qml' -delete"), "DESTRUCTIVE")


