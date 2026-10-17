#!/usr/bin/env bash
# On-device assistant: all five suites — Layer 1 (deterministic rules),
# Layer 2 (offline retrieval), Layer 3 (optional generative), Layer 4 (issue
# drafting), and the settings layer (#120 Phase 1: natural-language settings
# editing) — rule files valid, no-executor import policy holds, historical
# regression cases match, corpus/index/search behave, generative safety
# guarantees hold, issue drafting round-trips, settings intents validate and
# gated applies hold.

set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/.." && pwd)"

source "$TESTS_DIR/helpers.sh"

run_assistant_suite() {
    local suite_dir="$1"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 not available; skipping assistant tests" >&2
        return 0
    fi
    # -t "$REPO_ROOT" keeps `assistant` a proper package (relative imports
    # stay valid) while discovering only this suite's tests.
    PYTHONPATH="$REPO_ROOT" python3 -m unittest discover \
        -s "$REPO_ROOT/$suite_dir" -t "$REPO_ROOT" >/dev/null 2>&1
    assert_status 0 "$?" "assistant suite $suite_dir"
}

test_assistant_layer1() {
    run_assistant_suite "tests/diagnostics"
}

test_assistant_layer2() {
    run_assistant_suite "tests/retrieval"
}

test_assistant_layer3() {
    run_assistant_suite "tests/generative"
}

test_assistant_layer4() {
    run_assistant_suite "tests/issues"
}

test_assistant_layer5() {
    run_assistant_suite "tests/settings"
}

test_assistant_brain() {
    run_assistant_suite "tests/brain"
}

test_assistant_layer1
test_assistant_layer2
test_assistant_layer3
test_assistant_layer4
test_assistant_layer5
test_assistant_brain
test_agent_suite() { run_assistant_suite "tests/agent"; }
test_scan_suite()   { run_assistant_suite "tests/scan"; }
test_cortex_suite() { run_assistant_suite "tests/core"; }
test_shellkb_suite() { run_assistant_suite "tests/shellkb"; }
test_genius_suite() { run_assistant_suite "tests/genius"; }
test_agent_suite
test_scan_suite
test_cortex_suite
test_shellkb_suite
test_genius_suite

test_properties_suite() {
  # F25 (exponential-build-5): the seeded, shrinking property helper
  # and the three safety properties (apply-then-undo identity, no
  # out-of-range plans, hub flag order). Runs in its own interpreter
  # with the repo root on the path (the file sets it itself too).
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not available; skipping property tests" >&2
    return 0
  fi
  local out code
  out=$(PYTHONPATH="$REPO_ROOT" timeout 300 python3 \
        "$TESTS_DIR/test_properties.py" 2>&1)
  code=$?
  if [ "$code" != "0" ]; then
    echo "FAIL - assistant property suite"
    echo "$out" | tail -20
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_NAMES+=("assistant property suite")
    return 1
  fi
  echo "$out" | grep "^prop ok:" || true
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "ok   - assistant property suite (seeded, shrinking)"
}
test_properties_suite

test_conformance_suite() {
  # A9 (exponential-build-5): the issue #120 bullet-by-bullet
  # conformance suite with the honest status table.
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not available; skipping conformance tests" >&2
    return 0
  fi
  PYTHONPATH="$REPO_ROOT" python3 -m unittest discover \
    -s "$REPO_ROOT/tests" -t "$REPO_ROOT" >/dev/null 2>&1
  assert_status 0 "$?" "tests/ (issue #120 conformance)"
}
test_conformance_suite

test_lazy_imports() {
  # R4 (exponential-build-5): `--help` must not import engine modules.
  # Needs a fresh interpreter, so it lives here (the Python suite runs
  # in-process and earlier tests may already have imported engines).
  local hits
  hits=$(python3 -X importtime -m assistant.hub --help 2>&1 >/dev/null \
         | grep -cE "assistant\.(capabilities\.(agent|genius|diagnostics)|core\.router)" || true)
  if [ "$hits" != "0" ]; then
    echo "FAIL: --help imported engine modules ($hits hits)"
    exit 1
  fi
  echo "lazy imports: --help pulls no engines (ok)"
}
test_golden_lock() {
  # Stage C1 behavior lock: the golden CLI outputs, byte-identical.
  # (count lives in the manifest: cases + min_cases — never a constant)
  # Lives in scripts/ (the import policy forbids subprocess inside
  # assistant/ Python); the unittest tree pins manifest structure.
  if [ ! -f "$REPO_ROOT/tests/goldens/manifest.json" ]; then
    echo "FAIL: golden manifest missing (run scripts/regen_goldens.py)"
    exit 1
  fi
  PYTHONPATH="$REPO_ROOT" python3 "$REPO_ROOT/scripts/golden_check.py" >/dev/null 2>&1
  assert_status 0 "$?" "golden lock: golden CLI outputs byte-identical"
}

test_lazy_imports
test_golden_lock

finish
