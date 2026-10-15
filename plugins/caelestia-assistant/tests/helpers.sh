#!/usr/bin/env bash
# Minimal assertion helpers for the shell test harness (tests/test_assistant.sh).

PASS_COUNT=0
FAIL_COUNT=0
FAILED_NAMES=()

assert_status() {
    # assert_status <expected> <actual> <name>
    local expected="$1" actual="$2" name="$3"
    if [ "$expected" = "$actual" ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "ok   - $name"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_NAMES+=("$name")
        echo "FAIL - $name (expected $expected, got $actual)"
    fi
}

assert_contains() {
    # assert_contains <haystack> <needle> <name>
    local haystack="$1" needle="$2" name="$3"
    case "$haystack" in
        *"$needle"*)
            PASS_COUNT=$((PASS_COUNT + 1))
            echo "ok   - $name"
            ;;
        *)
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_NAMES+=("$name")
            echo "FAIL - $name (missing: $needle)"
            ;;
    esac
}

finish() {
    echo
    echo "passed: $PASS_COUNT  failed: $FAIL_COUNT"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        printf 'failed: %s\n' "${FAILED_NAMES[@]}"
        exit 1
    fi
    exit 0
}
