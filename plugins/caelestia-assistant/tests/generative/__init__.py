"""Tests for Layer 3 (generative): offline only.

No test in this package opens a real socket: every network touch goes through
an injected fake connection factory. The loopback guard, the SUGGESTED_NOT_
EXECUTED post-processing, the destructive-command withholding, the disabled
paths, and the import policy are all verified without any server.
"""
