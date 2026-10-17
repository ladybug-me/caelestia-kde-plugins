"""dedupe — the cleanup brain (capability 5): duplicate detection
(size -> partial hash -> full hash), last-access survival ranking,
and TTL quarantine instead of deletion. Every destructive step is a
journaled move with a restore path."""
