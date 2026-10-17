"""Proposal ledger: every autonomous change is a proposal until the user decides.

Approve/reject decisions are the only training signal the learners use, so the
ledger is also the audit trail. Nothing here touches the notes themselves.
"""
import json
import os
import pathlib
from datetime import datetime, timezone


class Ledger:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.items = self._load()

    def _load(self):
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8")).get("proposals", [])

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"proposals": self.items}, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def propose(self, kind, target, diff, reason, confidence):
        pid = max((i["id"] for i in self.items), default=0) + 1
        self.items.append({
            "id": pid, "kind": kind, "target": target, "diff": diff,
            "reason": reason, "confidence": round(float(confidence), 3),
            "status": "pending", "decided_at": None,
        })
        self._save()
        return pid

    def _get(self, pid):
        for item in self.items:
            if item["id"] == pid:
                return item
        raise KeyError(f"no proposal {pid}")

    def decide(self, pid, approve):
        item = self._get(pid)
        if item["status"] != "pending":
            raise ValueError(f"proposal {pid} already {item['status']}")
        item["status"] = "approved" if approve else "rejected"
        item["decided_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()
        return item

    def pending(self):
        return [i for i in self.items if i["status"] == "pending"]

    def labeled(self, kind=None):
        out = [i for i in self.items if i["status"] in ("approved", "rejected")]
        return [i for i in out if kind is None or i["kind"] == kind]
