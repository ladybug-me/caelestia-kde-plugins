"""Optional ONNX encoder seam (model is NEVER bundled).

Upstream constraint: no model files in the repo or the wheel. When a
user supplies an ONNX encoder via CAELESTIA_ENCODER_ONNX (and installs
the `encode` extra), the seam loads it; otherwise every surface returns
an honest {"available": False} and callers keep the pure-Python path.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

__all__ = ["load_encoder"]

_STATE: Dict[str, Any] = {"checked": False, "session": None, "dim": None}


def _try_load() -> None:
    _STATE["checked"] = True
    path = os.environ.get("CAELESTIA_ENCODER_ONNX", "")
    if not path or not os.path.isfile(path):
        return
    try:
        import onnxruntime as ort  # optional extra
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # bounded: the governor's CPU budget
        _STATE["session"] = ort.InferenceSession(path, opts)
        meta = _STATE["session"].get_outputs()[0].shape
        _STATE["dim"] = meta[-1] if isinstance(meta, list) and meta else None
    except Exception:  # honest degradation: any load failure disables
        _STATE["session"] = None


def load_encoder() -> Dict[str, Any]:
    """{"available": bool, "dim": int|None, "encode": fn|None}."""
    if not _STATE["checked"]:
        _try_load()
    sess = _STATE["session"]
    if sess is None:
        return {"available": False, "dim": None, "encode": None,
                "hint": "set CAELESTIA_ENCODER_ONNX to a model path and "
                        "install the 'encode' extra"}

    def encode(texts: List[str]) -> List[List[float]]:
        feeds = {i.name: t for i, t in zip(sess.get_inputs(), _inputs(texts, sess))}
        out = sess.run(None, feeds)[0]
        return [list(map(float, row)) for row in out]

    def _inputs(texts, session):
        import numpy  # onnxruntime requires numpy; guaranteed by the extra
        inp = session.get_inputs()[0]
        ids = [[min(ord(c), 255) for c in t[:512]] or [0] for t in texts]
        width = max(len(r) for r in ids)
        ids = [r + [0] * (width - len(r)) for r in ids]
        return [numpy.array(ids, dtype=_np_dtype(inp.type))]

    def _np_dtype(ort_type: str):
        import numpy
        return {"tensor(int64)": numpy.int64,
                "tensor(int32)": numpy.int32,
                "tensor(float)": numpy.float32}.get(ort_type, numpy.int64)

    return {"available": True, "dim": _STATE["dim"], "encode": encode}
