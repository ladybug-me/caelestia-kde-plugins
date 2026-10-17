"""Accessibility checks (exp-build-5 F22): WCAG 2.x contrast auditing and
color-vision-deficiency simulation, for the what-if/preset surfaces.

Honesty notes, up front:
- WCAG 2.x relative luminance and contrast ratio are the published W3C
  formulas (WCAG 2.1 section 1.4.3/1.4.6/1.4.11 math); thresholds are the
  SC levels (AA 4.5 normal / 3.0 large & UI, AAA 7.0 normal).
- The CVD matrices are the published Machado, Oliveira & Fernandes (2009)
  full-severity matrices (IEEE TVCG 15(6), DOI 10.1109/TVCG.2009.113),
  cross-checked from the authors' project page
  (inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/). Intermediate
  severities are LINEAR BLENDS between identity (severity 0) and the
  published full-severity matrix — an explicit approximation, NOT the
  paper's per-severity series; it is used because every shipped number
  must be traceable to a published source we can cite. The blend
  preserves the paper matrices' neutral-gray axis (row sums are 1).
- Matrices apply in LINEAR RGB; the sRGB transfer function is applied
  around them (IEC 61966-2-1).
- APCA is deliberately NOT implemented: its license terms were not
  verified for this repository (the F22 spec says skip unless verified).

Pure math over stdlib; no I/O; deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

__all__ = ["hex_to_rgb", "rgb_to_hex", "relative_luminance",
           "contrast_ratio", "wcag_findings", "simulate_cvd",
           "CVD_TYPES", "SIMULATE_CITATION", "WCAG_CITATION"]

WCAG_CITATION = "W3C WCAG 2.1, SC 1.4.3/1.4.6/1.4.11 (published contrast math)"
SIMULATE_CITATION = ("Machado, Oliveira & Fernandes 2009, IEEE TVCG 15(6), "
                     "DOI 10.1109/TVCG.2009.113 (full-severity matrices; "
                     "partial severity is our documented linear blend)")

CVD_TYPES = ("protanopia", "deuteranopia", "tritanopia")

# Machado et al. 2009, full severity (1.0), row-major, linear RGB.
_MACHADO: Dict[str, Tuple[Tuple[float, float, float], ...]] = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


# ---------------------------------------------------------------------------
# color spaces
# ---------------------------------------------------------------------------

def _linearize(v: float) -> float:
    """sRGB EOTF (IEC 61966-2-1)."""
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _delinearize(v: float) -> float:
    return 12.92 * v if v <= 0.0031308 else 1.055 * (v ** (1.0 / 2.4)) - 0.055


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def hex_to_rgb(text: str) -> Tuple[float, float, float]:
    """'#rrggbb' -> (r, g, b) in 0..1 (sRGB non-linear). Raises on garbage."""
    t = (text or "").strip().lstrip("#")
    if len(t) == 3:
        t = "".join(ch * 2 for ch in t)
    if len(t) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in t):
        raise ValueError(f"not a #rrggbb color: {text!r}")
    return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(
        f"{round(_clamp01(c) * 255):02x}" for c in rgb)


# ---------------------------------------------------------------------------
# WCAG 2.x contrast
# ---------------------------------------------------------------------------

def relative_luminance(rgb: Sequence[float]) -> float:
    r, g, b = (_linearize(_clamp01(c)) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: Sequence[float], bg: Sequence[float]) -> float:
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def wcag_findings(fg: Sequence[float], bg: Sequence[float]) -> List[Dict[str, object]]:
    """The WCAG verdicts for one fg/bg pair, with the SC citation."""
    ratio = contrast_ratio(fg, bg)
    out: List[Dict[str, object]] = []
    for level, threshold, scope in (
            ("AA normal text", 4.5, "SC 1.4.3"),
            ("AA large text & UI components", 3.0, "SC 1.4.3 / 1.4.11"),
            ("AAA normal text", 7.0, "SC 1.4.6"),
    ):
        out.append({
            "level": level, "threshold": threshold, "ratio": round(ratio, 2),
            "pass": ratio >= threshold, "cite": f"{WCAG_CITATION} ({scope})",
        })
    return out


# ---------------------------------------------------------------------------
# CVD simulation (Machado 2009 endpoints, linear blend between)
# ---------------------------------------------------------------------------

def _blend_matrix(kind: str, severity: float) -> Tuple[Tuple[float, float, float], ...]:
    if kind not in _MACHADO:
        raise ValueError(f"unknown CVD type {kind!r}; have {list(_MACHADO)}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    full = _MACHADO[kind]
    if severity >= 1.0:
        return full
    if severity <= 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # documented linear blend: identity -> full severity
    return tuple(tuple((1.0 - severity) * i + severity * m
                       for i, m in zip(identity_row, machado_row))
                 for identity_row, machado_row in zip(
                     ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                     full))


def simulate_cvd(rgb: Sequence[float], kind: str = "deuteranopia",
                 severity: float = 1.0) -> Tuple[float, float, float]:
    """Simulate one color under a color-vision deficiency. Linear RGB in
    the middle, sRGB on both ends; severity 0 is the identity (property-
    tested), severity 1 is exactly the published matrix."""
    m = _blend_matrix(kind, severity)
    lin = [_linearize(_clamp01(c)) for c in rgb]
    out = tuple(_clamp01(sum(m[i][j] * lin[j] for j in range(3)))
                for i in range(3))
    return (_delinearize(out[0]), _delinearize(out[1]), _delinearize(out[2]))
