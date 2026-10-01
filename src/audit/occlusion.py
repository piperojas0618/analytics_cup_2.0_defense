"""Week-1 occlusion audit (plan §3.1, §5, §7 Gate 1).

Everything here operates on the output of ``transform.build_normalised`` —
i.e. normalised coordinates, team in possession attacking +x, defending goal
at x = +L/2. The "danger zone" is the defending team's defensive third:
x > L/2 - L/3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_io.load import Match

QUAD = [("x_top_left", "y_top_left"), ("x_bottom_left", "y_bottom_left"),
        ("x_bottom_right", "y_bottom_right"), ("x_top_right", "y_top_right")]


# ---------------------------------------------------------------------------
# Camera polygon
# ---------------------------------------------------------------------------
def points_in_quads(qx: np.ndarray, qy: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Vectorised point-in-convex-quadrilateral.

    qx, qy: (F, 4) polygon vertices per frame (any consistent winding).
    px, py: (P,) test points shared by all frames.
    Returns bool (F, P). Frames with NaN vertices → all False.
    """
    F = qx.shape[0]
    inside_pos = np.ones((F, px.size), dtype=bool)
    inside_neg = np.ones((F, px.size), dtype=bool)
    for i in range(4):
        j = (i + 1) % 4
        ex = (qx[:, j] - qx[:, i])[:, None]
        ey = (qy[:, j] - qy[:, i])[:, None]
        cross = ex * (py[None, :] - qy[:, i][:, None]) - ey * (px[None, :] - qx[:, i][:, None])
        inside_pos &= cross >= 0
        inside_neg &= cross <= 0
    out = inside_pos | inside_neg
    out[np.isnan(qx).any(axis=1) | np.isnan(qy).any(axis=1)] = False
    return out


def danger_zone_grid(pitch_length: float, pitch_width: float, step: float = 2.0):
    """Cell centres of the defending third (normalised coords)."""
    x0 = pitch_length / 2 - pitch_length / 3
    xs = np.arange(x0 + step / 2, pitch_length / 2, step)
    ys = np.arange(-pitch_width / 2 + step / 2, pitch_width / 2, step)
    XX, YY = np.meshgrid(xs, ys)
    return XX.ravel(), YY.ravel()


def danger_zone_visibility(frames: pd.DataFrame, pitch_length: float, pitch_width: float,
                           step: float = 2.0, chunk: int = 5000) -> np.ndarray:
    """Fraction of the defending third inside the camera polygon, per frame."""
    px, py = danger_zone_grid(pitch_length, pitch_width, step)
    qx = frames[[a for a, _ in QUAD]].to_numpy(float)
    qy = frames[[b for _, b in QUAD]].to_numpy(float)
    out = np.empty(len(frames))
    for s in range(0, len(frames), chunk):
        out[s:s + chunk] = points_in_quads(qx[s:s + chunk], qy[s:s + chunk], px, py).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Frame-level quality flags
# ---------------------------------------------------------------------------
def frame_quality(frames: pd.DataFrame, tracking: pd.DataFrame, match: Match,
                  cfg: dict) -> pd.DataFrame:
    """One row per (normalised) frame with every §5 filter as a boolean column.

    ``passes_all`` is the AND of the filters switched on in config.
    """
    ff = cfg["frame_filters"]
    q = frames[["frame", "period", "phase_index", "in_phase", "attacking_team_id",
                "defending_team_id", "team_out_of_possession_phase_type",
                "possession_consistent", "ball_x", "ball_y", "is_detected_ball"]].copy()

    d = tracking[tracking.is_outfield_defender].groupby("frame").agg(
        n_outfield_def=("is_detected", "size"),
        n_det_outfield_def=("is_detected", "sum"))
    a = tracking[tracking.in_possession & ~tracking.is_gk].groupby("frame").is_detected.sum()
    q = q.join(d, on="frame").join(a.rename("n_det_outfield_att"), on="frame")
    q[["n_outfield_def", "n_det_outfield_def", "n_det_outfield_att"]] = \
        q[["n_outfield_def", "n_det_outfield_def", "n_det_outfield_att"]].fillna(0).astype(int)

    q["danger_vis_frac"] = danger_zone_visibility(
        frames, match.pitch_length, match.pitch_width, ff["visibility_grid_step_m"])

    q["f_in_phase"] = q.in_phase
    q["f_possession_consistent"] = q.possession_consistent.astype(bool)
    q["f_detection"] = q.n_det_outfield_def >= ff["min_detected_outfield_defenders"]
    q["f_ball"] = np.isfinite(q.ball_x) & np.isfinite(q.ball_y)
    q["f_ball_detected"] = q.is_detected_ball.fillna(False).astype(bool)
    q["f_visible"] = q.danger_vis_frac >= ff["danger_zone_visible_min_frac"]
    q["f_phase_type"] = ~q.team_out_of_possession_phase_type.isin(ff["excluded_oop_phase_types"])

    active = ["f_detection", "f_visible", "f_phase_type"]
    if ff["require_in_phase"]:
        active.append("f_in_phase")
    if ff["require_ball_xy"]:
        active.append("f_ball")
    if ff["require_ball_detected"]:
        active.append("f_ball_detected")
    if ff.get("require_possession_consistent"):
        active.append("f_possession_consistent")
    q["passes_all"] = q[active].all(axis=1)
    q.attrs["active_filters"] = active
    q["match_id"] = match.match_id
    return q


# ---------------------------------------------------------------------------
# Phase-level survival
# ---------------------------------------------------------------------------
def eligible_phases(phases: pd.DataFrame, cfg: dict, analysis: bool) -> pd.Series:
    pf = cfg["phase_filters"]
    if not analysis:
        return (phases.team_out_of_possession_phase_type.isin(pf["final_phase_types"])
                & (phases.third_end == pf["third_end"]))
    else:
        return (phases.team_out_of_possession_phase_type.isin(pf["oop_phase_types"])
                & (phases.third_end == pf["third_end"]))


def phase_survival(fq: pd.DataFrame, phases: pd.DataFrame, cfg: dict, analysis: bool) -> pd.DataFrame:
    """Per eligible phase: frame pass rate per filter and the survival verdict.

    Frames of the phase that were dropped by the transform (no attacking team)
    count as failing — the denominator is the phase's full frame span.
    """
    ph = phases[eligible_phases(phases, cfg, analysis)].copy()
    ph["n_frames"] = ph.frame_end - ph.frame_start
    flag_cols = [c for c in fq.columns if c.startswith("f_")] + ["passes_all"]
    agg = fq[fq.phase_index >= 0].groupby("phase_index")[flag_cols].sum()
    agg = agg.add_prefix("n_")
    ph = ph.join(agg, on="phase_index")
    for c in flag_cols:
        ph[f"rate_{c}"] = ph[f"n_{c}"].fillna(0) / ph.n_frames
    ph["frame_pass_rate"] = ph["rate_passes_all"]
    ph["survives"] = ph.frame_pass_rate >= cfg["phase_filters"]["min_frame_pass_rate"]
    return ph


def survival_sensitivity(fq: pd.DataFrame, phases: pd.DataFrame, cfg: dict, analysis: bool = False,
                         det_floors=(5, 6, 7, 8, 9),
                         vis_mins=(0.0, 0.25, 0.5, 0.75)) -> pd.DataFrame:
    """Phase survival rate over a grid of (detection floor, visibility min).

    ``fq``/``phases`` may be concatenated across matches (keys: match_id,
    phase_index). Other frame filters stay as configured.
    """
    ff = cfg["frame_filters"]
    base = fq.f_phase_type & fq.f_in_phase
    if ff["require_ball_xy"]:
        base &= fq.f_ball
    if ff["require_ball_detected"]:
        base &= fq.f_ball_detected
    if ff.get("require_possession_consistent"):
        base &= fq.f_possession_consistent

    phases = phases[eligible_phases(phases, cfg, analysis)].copy()
    ph = phases[["match_id", "phase_index", "frame_start", "frame_end"]].copy()
    ph["n_frames"] = ph.frame_end - ph.frame_start
    min_rate = cfg["phase_filters"]["min_frame_pass_rate"]
    rows = []
    for d in det_floors:
        for v in vis_mins:
            ok = base & (fq.n_det_outfield_def >= d) & (fq.danger_vis_frac >= v)
            n_ok = fq.assign(ok=ok).groupby(["match_id", "phase_index"]).ok.sum()
            r = ph.join(n_ok, on=["match_id", "phase_index"]).ok.fillna(0) / ph.n_frames
            rows.append({"det_floor": d, "vis_min": v, "survival": float((r >= min_rate).mean())})
    return pd.DataFrame(rows).pivot(index="det_floor", columns="vis_min", values="survival")


# ---------------------------------------------------------------------------
# Detection-rate heatmap data
# ---------------------------------------------------------------------------
def detection_by_location(tracking: pd.DataFrame, frames_mask: pd.Series | None,
                          pitch_length: float, pitch_width: float, bin_m: float = 5.0,
                          who: str = "outfield_defenders") -> pd.DataFrame:
    """Counts of (detected, total) player-frames per spatial bin, normalised coords.

    Returns long df: xbin, ybin (bin centres), n, n_det — summable across matches.
    """
    t = tracking
    if frames_mask is not None:
        t = t[t.frame.isin(frames_mask[frames_mask].index)]
    if who == "outfield_defenders":
        t = t[t.is_outfield_defender]
    elif who == "outfield_attackers":
        t = t[t.in_possession & ~t.is_gk]
    xe = np.arange(-60, 60 + bin_m, bin_m)
    ye = np.arange(-40, 40 + bin_m, bin_m)
    xb = pd.cut(t.x, xe, labels=(xe[:-1] + bin_m / 2))
    yb = pd.cut(t.y, ye, labels=(ye[:-1] + bin_m / 2))
    g = t.groupby([xb, yb], observed=True).is_detected.agg(["size", "sum"]).reset_index()
    g.columns = ["xbin", "ybin", "n", "n_det"]
    g["xbin"] = g.xbin.astype(float)
    g["ybin"] = g.ybin.astype(float)
    return g


def detection_by_role(tracking: pd.DataFrame, frames_mask: pd.Series | None) -> pd.DataFrame:
    t = tracking[tracking.is_defender]
    if frames_mask is not None:
        t = t[t.frame.isin(frames_mask[frames_mask].index)]
    return t.groupby("role").is_detected.agg(["size", "sum"]).rename(
        columns={"size": "n", "sum": "n_det"}).reset_index()
