"""THE coordinate transform, plus frame → phase labelling.

Only one normalisation exists in this project: the **possession-based** flip
from SkillCorner's Part4 tutorial. After it, the team in possession always
attacks toward +x, and the defending team's goal is at +x.

Why not Part7's flip? Part7 flips each player by *their own team's* direction,
so both teams end up "attacking left→right" — it mirrors the two teams onto
each other. Fine for average-shape plots, fatal for pitch control. Never use it
here (plan §2.1).

Implementation note: Part4 builds a per-row mask. It is equivalent to a single
per-frame sign  s = -1 if the attacking team plays right_to_left in this period
else +1, applied to *every* coordinate in the frame (players, ball, camera
polygon). Written that way the no-possession bug (plan §2.2) cannot happen:
frames without an attacking team get s = NaN and are dropped, instead of home
and away being flipped in opposite directions.

The flip is a 180° rotation (x and y both negate), so left/right channels are
preserved from the attacker's point of view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import CORNER_KEYS, Match, attacking_direction_by_period

PHASE_COLS = [
    "phase_index", "team_in_possession_id", "team_out_of_possession_id",
    "team_in_possession_phase_type", "team_out_of_possession_phase_type",
    "third_end", "channel_end", "team_possession_lead_to_shot",
    "team_possession_lead_to_goal",
]


# ---------------------------------------------------------------------------
# Frame → phase
# ---------------------------------------------------------------------------
def label_frames(frames: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    """Attach phase columns to each frame (Part7's IntervalIndex join).

    Frames outside every phase get ``phase_index = -1`` and NaN phase columns
    (they are kept, so the audit can count them). Phases are half-open
    [frame_start, frame_end).
    """
    phases = phases.sort_values("frame_start").reset_index(drop=True)
    idx = pd.IntervalIndex.from_arrays(phases.frame_start, phases.frame_end, closed="left")
    if idx.is_overlapping:
        raise ValueError("phases overlap — frame→phase assignment would be ambiguous")
    matched = idx.get_indexer(frames.frame.to_numpy())
    out = frames.copy()
    out["in_phase"] = matched != -1
    ph = phases.reindex(matched)[PHASE_COLS].reset_index(drop=True)
    ph.index = out.index
    for c in PHASE_COLS:
        out[c] = ph[c]
    out["phase_index"] = out["phase_index"].fillna(-1).astype(int)
    return out


# ---------------------------------------------------------------------------
# Attacking team per frame
# ---------------------------------------------------------------------------
def attacking_team_per_frame(frames: pd.DataFrame, match: Match) -> pd.DataFrame:
    """Add ``attacking_team_id`` (team in possession) and a consistency flag.

    Source of truth, in order:
      1. the phase's ``team_in_possession_id`` (if the frame is in a phase),
      2. otherwise ``possession_group`` from tracking.
    ``possession_consistent`` is False when (1) and (2) disagree — reported by
    the audit, and usable as a filter.
    """
    out = frames.copy()
    grp = out.possession_group.map({"home team": match.home_team_id,
                                    "away team": match.away_team_id})
    phase_team = out.get("team_in_possession_id")
    if phase_team is not None:
        att = phase_team.where(phase_team.notna(), grp)
        out["possession_consistent"] = phase_team.isna() | grp.isna() | (phase_team == grp)
    else:
        att = grp
        out["possession_consistent"] = True
    out["attacking_team_id"] = att.astype("Int64")
    out["defending_team_id"] = out.attacking_team_id.map(
        {match.home_team_id: match.away_team_id, match.away_team_id: match.home_team_id}
    ).astype("Int64")
    return out


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
def frame_sign(frames: pd.DataFrame, meta: dict) -> pd.Series:
    """+1 / -1 per frame so that the attacking team attacks toward +x. NaN if undefined."""
    directions = attacking_direction_by_period(meta)
    keys = zip(frames.attacking_team_id.astype("float"), frames.period.astype("float"))
    sign = []
    for team, period in keys:
        if np.isnan(team) or np.isnan(period):
            sign.append(np.nan)
            continue
        d = directions.get((int(team), int(period)))
        sign.append(np.nan if d is None else (1.0 if d == "left_to_right" else -1.0))
    return pd.Series(sign, index=frames.index, dtype="float64")


def normalise_direction(frames: pd.DataFrame, tracking: pd.DataFrame,
                        match: Match) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Possession-based flip (Part4). Returns (frames_norm, tracking_norm).

    Requires ``attacking_team_id`` on ``frames`` (see ``attacking_team_per_frame``).
    Frames without a defined attacking team are DROPPED — the flip is not
    defined for them (plan §2.2).

    tracking_norm gains: team_id, is_gk, role, in_possession, is_defender,
    is_outfield_defender, and x/y in normalised coordinates.
    """
    f = frames.copy()
    f["sign"] = frame_sign(f, match.meta)
    f = f[f.sign.notna()].copy()
    s = f.sign.to_numpy()
    for c in ["ball_x", "ball_y"] + CORNER_KEYS:
        f[c] = f[c].to_numpy() * s

    t = tracking.merge(
        f[["frame", "sign", "attacking_team_id", "defending_team_id"]], on="frame", how="inner")
    t = t.merge(match.players[["player_id", "team_id", "is_gk", "role", "short_name", "number"]],
                on="player_id", how="left")
    if t.team_id.isna().any():
        missing = t.loc[t.team_id.isna(), "player_id"].unique()[:5]
        raise ValueError(f"tracking players missing from match.json: {missing}")
    t["x"] = t.x * t.sign
    t["y"] = t.y * t.sign
    t["in_possession"] = t.team_id == t.attacking_team_id
    t["is_defender"] = t.team_id == t.defending_team_id
    t["is_outfield_defender"] = t.is_defender & ~t.is_gk
    t = t.drop(columns=["sign"])
    return f, t


def build_normalised(match: Match) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full Day-1 pipeline for one match: label → attacking team → transform."""
    frames = label_frames(match.frames, match.phases)
    frames = attacking_team_per_frame(frames, match)
    return normalise_direction(frames, match.tracking, match)


# ---------------------------------------------------------------------------
# Events: coordinates
# ---------------------------------------------------------------------------
def event_xy_to_raw(events: pd.DataFrame, cols=(("x_start", "y_start"), ("x_end", "y_end"))) -> pd.DataFrame:
    """Dynamic-event coordinates → raw tracking coordinates.

    Checked empirically (``audit.checks.event_tracking_alignment``): SkillCorner
    dynamic-event x/y are ALREADY in metres AND already direction-normalised
    (team in possession attacks +x) — the same frame as ``normalise_direction``.
    So events need NO transform to overlay on normalised tracking. This helper
    only exists for overlaying events on raw (un-normalised) tracking:
    flip where ``attacking_side == 'right_to_left'``.
    """
    e = events.copy()
    s = np.where(e.attacking_side == "right_to_left", -1.0, 1.0)
    for cx, cy in cols:
        if cx in e:
            e[f"{cx}_raw"] = e[cx] * s
            e[f"{cy}_raw"] = e[cy] * s
    return e
