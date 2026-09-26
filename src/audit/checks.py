"""Day-1 data checks: string values used by the aggregators (plan §2.3) and
event ↔ tracking coordinate alignment (plan §3.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_io.load import Match
from ..data_io.transform import event_xy_to_raw

# Literal values the vendored aggregators / our filters compare against.
EXPECTED_VALUES = {
    "event_type": {"on_ball_engagement", "passing_option", "player_possession", "off_ball_run"},
    "furthest_line_break": {"first", "second_last", "last"},
}
EXPECTED_PHASE_VALUES = {
    "team_out_of_possession_phase_type": {
        "low_block", "medium_block", "high_block", "defending_transition",
        "defending_quick_break", "defending_direct", "defending_set_play", "chaotic"},
    "third_end": {"attacking_third", "middle_third", "defensive_third"},
}


def reconcile_strings(events: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    """For each column: values in the data vs values the code expects.

    Anything in ``missing_in_data`` would make a filter silently return 0 rows.
    """
    rows = []
    for df, spec in ((events, EXPECTED_VALUES), (phases, EXPECTED_PHASE_VALUES)):
        for col, expected in spec.items():
            found = set(df[col].dropna().unique()) if col in df else set()
            rows.append({"column": col,
                         "found": sorted(found),
                         "missing_in_data": sorted(expected - found),
                         "unexpected_in_data": sorted(found - expected)})
    return pd.DataFrame(rows)


def event_tracking_alignment(match: Match, frames_norm: pd.DataFrame,
                             tracking_norm: pd.DataFrame, n: int = 500) -> dict:
    """Distance between a player_possession event's (x_start, y_start) and the
    carrier's tracking position at frame_start.

    * ``err_norm``: event x/y as-is vs NORMALISED tracking — should be < ~1 m
      (events are already direction-normalised, in metres).
    * ``err_raw``: ``event_xy_to_raw`` vs RAW tracking — should also be < ~1 m.
    """
    ev = match.events[match.events.event_type == "player_possession"]
    ev = ev.dropna(subset=["x_start", "y_start", "player_id"]).head(n)
    ev = event_xy_to_raw(ev)
    raw = match.tracking[["frame", "player_id", "x", "y"]]
    j = ev.merge(raw, left_on=["frame_start", "player_id"], right_on=["frame", "player_id"])
    d_raw = np.hypot(j.x_start_raw - j.x, j.y_start_raw - j.y)
    jn = ev.merge(tracking_norm[["frame", "player_id", "x", "y"]],
                  left_on=["frame_start", "player_id"], right_on=["frame", "player_id"])
    d_norm = np.hypot(jn.x_start - jn.x, jn.y_start - jn.y)
    return {"match_id": match.match_id, "n_raw": len(j),
            "median_err_raw_m": float(np.median(d_raw)) if len(j) else np.nan,
            "p90_err_raw_m": float(np.quantile(d_raw, .9)) if len(j) else np.nan,
            "n_norm": len(jn),
            "median_err_norm_m": float(np.median(d_norm)) if len(jn) else np.nan,
            "p90_err_norm_m": float(np.quantile(d_norm, .9)) if len(jn) else np.nan}
