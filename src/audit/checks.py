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


def event_tracking_alignment(match: Match, tracking_norm: pd.DataFrame) -> pd.DataFrame:
    """Distance between an event's own x/y and the tracked player's position
    at the matching frame.

    * ``player_possession``: ball carrier's (x_start, y_start) at frame_start.
    * ``passing_option``: the option's (x_end, y_end) at frame_end — the
      location the option's value (``xthreat``) is attached to, used
      downstream for the value surface (``features.value_surface.vxy``).

    ``d_norm``: event x/y as-is vs NORMALISED tracking — should be < ~1 m
    (events are already direction-normalised, in metres).
    ``d_raw``: ``event_xy_to_raw`` vs RAW tracking — should also be < ~1 m.
    """
    specs = [
        ("player_possession", "frame_start", "x_start", "y_start"),
        ("passing_option", "frame_end", "x_end", "y_end"),
    ]
    raw = match.tracking[["frame", "player_id", "x", "y"]]
    norm = tracking_norm[["frame", "player_id", "x", "y"]]
    out = []
    for event_type, frame_col, x_col, y_col in specs:
        ev = match.events[match.events.event_type == event_type]
        ev = ev.dropna(subset=[x_col, y_col, "player_id"])
        ev = event_xy_to_raw(ev)
        j = ev.merge(raw, left_on=[frame_col, "player_id"], right_on=["frame", "player_id"])
        d_raw = np.hypot(j[f"{x_col}_raw"] - j.x, j[f"{y_col}_raw"] - j.y)
        jn = ev.merge(norm, left_on=[frame_col, "player_id"], right_on=["frame", "player_id"])
        d_norm = np.hypot(jn[x_col] - jn.x, jn[y_col] - jn.y)
        jn["d_norm"] = d_norm
        jn["d_raw"] = d_raw
        jn["match_id"] = match.match_id
        jn["check_frame"] = frame_col
        out.append(jn)
    return pd.concat(out, ignore_index=True)
