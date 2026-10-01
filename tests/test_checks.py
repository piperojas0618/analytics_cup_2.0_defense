"""Tests for audit.checks.event_tracking_alignment, synthetic + real-data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit.checks import event_tracking_alignment
from src.data_io.load import Match
from src.data_io.transform import attacking_team_per_frame, build_normalised, normalise_direction


def _normalise(m):
    """Synthetic matches here have no phases; skip label_frames (same as
    test_transform's synthetic tests) and normalise directly."""
    f = attacking_team_per_frame(m.frames, m)
    return normalise_direction(f, m.tracking, m)

CAM_COLS = ["x_top_left", "y_top_left", "x_bottom_left", "y_bottom_left",
            "x_bottom_right", "y_bottom_right", "x_top_right", "y_top_right"]


def _synthetic_match():
    """Two frames, home attacking left_to_right (no sign flip), one
    player_possession event (checked at frame_start) and one passing_option
    event (checked at frame_end) with correct event<->tracking coordinates."""
    meta = {"id": 1, "home_team": {"id": 1, "name": "H"}, "away_team": {"id": 2, "name": "A"},
            "home_team_side": ["left_to_right", "right_to_left"],
            "pitch_length": 105, "pitch_width": 68, "players": []}
    players = pd.DataFrame({
        "player_id": [10, 11], "team_id": [1, 2], "is_gk": [False, False],
        "role": ["CF", "CF"], "short_name": ["a", "b"], "number": [9, 9]})
    frames = pd.DataFrame({
        "frame": [0, 1], "period": pd.array([1, 1], dtype="Int64"),
        "possession_group": ["home team", "home team"],
        "ball_x": [0.0, 0.0], "ball_y": [0.0, 0.0],
        **{k: [np.nan] * 2 for k in CAM_COLS}})
    tracking = pd.DataFrame({
        "frame": [0, 0, 1, 1], "player_id": [10, 11, 10, 11],
        "x": [5.0, -5.0, 6.0, -6.0], "y": [1.0, -1.0, 2.0, -2.0], "is_detected": True})
    events = pd.DataFrame({
        "event_id": ["pp_0", "po_0"],
        "event_type": ["player_possession", "passing_option"],
        "frame_start": [0, 0], "frame_end": [0, 1],
        "player_id": [10, 11],
        "x_start": [5.0, -5.0], "y_start": [1.0, -1.0],
        "x_end": [5.0, -6.0], "y_end": [1.0, -2.0],
        "xthreat": [np.nan, 0.05],
        "attacking_side": ["left_to_right", "left_to_right"],
    })
    return Match(1, meta, players, frames, tracking, events, pd.DataFrame())


def test_player_possession_checked_at_frame_start():
    m = _synthetic_match()
    _, tn = _normalise(m)
    out = event_tracking_alignment(m, tn)
    pp = out[out.event_type == "player_possession"].iloc[0]
    assert pp.check_frame == "frame_start"
    assert pp.d_norm == pytest.approx(0.0)
    assert pp.d_raw == pytest.approx(0.0)


def test_passing_option_checked_at_frame_end():
    m = _synthetic_match()
    _, tn = _normalise(m)
    out = event_tracking_alignment(m, tn)
    po = out[out.event_type == "passing_option"].iloc[0]
    assert po.check_frame == "frame_end"
    assert po.d_norm == pytest.approx(0.0)
    assert po.xthreat == pytest.approx(0.05)


def test_passing_option_mismatch_is_flagged():
    m = _synthetic_match()
    m.events.loc[m.events.event_id == "po_0", ["x_end", "y_end"]] = [50.0, 50.0]
    _, tn = _normalise(m)
    out = event_tracking_alignment(m, tn)
    po = out[out.event_type == "passing_option"].iloc[0]
    assert po.d_norm > 10


def test_both_event_types_present_and_tagged():
    m = _synthetic_match()
    _, tn = _normalise(m)
    out = event_tracking_alignment(m, tn)
    assert set(out.event_type) == {"player_possession", "passing_option"}
    assert set(out.match_id) == {1}


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def _real_ids():
    try:
        from src.config import load_config
        from src.data_io.load import list_match_ids
        d = load_config()["data"]["matches_dir"]
        return d, list_match_ids(d)[:1]
    except Exception:
        return None, []


_DIR, _IDS = _real_ids()


@pytest.mark.skipif(not _IDS, reason="opendata not available")
def test_real_passing_options_carry_xthreat():
    from src.data_io.load import load_match
    m = load_match(_DIR, _IDS[0])
    _, tn = _normalise(m)
    out = event_tracking_alignment(m, tn)
    po = out[out.event_type == "passing_option"]
    assert len(po) > 0
    assert po.xthreat.notna().all()
    assert po.d_norm.median() < 1.0
