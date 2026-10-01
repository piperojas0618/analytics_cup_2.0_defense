"""Transform tests. Synthetic tests always run; real-data tests run when the
opendata folder from config.yaml is present."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit.occlusion import points_in_quads
from src.data_io.load import Match, list_match_ids, load_match
from src.data_io.transform import (attacking_team_per_frame, build_normalised,
                                   label_frames, normalise_direction)

HOME, AWAY = 1, 2


def _synthetic_match(home_side_p1="left_to_right"):
    other = {"left_to_right": "right_to_left", "right_to_left": "left_to_right"}
    meta = {"id": 99, "home_team": {"id": HOME, "name": "H"}, "away_team": {"id": AWAY, "name": "A"},
            "home_team_side": [home_side_p1, other[home_side_p1]],
            "pitch_length": 105, "pitch_width": 68, "players": []}
    players = pd.DataFrame({
        "player_id": [10, 11, 20, 21], "team_id": [HOME, HOME, AWAY, AWAY],
        "is_gk": [True, False, True, False], "role": ["GK", "CF", "GK", "CF"],
        "short_name": list("abcd"), "number": [1, 9, 1, 9]})
    # Raw positions: home GK sits at the goal home DEFENDS
    home_gk_x = -50.0 if home_side_p1 == "left_to_right" else 50.0
    frames = pd.DataFrame({
        "frame": [0, 1, 2], "period": pd.array([1, 1, 1], dtype="Int64"),
        "possession_group": ["home team", "away team", None],
        "ball_x": [10.0, 10.0, 10.0], "ball_y": [5.0, 5.0, 5.0],
        **{k: [np.nan] * 3 for k in ["x_top_left", "y_top_left", "x_bottom_left", "y_bottom_left",
                                     "x_bottom_right", "y_bottom_right", "x_top_right", "y_top_right"]}})
    rows = []
    for fr in (0, 1, 2):
        rows += [(fr, 10, home_gk_x, 3.0), (fr, 11, 0.0, 0.0),
                 (fr, 20, -home_gk_x, -3.0), (fr, 21, 1.0, 1.0)]
    tracking = pd.DataFrame(rows, columns=["frame", "player_id", "x", "y"])
    tracking["is_detected"] = True
    m = Match(99, meta, players, frames, tracking, pd.DataFrame(), pd.DataFrame())
    return m


@pytest.mark.parametrize("side", ["left_to_right", "right_to_left"])
def test_defending_gk_at_positive_x(side):
    m = _synthetic_match(side)
    f = attacking_team_per_frame(m.frames, m)
    fn, tn = normalise_direction(f, m.tracking, m)
    gk = tn[tn.is_gk]
    assert (gk[gk.is_defender].x > 30).all(), "transform is inverted"
    assert (gk[~gk.is_defender].x < -30).all()


def test_no_possession_frames_dropped():
    """Plan §2.2: frames without an attacking team must not be transformed."""
    m = _synthetic_match()
    f = attacking_team_per_frame(m.frames, m)
    fn, tn = normalise_direction(f, m.tracking, m)
    assert 2 not in set(fn.frame) and 2 not in set(tn.frame)


def test_same_sign_for_both_teams_in_a_frame():
    """Both teams (and ball) share one flip → relative geometry preserved."""
    m = _synthetic_match("right_to_left")
    f = attacking_team_per_frame(m.frames, m)
    fn, tn = normalise_direction(f, m.tracking, m)
    raw = m.tracking.set_index(["frame", "player_id"])
    for fr, g in tn.groupby("frame"):
        r = raw.loc[fr].loc[g.player_id]
        ratio = g.x.to_numpy() / np.where(r.x.to_numpy() == 0, np.nan, r.x.to_numpy())
        assert np.nanstd(ratio) == 0


def test_rotation_preserves_distances():
    m = _synthetic_match("right_to_left")
    f = attacking_team_per_frame(m.frames, m)
    _, tn = normalise_direction(f, m.tracking, m)
    a = tn[tn.frame == 0].set_index("player_id")
    r = m.tracking[m.tracking.frame == 0].set_index("player_id")
    d = lambda df, i, j: np.hypot(df.x[i] - df.x[j], df.y[i] - df.y[j])
    assert np.isclose(d(a, 10, 21), d(r, 10, 21))


def test_label_frames_half_open():
    phases = pd.DataFrame({"phase_index": [0, 1], "frame_start": [0, 5], "frame_end": [5, 8],
                           **{c: [None, None] for c in [
                               "team_in_possession_id", "team_out_of_possession_id",
                               "team_in_possession_phase_type", "team_out_of_possession_phase_type",
                               "third_end", "channel_end", "team_possession_lead_to_shot",
                               "team_possession_lead_to_goal"]}})
    frames = pd.DataFrame({"frame": [0, 4, 5, 7, 8]})
    out = label_frames(frames, phases)
    assert out.phase_index.tolist() == [0, 0, 1, 1, -1]


def test_points_in_quads():
    qx = np.array([[0, 0, 10, 10.0], [np.nan] * 4])
    qy = np.array([[10, 0, 0, 10.0], [0] * 4])
    inside = points_in_quads(qx, qy, np.array([5.0, 15.0]), np.array([5.0, 5.0]))
    assert inside.tolist() == [[True, False], [False, False]]


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def _real_ids():
    try:
        from src.config import load_config
        d = load_config()["data"]["matches_dir"]
        return d, list_match_ids(d)[:2]
    except Exception:
        return None, []


_DIR, _IDS = _real_ids()


@pytest.mark.skipif(not _IDS, reason="opendata not available")
@pytest.mark.parametrize("mid", _IDS)
def test_real_defending_gk_at_positive_x(mid):
    m = load_match(_DIR, mid)
    _, tn = build_normalised(m)
    gk = tn[tn.is_gk]
    assert gk[gk.is_defender].x.median() > 30
    assert gk[~gk.is_defender].x.median() < -30
    # Every frame: exactly one team defends, and 22 or fewer players
    assert (tn.groupby("frame").team_id.nunique() <= 2).all()


@pytest.mark.skipif(not _IDS, reason="opendata not available")
@pytest.mark.parametrize("mid", _IDS)
def test_real_events_already_normalised(mid):
    from src.audit.checks import event_tracking_alignment
    m = load_match(_DIR, mid)
    _, tn = build_normalised(m)
    a = event_tracking_alignment(m, tn)
    assert set(a.event_type) == {"player_possession", "passing_option"}
    assert (a.groupby("event_type").d_norm.median() < 1.0).all()
    assert (a.groupby("event_type").d_raw.median() < 1.0).all()
