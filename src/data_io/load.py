"""Loaders for SkillCorner open data, one match at a time, loop-ready.

Adapted from SkillCorner's tutorial notebooks (Part4 / Part7) and Felipe's
``GetData.py``. Differences vs. the tutorials:

* ``image_corners_projection`` is kept (needed for the visibility mask).
* Tracking is built with plain numpy instead of ``pd.json_normalize`` over
  nested dicts (~5x faster, same columns).
* Player on-pitch intervals come from ``playing_time.total.start_frame /
  end_frame`` in match.json — frame-exact, and correct for players whose
  ``end_time`` is null because they finished the match (``GetData.py``
  hard-coded those to 90:00, which is wrong for second-half stoppage time).
* No top-N player filter: tracking frames only contain the players actually
  on the pitch, which is what VSD needs (plan §2.4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

CORNER_KEYS = [
    "x_top_left", "y_top_left",
    "x_bottom_left", "y_bottom_left",
    "x_bottom_right", "y_bottom_right",
    "x_top_right", "y_top_right",
]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class Match:
    match_id: int
    meta: dict
    players: pd.DataFrame          # one row per player who appeared
    frames: pd.DataFrame           # one row per frame: ball, possession, camera
    tracking: pd.DataFrame         # one row per (frame, player)
    events: pd.DataFrame
    phases: pd.DataFrame
    pitch_length: float = field(init=False)
    pitch_width: float = field(init=False)

    def __post_init__(self):
        self.pitch_length = float(self.meta.get("pitch_length") or 105.0)
        self.pitch_width = float(self.meta.get("pitch_width") or 68.0)

    @property
    def home_team_id(self) -> int:
        return int(self.meta["home_team"]["id"])

    @property
    def away_team_id(self) -> int:
        return int(self.meta["away_team"]["id"])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def list_match_ids(matches_dir: str | Path) -> list[int]:
    """All match folders that contain a tracking file (sorted)."""
    matches_dir = Path(matches_dir)
    ids = []
    for p in matches_dir.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / f"{p.name}_tracking_extrapolated.jsonl").exists():
            ids.append(int(p.name))
    return sorted(ids)


def _path(matches_dir: Path, match_id: int, suffix: str) -> Path:
    return Path(matches_dir) / str(match_id) / f"{match_id}_{suffix}"


def _check_not_lfs_pointer(path: Path) -> None:
    if path.stat().st_size < 1000:
        head = path.read_text(errors="ignore")[:200]
        if "git-lfs" in head:
            raise RuntimeError(
                f"{path.name} is a Git LFS pointer, not data. Run `git lfs pull` "
                "in the opendata repo (or download from media.githubusercontent.com).")


# ---------------------------------------------------------------------------
# Match metadata -> players
# ---------------------------------------------------------------------------
def load_meta(matches_dir: str | Path, match_id: int) -> dict:
    with open(_path(matches_dir, match_id, "match.json")) as f:
        return json.load(f)


def attacking_direction_by_period(meta: dict) -> dict[tuple[int, int], str]:
    """{(team_id, period): 'left_to_right' | 'right_to_left'}.

    ``home_team_side`` is a list indexed by period: the home team's attacking
    direction in period 1, 2 (and extra-time periods if present). The away team
    attacks the other way. Validated empirically by
    ``tests/test_transform.py::test_real_defending_gk_at_positive_x``.
    """
    sides = meta["home_team_side"]
    if isinstance(sides, str):  # defensive: sometimes stringified
        sides = [s.strip(" '\"") for s in sides.strip("[]").split(",")]
    other = {"left_to_right": "right_to_left", "right_to_left": "left_to_right"}
    home, away = int(meta["home_team"]["id"]), int(meta["away_team"]["id"])
    out = {}
    for i, side in enumerate(sides, start=1):
        out[(home, i)] = side
        out[(away, i)] = other[side]
    return out


def load_players(meta: dict) -> pd.DataFrame:
    """One row per player who actually appeared in the match."""
    home_id, away_id = int(meta["home_team"]["id"]), int(meta["away_team"]["id"])
    rows = []
    for p in meta["players"]:
        pt = (p.get("playing_time") or {}).get("total") or {}
        if pt.get("start_frame") is None:
            continue  # unused substitute
        rows.append({
            "player_id": int(p["id"]),
            "short_name": p.get("short_name"),
            "number": p.get("number"),
            "team_id": int(p["team_id"]),
            "home_away": "home" if int(p["team_id"]) == home_id else "away",
            "role": p["player_role"]["acronym"],
            "role_name": p["player_role"]["name"],
            "position_group": p["player_role"]["position_group"],
            "is_gk": p["player_role"]["acronym"] == "GK",
            "start_frame": int(pt["start_frame"]),
            "end_frame": int(pt["end_frame"]),
            "minutes_played": pt.get("minutes_played"),
        })
    players = pd.DataFrame(rows)
    team_names = {home_id: meta["home_team"]["name"], away_id: meta["away_team"]["name"]}
    players["team_name"] = players.team_id.map(team_names)
    return players


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
def load_tracking(matches_dir: str | Path, match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frames, tracking).

    frames:   frame, period, timestamp, possession_group, possession_player_id,
              ball_x, ball_y, ball_z, is_detected_ball, + 8 camera-corner columns.
    tracking: frame, player_id, x, y, is_detected  (only frames with players).
    Coordinates are raw SkillCorner metres, origin at the centre spot.
    """
    path = _path(matches_dir, match_id, "tracking_extrapolated.jsonl")
    _check_not_lfs_pointer(path)

    f_rows, p_frame, p_id, p_x, p_y, p_det = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            b = r.get("ball_data") or {}
            pos = r.get("possession") or {}
            cam = r.get("image_corners_projection") or {}
            f_rows.append((
                r["frame"], r.get("period"), r.get("timestamp"),
                pos.get("group"), pos.get("player_id"),
                b.get("x"), b.get("y"), b.get("z"), b.get("is_detected"),
                *[cam.get(k) for k in CORNER_KEYS],
            ))
            for pd_ in r.get("player_data") or ():
                p_frame.append(r["frame"])
                p_id.append(pd_["player_id"])
                p_x.append(pd_["x"])
                p_y.append(pd_["y"])
                p_det.append(pd_["is_detected"])

    frames = pd.DataFrame(f_rows, columns=[
        "frame", "period", "timestamp", "possession_group", "possession_player_id",
        "ball_x", "ball_y", "ball_z", "is_detected_ball", *CORNER_KEYS])
    frames["period"] = frames["period"].astype("Int64")
    for c in ["ball_x", "ball_y", "ball_z", *CORNER_KEYS]:
        frames[c] = pd.to_numeric(frames[c], errors="coerce").astype("float64")
    frames["is_detected_ball"] = frames["is_detected_ball"].astype("boolean")
    frames["match_id"] = match_id

    tracking = pd.DataFrame({
        "frame": np.asarray(p_frame, dtype=np.int64),
        "player_id": np.asarray(p_id, dtype=np.int64),
        "x": np.asarray(p_x, dtype=np.float64),
        "y": np.asarray(p_y, dtype=np.float64),
        "is_detected": np.asarray(p_det, dtype=bool),
    })
    tracking["match_id"] = match_id
    return frames, tracking


# ---------------------------------------------------------------------------
# Events & phases
# ---------------------------------------------------------------------------
def load_events(matches_dir: str | Path, match_id: int) -> pd.DataFrame:
    return pd.read_csv(_path(matches_dir, match_id, "dynamic_events.csv"), low_memory=False)


def load_phases(matches_dir: str | Path, match_id: int, meta: dict) -> pd.DataFrame:
    """Phases of play + ``team_out_of_possession_id`` (not in the CSV).

    Derived from the match's two team ids, not from ``unique()`` over the phase
    rows (that silently breaks if one team never has a phase in a period).
    """
    phases = pd.read_csv(_path(matches_dir, match_id, "phases_of_play.csv"))
    phases = phases.rename(columns={"index": "phase_index"})
    home, away = int(meta["home_team"]["id"]), int(meta["away_team"]["id"])
    phases["team_out_of_possession_id"] = phases.team_in_possession_id.map({home: away, away: home})
    if phases.team_out_of_possession_id.isna().any():
        raise ValueError(f"match {meta['id']}: phase team ids don't match match.json")
    phases["team_out_of_possession_id"] = phases.team_out_of_possession_id.astype(int)
    return phases


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------
def load_match(matches_dir: str | Path, match_id: int, with_tracking: bool = True) -> Match:
    meta = load_meta(matches_dir, match_id)
    players = load_players(meta)
    if with_tracking:
        frames, tracking = load_tracking(matches_dir, match_id)
    else:
        frames, tracking = pd.DataFrame(), pd.DataFrame()
    return Match(
        match_id=int(match_id), meta=meta, players=players,
        frames=frames, tracking=tracking,
        events=load_events(matches_dir, match_id),
        phases=load_phases(matches_dir, match_id, meta),
    )


def iter_matches(matches_dir: str | Path, match_ids: list[int] | None = None,
                 with_tracking: bool = True) -> Iterator[Match]:
    """Yield matches one at a time — never hold 10+ tracking files in RAM."""
    for mid in match_ids or list_match_ids(matches_dir):
        yield load_match(matches_dir, mid, with_tracking=with_tracking)
