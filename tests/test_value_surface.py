"""Tests for features.value_surface: vxy (value surface) and savitzky_golay
(gap-aware velocity smoothing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.value_surface import savitzky_golay, vxy


# ---------------------------------------------------------------------------
# vxy
# ---------------------------------------------------------------------------
def _synthetic_alignment():
    """passing_option samples placed exactly at two grid-cell centres (bin_m=10,
    pitch 20x20 -> cells centred at (-5,-5) and (5,5)), plus a player_possession
    row that vxy must ignore."""
    po = pd.DataFrame({
        "event_type": ["passing_option"] * 3,
        "x_start": [-5.0, -5.0, 5.0], "y_start": [-5.0, -5.0, 5.0],
        "x_end": [-5.0, -5.0, 5.0], "y_end": [-5.0, -5.0, 5.0],
        "xthreat": [0.1, 0.3, 0.9],
    })
    pp = pd.DataFrame({
        "event_type": ["player_possession"], "x_start": [0.0], "y_start": [0.0],
        "x_end": [0.0], "y_end": [0.0], "xthreat": [np.nan],
    })
    return pd.concat([po, pp], ignore_index=True)


def test_vxy_grid_ignores_other_event_types_and_averages_per_cell():
    events = _synthetic_alignment()
    v = vxy(events, method="grid", bin_m=10, pitch_length=20, pitch_width=20)
    cell = v.set_index(["x", "y"])
    assert cell.loc[(-5.0, -5.0), "V"] == pytest.approx(0.2)  # mean(0.1, 0.3)
    assert cell.loc[(-5.0, -5.0), "n"] == 2
    assert cell.loc[(5.0, 5.0), "V"] == pytest.approx(0.9)
    assert cell.loc[(5.0, 5.0), "n"] == 1
    assert np.isnan(cell.loc[(-5.0, 5.0), "V"])  # empty cell
    assert cell.loc[(-5.0, 5.0), "n"] == 0


def test_vxy_kde_has_no_nan_and_is_bounded_by_samples():
    events = _synthetic_alignment()
    v = vxy(events, method="kde", bin_m=10, pitch_length=20, pitch_width=20, bandwidth=5.0)
    assert v.V.notna().all()
    assert v.V.between(0.1, 0.9).all()


def test_vxy_rejects_unknown_method_and_loc():
    events = _synthetic_alignment()
    with pytest.raises(ValueError):
        vxy(events, pitch_length=20, pitch_width=20, method="nonsense")
    with pytest.raises(ValueError):
        vxy(events, pitch_length=20, pitch_width=20, loc="middle")


def test_vxy_loc_start_vs_end_select_different_columns():
    events = _synthetic_alignment().copy()
    events.loc[events.event_type == "passing_option", ["x_start", "y_start"]] = [[-5.0, 5.0]] * 3
    # x_end/y_end stay at the original locations
    v_start = vxy(events, method="grid", loc="start", bin_m=10, pitch_length=20, pitch_width=20)
    v_end = vxy(events, method="grid", loc="end", bin_m=10, pitch_length=20, pitch_width=20)
    assert not v_start.set_index(["x", "y"]).V.equals(v_end.set_index(["x", "y"]).V)


def test_vxy_spline_runs_on_enough_scattered_points():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(-50, 50, n)
    y = rng.uniform(-30, 30, n)
    v = 0.001 * x + 0.002 * y  # smooth deterministic surface
    events = pd.DataFrame({
        "event_type": ["passing_option"] * n,
        "x_start": x, "y_start": y, "x_end": x, "y_end": y, "xthreat": v,
    })
    out = vxy(events, method="spline", bin_m=10, pitch_length=100, pitch_width=60)
    assert out.V.notna().all()
    assert len(out) == len(out.x.unique()) * len(out.y.unique())


# ---------------------------------------------------------------------------
# savitzky_golay
# ---------------------------------------------------------------------------
def _constant_velocity_tracking(n=20, vx=2.0, vy=-1.0, fps=10.0, player_id=1):
    dt = 1.0 / fps
    t = np.arange(n) * dt
    return pd.DataFrame({
        "frame": np.arange(n), "player_id": player_id,
        "x": vx * t, "y": vy * t, "is_detected": True,
    })


def test_savitzky_golay_recovers_constant_velocity():
    tr = _constant_velocity_tracking(vx=2.0, vy=-1.0)
    out = savitzky_golay(tr, fps=10.0, window_length=9, polyorder=2)
    assert np.allclose(out.vx, 2.0, atol=1e-8)
    assert np.allclose(out.vy, -1.0, atol=1e-8)
    assert np.allclose(out.speed, np.hypot(2.0, -1.0), atol=1e-8)


def test_savitzky_golay_undetected_gap_splits_runs_and_is_nan():
    tr = _constant_velocity_tracking(n=30)
    tr.loc[tr.frame.between(10, 14), "is_detected"] = False
    out = savitzky_golay(tr, fps=10.0, window_length=9, polyorder=2)
    gap = out[out.frame.between(10, 14)]
    assert gap.vx.isna().all() and gap.vy.isna().all()
    run1 = out[out.frame.between(0, 9)]
    run2 = out[out.frame.between(15, 29)]
    assert run1.vx.notna().all() and run2.vx.notna().all()
    assert np.allclose(run1.vx, 2.0, atol=1e-8)
    assert np.allclose(run2.vx, 2.0, atol=1e-8)


def test_savitzky_golay_frame_number_gap_splits_runs():
    tr = _constant_velocity_tracking(n=20)
    tr.loc[tr.frame >= 10, "frame"] += 100  # non-contiguous frame numbers, all still "detected"
    out = savitzky_golay(tr, fps=10.0, window_length=9, polyorder=2)
    # both sides are long enough runs on their own -> no NaNs introduced by the jump
    assert out.vx.notna().all()
    assert np.allclose(out.vx, 2.0, atol=1e-8)


def test_savitzky_golay_short_run_stays_nan():
    tr = _constant_velocity_tracking(n=20)
    tr.loc[tr.frame.between(5, 8), "is_detected"] = False  # splits into runs of 5 and 11
    out = savitzky_golay(tr, fps=10.0, window_length=9, polyorder=2)
    short_run = out[out.frame.between(0, 4)]
    long_run = out[out.frame.between(9, 19)]
    assert short_run.vx.isna().all()  # len 5 < window_length 9
    assert long_run.vx.notna().all()  # len 11 >= 9


def test_savitzky_golay_rejects_even_window():
    tr = _constant_velocity_tracking(n=10)
    with pytest.raises(ValueError):
        savitzky_golay(tr, window_length=8)


def test_savitzky_golay_mislabelled_teleport_is_treated_as_a_gap():
    """A same-frame-marked-detected teleport (broadcast camera-cut artefact,
    not a real gap) must not get smoothed into a velocity spike when
    max_speed_mps is given."""
    tr = _constant_velocity_tracking(n=20, vx=2.0, vy=0.0)
    tr.loc[tr.frame == 10, "x"] += 100.0  # ~1000 m/s implied jump, still "detected"
    out = savitzky_golay(tr, fps=10.0, window_length=9, polyorder=2, max_speed_mps=12.0)
    assert out.speed.max() < 12.0 + 1e-6
    without_guard = savitzky_golay(tr, fps=10.0, window_length=9, polyorder=2, max_speed_mps=None)
    assert without_guard.speed.max() > 50.0  # confirms the teleport does spike speed without the guard
