from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..data_io.load import Match


def savitzky_golay(tracking: pd.DataFrame, fps: float = 10.0, window_length: int = 9,
                   polyorder: int = 2, max_speed_mps: float | None = None) -> pd.DataFrame:
    """Smoothed velocity (vx, vy, speed) per (match_id, player_id, frame).

    SkillCorner's tracking file is already gap-filled by extrapolation
    (``is_detected`` marks which frames are real detections vs. filled-in), and
    extrapolated stretches are not true kinematics — smoothing or
    differentiating through them would manufacture fake velocity. So the
    Savitzky-Golay filter is applied only within contiguous runs of
    ``is_detected == True`` frames; a gap (an undetected frame, or a missing
    frame number) ends the current run. Runs shorter than ``window_length``
    are left NaN — too few points for a stable 2nd-order fit.

    ``is_detected`` isn't a perfect gap flag though: broadcast tracking
    occasionally mis-projects a position for a frame or two (e.g. on a camera
    cut) while still marking it detected — this shows up as a ~100 m,
    one-frame teleport, often hitting several players at the same frame. If
    ``max_speed_mps`` is given, a detected-to-detected step implying a higher
    speed than that is treated as a run break too, same as a real gap,
    instead of being smoothed into a velocity spike.
    """
    group_cols = [c for c in ("match_id", "player_id") if c in tracking.columns]
    dt = 1.0 / fps
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")

    out = []
    for _, g in tracking.sort_values(group_cols + ["frame"]).groupby(group_cols, sort=False):
        g = g.reset_index(drop=True)
        detected = g.is_detected.to_numpy()
        gap = g.frame.diff().to_numpy() != 1  # first row's NaN diff -> False, not a gap
        gap[0] = False
        if max_speed_mps is not None:
            frame_diff = g.frame.diff().to_numpy()
            step_disp = np.hypot(g.x.diff().to_numpy(), g.y.diff().to_numpy())
            with np.errstate(invalid="ignore", divide="ignore"):
                implied_speed = step_disp / (frame_diff * dt)
            jump = np.where(frame_diff > 0, implied_speed > max_speed_mps, False)
            gap = gap | jump
        run_id = (~detected | gap).cumsum()

        vx = np.full(len(g), np.nan)
        vy = np.full(len(g), np.nan)
        for rid in np.unique(run_id[detected]):
            idx = np.flatnonzero(detected & (run_id == rid))
            if len(idx) < window_length:
                continue
            vx[idx] = savgol_filter(g.x.to_numpy()[idx], window_length, polyorder, deriv=1, delta=dt)
            vy[idx] = savgol_filter(g.y.to_numpy()[idx], window_length, polyorder, deriv=1, delta=dt)
        g["vx"], g["vy"] = vx, vy
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    res["speed"] = np.hypot(res.vx, res.vy)
    return res


def normalised_velocities(match: Match, frames_norm: pd.DataFrame, tracking_norm: pd.DataFrame,
                          fps: float = 10.0, window_length: int = 9, polyorder: int = 2,
                          max_speed_mps: float | None = None) -> pd.DataFrame:
    """tracking_norm plus vx, vy, speed in the NORMALISED frame.

    Differentiate in RAW coordinates: the possession flip rotates the pitch
    180° at every turnover, so in normalised coords each player jumps from
    (x, y) to (-x, -y). The flip is one rotation per frame, so the velocity
    rotates the same way: v_norm = sign * v_raw.
    """
    raw = match.tracking[["frame", "player_id", "x", "y", "is_detected"]]
    v = savitzky_golay(raw, fps=fps, window_length=window_length, polyorder=polyorder,
                       max_speed_mps=max_speed_mps)[["frame", "player_id", "vx", "vy"]]
    t = (tracking_norm.drop(columns=["vx", "vy", "speed"], errors="ignore")
         .merge(v, on=["frame", "player_id"], how="left")
         .merge(frames_norm[["frame", "sign"]], on="frame", how="left"))
    t["vx"] = t.vx * t.sign
    t["vy"] = t.vy * t.sign
    t["speed"] = np.hypot(t.vx, t.vy)
    return t.drop(columns=["sign"])


def vxy(events: pd.DataFrame, pitch_length: float, pitch_width: float,
        method: str = "grid", *, loc: str = "end", bin_m: float = 5.0,
        bandwidth: float = 3.0) -> pd.DataFrame:
    """Value surface V(x, y): xThreat of ``passing_option`` events, mapped onto
    the pitch at the *option's own* location (not the passer's).

    ``events`` is the alignment table from
    ``audit.checks.event_tracking_alignment`` (or any frame with the same
    passing_option columns: x_start/y_start, x_end/y_end, xthreat). Three
    interchangeable ways to go from scattered (x, y, xthreat) samples to a
    surface, picked via ``method``:

    * "grid"   — pitch binned into ``bin_m`` x ``bin_m`` cells, V = mean
                 xthreat per cell. Same binning convention as
                 ``audit.occlusion.detection_by_location``. Cheap, but blocky
                 and noisy in sparsely-sampled cells (see ``n``).
    * "kde"    — Nadaraya-Watson: Gaussian-kernel-weighted mean xthreat,
                 evaluated at the same grid centres. Smoother than "grid",
                 adds a ``bandwidth`` (m) hyperparameter.
    * "spline" — ``scipy.interpolate.SmoothBivariateSpline`` fit through the
                 scattered samples, evaluated on the same grid. Continuous
                 and differentiable, but can ring/overshoot where samples are
                 sparse (e.g. near the touchlines).

    ``loc``: "end" (default) uses the option's position at frame_end — where
    the ball would actually arrive — or "start", its position when it first
    became an option.

    Returns a long-form DataFrame: x, y (cell centres), V, n (sample count per
    cell, independent of ``method`` — use it to see where V is well-sampled),
    method.
    """
    if loc not in ("start", "end"):
        raise ValueError("loc must be 'start' or 'end'")
    x_col, y_col = f"x_{loc}", f"y_{loc}"

    po = events[events.event_type == "passing_option"]
    po = po.dropna(subset=[x_col, y_col, "xthreat"])
    x = po[x_col].to_numpy()
    y = po[y_col].to_numpy()
    v = po["xthreat"].to_numpy()

    x_edges = np.arange(-pitch_length / 2, pitch_length / 2 + bin_m, bin_m)
    y_edges = np.arange(-pitch_width / 2, pitch_width / 2 + bin_m, bin_m)
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    XC, YC = np.meshgrid(xc, yc, indexing="ij")

    n, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if method == "grid":
        sum_v, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=v)
        V = np.divide(sum_v, n, out=np.full_like(sum_v, np.nan), where=n > 0)
    elif method == "kde":
        d2 = (XC.ravel()[:, None] - x[None, :]) ** 2 + (YC.ravel()[:, None] - y[None, :]) ** 2
        w = np.exp(-d2 / (2 * bandwidth ** 2))
        den = w.sum(axis=1)
        num = w @ v
        V = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0).reshape(XC.shape)
    elif method == "spline":
        from scipy.interpolate import SmoothBivariateSpline
        spline = SmoothBivariateSpline(x, y, v, kx=3, ky=3)
        V = spline(xc, yc)
        V = V.clip(0, 1)
    else:
        raise ValueError(f"unknown method {method!r}: use 'grid', 'kde' or 'spline'")

    return pd.DataFrame({
        "x": XC.ravel(), "y": YC.ravel(), "V": np.asarray(V).ravel(),
        "n": n.ravel(), "method": method,
    })
