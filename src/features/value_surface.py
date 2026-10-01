from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..data_io.load import Match


def savitzky_golay(tracking: pd.DataFrame, fps: float = 10.0, window_length: int = 9,
                    polyorder: int = 2) -> pd.DataFrame:
    """Smoothed velocity (vx, vy, speed) per (match_id, player_id, frame).

    SkillCorner's tracking file is already gap-filled by extrapolation
    (``is_detected`` marks which frames are real detections vs. filled-in), and
    extrapolated stretches are not true kinematics — smoothing or
    differentiating through them would manufacture fake velocity. So the
    Savitzky-Golay filter is applied only within contiguous runs of
    ``is_detected == True`` frames; a gap (an undetected frame, or a missing
    frame number) ends the current run. Runs shorter than ``window_length``
    are left NaN — too few points for a stable 2nd-order fit.
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


def vxy(events: pd.DataFrame, method: str = "grid", *, loc: str = "end",
        bin_m: float = 5.0, pitch_length: float = 105.0, pitch_width: float = 68.0,
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
    else:
        raise ValueError(f"unknown method {method!r}: use 'grid', 'kde' or 'spline'")

    return pd.DataFrame({
        "x": XC.ravel(), "y": YC.ravel(), "V": np.asarray(V).ravel(),
        "n": n.ravel(), "method": method,
    })
