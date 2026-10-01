from __future__ import annotations

import time
from dataclasses import dataclass
import pandas as pd

from ..config import REPO_ROOT, load_config
from ..data_io.load import iter_matches, list_match_ids
from ..data_io.transform import build_normalised
from ..features.value_surface import normalised_velocities, vxy


@dataclass
class VelocitiesResult:
    surfaces_grid: pd.DataFrame
    surfaces_kde: pd.DataFrame
    surfaces_spline: pd.DataFrame
    savitzky: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        pass


def run(match_ids: list[int] | None = None, cfg: dict | None = None,
        verbose: bool = True) -> VelocitiesResult:
    """V(x, y) is pooled across every match into ONE surface, not built per
    match and concatenated: a match-by-match grid would put e.g. match A's
    (50, 0) cell and match B's (50, 0) cell at different physical distances
    from goal whenever their pitch dimensions differ (they do — SkillCorner
    pitches here range 104-106m long), silently fragmenting ~2.4k
    passing-option samples per match instead of pooling ~48k. So events (tiny
    vs. tracking) are accumulated across the whole loop and ``vxy`` is called
    once, on one shared grid sized to the largest pitch in the set.
    """
    cfg = cfg or load_config()
    mdir = cfg["data"]["matches_dir"]
    match_ids = match_ids or list_match_ids(mdir)
    vcfg = cfg["velocities"]
    SA, EV = [], []
    pitch_length = pitch_width = 0.0
    for m in iter_matches(mdir, match_ids):
        t0 = time.time()
        frames, tracking = build_normalised(m)
        tracking = normalised_velocities(m, frames, tracking, fps=cfg["data"]["fps"],
                                         window_length=vcfg["window_length"],
                                         polyorder=vcfg["polyorder"],
                                         max_speed_mps=vcfg["max_speed_mps"])
        SA.append(tracking)
        EV.append(m.events)
        pitch_length = max(pitch_length, m.meta["pitch_length"])
        pitch_width = max(pitch_width, m.meta["pitch_width"])
        if verbose:
            print(f"Match ID: {m.match_id}"
                  f" | Mean speed : {tracking.speed.mean():.2f} m/s"
                  f" | Minimum speed: {tracking.speed.min():.2f} m/s"
                  f" | Maximum speed: {tracking.speed.max():.2f} m/s"
                  f" | Counting: {tracking.speed.count()}"
                  f" | Time: {time.time() - t0:.1f}s")
        del m, frames, tracking

    events = pd.concat(EV, ignore_index=True)
    bin_m = cfg["audit"]["heatmap_bin_m"]
    surfaces = {met: vxy(events, pitch_length=pitch_length, pitch_width=pitch_width, method=met, bin_m=bin_m)
                for met in ("grid", "kde", "spline")}
    if verbose:
        n_po = (events.event_type == "passing_option").sum()
        print(f"Pooled V(x, y): {n_po} passing_option samples over {len(match_ids)} matches, "
              f"shared grid {pitch_length}x{pitch_width}m, bin {bin_m}m")
        for met, v in surfaces.items():
            print(met, "V range:", f"[{v.V.min():.2f},", f"{v.V.max():.2f}]", "| sampled cells:",
                  (v.n > 0).sum(), "/", len(v))

    return VelocitiesResult(surfaces["grid"], surfaces["kde"], surfaces["spline"],
                            pd.concat(SA, ignore_index=True))
