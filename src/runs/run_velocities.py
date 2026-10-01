from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import pandas as pd
import numpy as np

from ..config import REPO_ROOT, load_config
from ..data_io.load import iter_matches, list_match_ids
from ..data_io.transform import build_normalised
from ..features.value_surface import savitzky_golay, vxy


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
    cfg = cfg or load_config()
    mdir = cfg["data"]["matches_dir"]
    match_ids = match_ids or list_match_ids(mdir)
    SG, SK, SS, SA = [], [], [], []
    for m in iter_matches(mdir, match_ids):
        t0 = time.time()
        frames, tracking = build_normalised(m)
        vcfg = cfg["velocities"]
        sg = savitzky_golay(tracking, fps=cfg["data"]["fps"], window_length=vcfg["window_length"],
                            polyorder=vcfg["polyorder"], max_speed_mps=vcfg["max_speed_mps"])
        tracking = tracking.merge(sg[["frame", "player_id", "vx", "vy", "speed"]], on=["frame", "player_id"])

        pitch_length = m.meta["pitch_length"]
        pitch_width = m.meta["pitch_width"]
        surfaces = {
            met: vxy(m.events, pitch_length=pitch_length, pitch_width=pitch_width, bin_m=cfg["audit"]["heatmap_bin_m"])
            for met in ("grid", "kde", "spline")}
        SA.append(tracking)
        SG.append(surfaces["grid"])
        SK.append(surfaces["kde"])
        SS.append(surfaces["spline"])
        if verbose:
            print(f"Match ID: {m.match_id}"
                  f" | Mean speed : {tracking.speed.mean():.2f} m/s"
                  f" | Minimum speed: {tracking.speed.min():.2f} m/s"
                  f" | Maximum speed: {tracking.speed.max():.2f} m/s"
                  f" | Counting: {tracking.speed.count()}"
                  f" | Time: {time.time() - t0:.1f}s")

            for met, v in surfaces.items():
                print(met, "V range:", v.V.min(), v.V.max(), "| sampled cells:", (v.n > 0).sum(), "/", len(v))
        del m, frames, tracking
    return VelocitiesResult(pd.concat(SG, ignore_index=True), pd.concat(SK, ignore_index=True),
                            pd.concat(SS, ignore_index=True), pd.concat(SA, ignore_index=True))
