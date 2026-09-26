"""Loop the Week-1 audit over every match. Holds only one tracking file in RAM.

    python -m src.audit.run_audit            # all matches in config
    python -m src.audit.run_audit 1886347    # subset
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import pandas as pd

from ..config import REPO_ROOT, load_config
from ..data_io.load import iter_matches, list_match_ids
from ..data_io.transform import build_normalised
from . import checks, occlusion


@dataclass
class AuditResult:
    frames: pd.DataFrame        # frame-level quality flags, all matches
    phases: pd.DataFrame        # eligible phases with survival verdict
    heat_def: pd.DataFrame      # detection counts by location (defenders)
    roles: pd.DataFrame         # detection counts by role
    strings: pd.DataFrame       # §2.3 reconciliation, per match
    alignment: pd.DataFrame     # §3.3 event↔tracking check, per match

    def summary(self, cfg: dict) -> pd.DataFrame:
        fr = self.frames
        inplay = fr[fr.in_phase]
        s = {
            "matches": fr.match_id.nunique(),
            "in-phase frames": len(inplay),
            "possession disagreement (in-phase frames)": 1 - inplay.f_possession_consistent.mean(),
            "frames ≥ detection floor": inplay.f_detection.mean(),
            "frames with danger zone visible": inplay.f_visible.mean(),
            "frames passing all filters": inplay.passes_all.mean(),
            "mean detected outfield defenders": inplay.n_det_outfield_def.mean(),
            "eligible final-third defensive phases": len(self.phases),
            "phases surviving": int(self.phases.survives.sum()),
            "phase survival rate": self.phases.survives.mean(),
            "GATE 1 threshold": cfg["gate1"]["min_phase_survival"],
        }
        s["GATE 1"] = "PASS" if s["phase survival rate"] >= s["GATE 1 threshold"] else "FAIL"
        return pd.Series(s, name="value").to_frame()


def run(match_ids: list[int] | None = None, cfg: dict | None = None, verbose: bool = True) -> AuditResult:
    cfg = cfg or load_config()
    mdir = cfg["data"]["matches_dir"]
    match_ids = match_ids or list_match_ids(mdir)
    bin_m = cfg["audit"]["heatmap_bin_m"]
    F, P, H, R, S, A = [], [], [], [], [], []
    for m in iter_matches(mdir, match_ids):
        t0 = time.time()
        frames, tracking = build_normalised(m)
        fq = occlusion.frame_quality(frames, tracking, m, cfg)
        ph = occlusion.phase_survival(fq, m.phases, cfg)
        ph["match_id"] = m.match_id

        # Heatmap over frames of eligible phases only (where VSD will be computed)
        elig_idx = set(ph.phase_index)
        mask = pd.Series(fq.phase_index.isin(elig_idx).to_numpy(), index=fq.frame.to_numpy())
        h = occlusion.detection_by_location(tracking, mask, m.pitch_length, m.pitch_width, bin_m)
        r = occlusion.detection_by_role(tracking, mask)
        h["match_id"] = r["match_id"] = m.match_id

        s = checks.reconcile_strings(m.events, m.phases)
        s["match_id"] = m.match_id
        a = checks.event_tracking_alignment(m, frames, tracking)

        F.append(fq); P.append(ph); H.append(h); R.append(r); S.append(s); A.append(a)
        if verbose:
            print(f"{m.match_id}: {len(ph)} eligible phases, {ph.survives.mean():.0%} survive "
                  f"({time.time() - t0:.1f}s)")
        del m, frames, tracking
    return AuditResult(pd.concat(F, ignore_index=True), pd.concat(P, ignore_index=True),
                       pd.concat(H, ignore_index=True), pd.concat(R, ignore_index=True),
                       pd.concat(S, ignore_index=True), pd.DataFrame(A))


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or None
    cfg = load_config()
    res = run(ids, cfg)
    out = REPO_ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.phases.to_csv(out / "01_phase_survival.csv", index=False)
    res.alignment.to_csv(out / "01_event_alignment.csv", index=False)
    print(res.summary(cfg).to_string())
