"""Grow a small track dataset into a large one.

The uploaded tracks are measured, noise-free examples. Per class (the `class` column) the expander
learns two things from them and draws new tracks that stay inside what those tracks show:

  1. the SHAPE of the class: a new track starts from one of the class's own tracks, begins at a random
     point of its pulse pattern (stagger cycle, ramp, hopping sequence ...) and runs for the wanted
     number of pulses, wrapping around the pattern;
  2. the SPREAD between tracks of the class: when the class has several tracks, every measurement of
     the new track is scaled (SNR: shifted) by a random amount drawn from how much that measurement
     differs between the class's tracks (never beyond +-2 standard deviations). A class with one track
     has no observed spread, so its new tracks differ only in where the pattern starts.

No measurement noise is added. Text columns (modulation type, categorical features) are copied from
the source track. New tracks get new instance names (class + capture date + number); the source tracks
are kept unchanged.
"""
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.sequence_features import SIMULATOR_INFO, order_columns, ordered_tracks

# Columns that identify rather than measure: copied as they are.
_IDENTITY = {"instance", "class", "pulse_idx", "true_class_id"}
_ADDITIVE = {"snr"}  # dB: shifted, every other measurement is scaled


def _measure_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _IDENTITY and c not in SIMULATOR_INFO
            and pd.api.types.is_numeric_dtype(df[c])]


def _spread(tracks: List[pd.DataFrame], cols: List[str]) -> Dict[str, float]:
    """Relative spread (std / |mean|, or the std itself for SNR) of each measurement's per-track
    median across the class's tracks; 0 when there is a single track."""
    out = {c: 0.0 for c in cols}
    if len(tracks) < 2:
        return out
    for c in cols:
        med = np.array([pd.to_numeric(t[c], errors="coerce").median() for t in tracks], dtype=float)
        med = med[np.isfinite(med)]
        if len(med) < 2:
            continue
        if c in _ADDITIVE:
            out[c] = float(med.std())
        elif abs(med.mean()) > 1e-12:
            out[c] = float(med.std() / abs(med.mean()))
    return out


def expand_dataset(df: pd.DataFrame, tracks_per_class: int, pulses_per_track: Optional[int] = None,
                   seed: int = 0, first_day: Optional[date] = None) -> pd.DataFrame:
    """`df` (instance, class, true_class_id + measurements) plus enough new tracks that every class has
    `tracks_per_class` tracks. Classes that already have that many are left alone."""
    rng = np.random.RandomState(seed)
    first_day = first_day or date.today()
    measures = _measure_columns(df)
    new_rows: List[pd.DataFrame] = []
    counter = df["instance"].nunique()
    # classes keyed by label; the unknown classes (true_class_id -1) are grown the same way
    for label, group in df.groupby("class", sort=False):
        tracks = [t.reset_index(drop=True) for _, t in ordered_tracks(group)]
        need = tracks_per_class - len(tracks)
        if need <= 0:
            continue
        spread = _spread(tracks, measures)
        class_id = int(group["true_class_id"].iloc[0]) if "true_class_id" in group.columns else -1
        for _ in range(need):
            src = tracks[int(rng.randint(len(tracks)))]
            n_src = len(src)
            n = int(pulses_per_track or n_src)
            start = int(rng.randint(n_src))
            idx = (start + np.arange(n)) % n_src
            t = src.iloc[idx].reset_index(drop=True).copy()
            if "PRI" in t.columns:
                pri = pd.to_numeric(src["PRI"], errors="coerce")
                filler = pri.median() if pri.notna().any() else np.nan
                t["PRI"] = pri.to_numpy()[idx]
                t.loc[t["PRI"].isna(), "PRI"] = filler  # the source's blank first PRI, now mid-track
                t.loc[0, "PRI"] = np.nan                  # this track's own first pulse has no previous pulse
            for c in measures:
                if spread[c] <= 0:
                    continue
                z = float(np.clip(rng.normal(), -2, 2))
                if c in _ADDITIVE:
                    t[c] = t[c] + z * spread[c]
                else:
                    t[c] = t[c] * (1 + z * spread[c])
            counter += 1
            stamp = (first_day + timedelta(days=int(rng.randint(0, 30)))).strftime("%Y%m%d")
            t["instance"] = f"{label}_{stamp}_{counter:04d}"
            t["class"] = label
            t["true_class_id"] = class_id
            t["pulse_idx"] = np.arange(n)
            for c in ("missed_before", "spurious_before", "is_spurious"):  # ground truth of the source's own pulses
                if c in t.columns:
                    t[c] = 0
            new_rows.append(t)
    if not new_rows:
        return order_columns(df)
    # keep the column dtypes of the source (rounded like the generator's output)
    out = pd.concat([df] + new_rows, ignore_index=True)
    return order_columns(out)
