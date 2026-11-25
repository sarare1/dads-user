"""Pulse-train (sequence) dataset: ordered pulses of ONE emitter per track, as delivered by an
upstream de-interleaver — the input of the sequence model (src/models/sequence_models.py).

The emitter families come from build_emitter_library (seed and ranges). What a track adds is
TIME ORDER:

  PRI      follows its family's modulation pattern pulse by pulse (stagger cycle, sliding ramp,
           wobulation sine, dwell-and-switch blocks, jitter, constant)
  freq     follows its family's agility pattern (fixed, pulse-to-pulse hopping, stepped cycle,
           burst-to-burst hopping)
  mode     persists for a dwell of 15-60 pulses, then transitions (search -> ... -> illumination)

and the imperfections of real de-interleaver output:
  missed pulses     a pulse is not intercepted -> the next PRI is the sum of two intervals
  spurious pulses   a pulse of another emitter wrongly assigned to the track -> splits an interval
  first pulse       has no PRI (there is no previous pulse) -> pri_us left blank

Columns (in file order): Frequency (MHz), PW (µs), PRI (µs), Modlntype, Modlnval, snr (dB),
rise (ns), fall (ns), timeBW (Hz), any user-defined extra columns, instance (the track: class name
+ capture date + number), class (the emitter name), then pulse_idx and true_class_id (bookkeeping,
not model inputs) and the simulator's ground truth (pri_pattern, freq_agility, operating_mode,
is_spurious, pri_mode_base_us, pri_deviation_pct, missed_before, spurious_before - never model
inputs). No time of arrival: the model needs only the PRI sequence.
"""
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.generator import aux_rng_for, build_emitter_library, noise_pw_pri, noisy_pulse_width
from src.data.operating_modes import OPERATING_MODES, sample_dwell, transition_mode
from src.data.sequence_features import MODULATION_TYPES
from src.data.pri_patterns import PGRI, PRI_PATTERN_TYPES, pgri_interval

FREQ_AGILITY_TYPES = ["fixed", "hopping", "stepped", "burst"]
SEQUENCE_COLUMNS = ["Frequency", "PW", "PRI", "Modlntype", "Modlnval", "snr", "rise", "fall", "timeBW",
                    "instance", "class", "pulse_idx", "true_class_id"]
MHZ = 1e6

# Intra-pulse modulation: Modlntype is the family's modulation type, Modlnval its parameter -
# the chirp rate (MHz/µs) of a (non-)linear FM pulse, the code length of a phase-coded one, 0 for none.
CODE_LENGTHS = {"Barker": [3, 4, 5, 7, 11, 13], "BPSK": [16, 32, 64, 128], "Frank": [16, 25, 36, 64]}
FM_TYPES = ("LFM", "NLFM")
EXTRA_KINDS = ("numeric", "categorical")


def balance_pri_patterns(library: Dict[int, Dict[str, Any]], holdout_class_ids: List[int], seed: int) -> None:
    """Hand the PRI types out round-robin (known families first, then held-out ones) instead of drawing
    each family's type at random — with 10 families a random draw easily misses a type altogether
    (no staggered family at all). PGRI families keep theirs. The order of the types is seeded."""
    order = [PRI_PATTERN_TYPES[i] for i in aux_rng_for(seed, "pattern_order").permutation(len(PRI_PATTERN_TYPES))]
    known = [c for c in sorted(library) if c not in holdout_class_ids]
    held = [c for c in sorted(library) if c in holdout_class_ids]
    n = 0
    for c in known + held:
        if library[c]["pri_pattern"] == PGRI:
            continue
        library[c]["pri_pattern"] = order[n % len(order)]
        n += 1


def build_sequence_library(num_classes: int, seed: int, ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                           include_pgri: bool = False,
                           balanced_holdout: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
    """Family archetypes (identical to the classification dataset's for this seed/ranges) plus,
    per family, the parameters of its PRI pattern and frequency agility over time. The extra
    parameters come from their own random stream, so the base archetypes never change."""
    library = build_emitter_library(num_classes, seed=seed, include_pgri=include_pgri, **(ranges or {}))
    if balanced_holdout is not None:  # None = the original random assignment (datasets made before balancing)
        balance_pri_patterns(library, balanced_holdout, seed)
    rng = aux_rng_for(seed, "sequence_patterns")
    for c in range(num_classes):
        lib = library[c]
        lib["seq"] = {
            "jitter_frac": float(rng.uniform(0.01, 0.05)),
            "stagger_levels": (1 + rng.uniform(-0.3, 0.3, size=int(rng.randint(2, 6)))).tolist(),
            "slide_period": int(rng.randint(16, 65)), "slide_span": float(rng.uniform(0.1, 0.3)),
            "wobble_period": int(rng.randint(12, 49)), "wobble_span": float(rng.uniform(0.1, 0.3)),
            "dwell_levels": (1 + rng.uniform(-0.3, 0.3, size=int(rng.randint(2, 5)))).tolist(),
            "dwell_block": int(rng.randint(16, 49)),
            "freq_agility": str(rng.choice(FREQ_AGILITY_TYPES)),
            "hop_span": float(rng.uniform(0.02, 0.08)),          # ± fraction of the carrier
            "step_size": float(rng.uniform(0.003, 0.01)),        # fraction of the carrier per step
            "step_count": int(rng.randint(3, 6)),
            "burst_len": int(rng.randint(8, 33)),
        }
    return library


def family_modulation(num_classes: int, seed: int) -> List[Tuple[str, float]]:
    """(Modlntype, code length / curvature) per family, from their own random stream so the base
    archetypes and patterns of a seed never change. Types are handed out round-robin from a seeded
    order so every type occurs once there are enough families."""
    rng = aux_rng_for(seed, "modulation")
    order = [MODULATION_TYPES[i] for i in rng.permutation(len(MODULATION_TYPES))]
    out = []
    for c in range(num_classes):
        t = order[c % len(order)]
        out.append((t, float(rng.choice(CODE_LENGTHS[t])) if t in CODE_LENGTHS else (float(rng.uniform(0.8, 1.2)) if t == "NLFM" else 0.0)))
    return out


def add_side_columns(pulses: List[Dict[str, Any]], modulation: Optional[Tuple[str, float]], rng: np.random.RandomState) -> None:
    """Adds the intra-pulse modulation (mod_type, mod_val) and the signal level (snr, dB) to a track's
    pulses, in place. `modulation` is the family's (type, parameter); None = an unstructured noise
    track (any type). A track has one capture level (range, aspect ...) with pulse-to-pulse scatter;
    a spurious pulse belongs to another emitter, so it gets a random modulation."""
    if modulation is None:
        mtype = str(rng.choice(MODULATION_TYPES))
        mval = float(rng.choice(CODE_LENGTHS[mtype])) if mtype in CODE_LENGTHS else 0.0
    else:
        mtype, mval = modulation
    level = float(rng.uniform(8.0, 35.0))
    for pl in pulses:
        if pl.get("spurious"):
            pl["mod_type"], pl["mod_val"] = str(rng.choice(MODULATION_TYPES)), float(rng.uniform(0, 64))
        else:
            pl["mod_type"] = mtype
            # chirp rate = swept bandwidth (MHz) / pulse width (us)
            pl["mod_val"] = pl["bw"] / max(pl["pw"], 1e-6) * (mval if mtype == "NLFM" else 1.0) if mtype in FM_TYPES else mval
        pl["snr"] = level + float(rng.normal(0, 1.0))


def validate_extra_features(extras: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """None when the user-defined extra columns are usable, else the reason."""
    seen = set(SEQUENCE_COLUMNS)
    for e in extras or []:
        name = str(e.get("name", "")).strip()
        if not name or len(name) > 40 or not all(ch.isalnum() or ch in "_-." for ch in name):
            return f"Extra feature name '{name}' must be 1-40 letters, digits, '_', '-' or '.'."
        if name in seen:
            return f"Extra feature name '{name}' is already used."
        seen.add(name)
        if e.get("kind") not in EXTRA_KINDS:
            return f"Extra feature '{name}': kind must be numeric or categorical."
        if e["kind"] == "numeric":
            lo, hi = e.get("min"), e.get("max")
            if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                return f"Extra feature '{name}': give a finite min <= max."
            if not 0 <= (e.get("noise_pct") or 0) <= 100:
                return f"Extra feature '{name}': noise must be 0-100 %."
        else:
            vals = [str(v).strip() for v in (e.get("values") or []) if str(v).strip()]
            if len(set(vals)) < 2 or len(set(vals)) > 50:
                return f"Extra feature '{name}': give 2-50 distinct category values."
    return None


def _extra_bases(extras: List[Dict[str, Any]], num_classes: int, seed: int) -> Dict[str, List[Any]]:
    """Each family's own value of every extra feature (numeric: drawn in [min, max]; categorical:
    handed out round-robin over the values), from a random stream per feature name."""
    bases: Dict[str, List[Any]] = {}
    for e in extras:
        rng = aux_rng_for(seed, "extra:" + e["name"])
        if e["kind"] == "numeric":
            bases[e["name"]] = rng.uniform(e["min"], e["max"], size=num_classes).tolist()
        else:
            vals = [str(v).strip() for v in e["values"] if str(v).strip()]
            order = [vals[i] for i in rng.permutation(len(vals))]
            bases[e["name"]] = [order[c % len(order)] for c in range(num_classes)]
    return bases


def extra_pulse_value(e: Dict[str, Any], base: Any, rng: np.random.RandomState) -> Any:
    """One pulse's value of a user-defined extra feature. `base` is the family's own value; None = a pulse
    that belongs to no family (spurious / noise track): a random value within the feature's range."""
    if e["kind"] == "numeric":
        if base is None:
            return round(float(rng.uniform(e["min"], e["max"])), 6)
        return round(base + rng.normal(0, (e.get("noise_pct") or 0) / 100.0 * max(abs(base), 1e-9)), 6)
    if base is None:
        return str(rng.choice([str(v).strip() for v in e["values"] if str(v).strip()]))
    return base


def add_extra_columns(pulses: List[Dict[str, Any]], extras: List[Dict[str, Any]], num_classes: int, seed: int,
                      family: Optional[int], rng: np.random.RandomState) -> None:
    """Sets the user-defined extra features on a track's pulses (key = feature name), in place - for the
    Live Simulation and the Inference example track, which don't go through build_sequence_dataset_rows."""
    bases = _extra_bases(extras, num_classes, seed)
    for pl in pulses:
        for e in extras:
            pl[e["name"]] = extra_pulse_value(e, None if family is None or pl.get("spurious") else bases[e["name"]][family], rng)


def _pri_at(pattern: str, p: Dict[str, Any], i: int, base: float, rng: np.random.RandomState) -> float:
    """PRI of pulse i (in track order) for a family's modulation pattern, before measurement noise."""
    if pattern == "constant":
        return base
    if pattern == "jittered":
        return base * (1 + rng.normal(0, p["jitter_frac"]))
    if pattern == "staggered":
        levels = p["stagger_levels"]
        return base * levels[i % len(levels)]
    if pattern == "sliding":  # triangular ramp between base*(1-span) and base*(1+span)
        phase = (i % p["slide_period"]) / p["slide_period"]
        tri = 2 * phase if phase < 0.5 else 2 * (1 - phase)
        return base * (1 - p["slide_span"] + 2 * p["slide_span"] * tri)
    if pattern == "wobulated":
        return base * (1 + p["wobble_span"] * np.sin(2 * np.pi * i / p["wobble_period"]))
    if pattern == "dwell_switch":
        levels = p["dwell_levels"]
        return base * levels[(i // p["dwell_block"]) % len(levels)]
    if pattern == PGRI and "pgri" in p:
        return pgri_interval(i % p["pgri"]["group_size"], base, p["pgri"])
    return base


def _freq_at(p: Dict[str, Any], i: int, base_mhz: float, rng: np.random.RandomState) -> float:
    agility = p["freq_agility"]
    if agility == "hopping":
        return base_mhz * (1 + rng.uniform(-p["hop_span"], p["hop_span"]))
    if agility == "stepped":
        return base_mhz * (1 + p["step_size"] * (i % p["step_count"]))
    if agility == "burst":  # a new random channel every burst_len pulses, fixed within a burst
        burst_rng = np.random.RandomState((int(base_mhz * 1000) * 1_000_003 + i // p["burst_len"]) % (2 ** 32))
        return base_mhz * (1 + burst_rng.uniform(-p["hop_span"], p["hop_span"]))
    return base_mhz


def generate_track(params: Dict[str, Any], n_pulses: int, rng: np.random.RandomState,
                   missed_pct: float = 0.0, spurious_pct: float = 0.0,
                   ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict[str, Any]]:
    """One emitter's pulse train as de-interleaver output (see module docstring)."""
    p = dict(params["seq"], pgri=params["pgri"]) if "pgri" in params else params["seq"]
    ranges = ranges or {}
    offset = int(rng.randint(0, 1000))  # random phase in the PRI / agility pattern
    mode_idx = int(rng.randint(len(OPERATING_MODES)))
    dwell_left = sample_dwell(rng)
    # Transmit pulses until n_pulses have been intercepted. Missed ones are dropped here and their
    # interval added to the next intercepted pulse, so the track always has exactly n_pulses.
    kept, carry, j, missed = [], 0.0, 0, 0
    while len(kept) < n_pulses:
        if dwell_left <= 0:
            mode_idx = transition_mode(rng, mode_idx)
            dwell_left = sample_dwell(rng)
        dwell_left -= 1
        mode = OPERATING_MODES[mode_idx]
        i = j + offset
        base_pri = params["pri"] * mode["pri_mult"]
        pulse = {
            "interval": max(0.5, _pri_at(params["pri_pattern"], p, i, base_pri, rng) + rng.normal(0, 0.002 * base_pri)),
            "freq_mhz": _freq_at(p, i, params["freq"] * mode["freq_mult"], rng) + rng.normal(0, 1.0),
            "pw": noisy_pulse_width(params["pw"] * mode["pw_mult"], rng),
            "rise": max(0.5, params["rise"] + rng.normal(0, 1.0)),
            "fall": max(0.5, params["fall"] + rng.normal(0, 1.5)),
            "bw": max(0.05, params["bw"] + rng.normal(0, 0.5)),
            "mode": mode["name"], "spurious": False, "base": base_pri, "missed_before": 0,
        }
        j += 1
        if kept and rng.random() < missed_pct:  # missed: not intercepted
            carry += pulse["interval"]
            missed += 1
            continue
        pulse["interval"] += carry
        pulse["missed_before"] = missed
        carry, missed = 0.0, 0
        kept.append(pulse)

    # Spurious pulses: another emitter's pulse inserted part-way through an interval.
    out = []
    for k, pulse in enumerate(kept):
        if k > 0 and rng.random() < spurious_pct:
            split = rng.uniform(0.1, 0.9)
            f_lo, f_hi = ranges.get("freq_range", (2000.0, 18000.0))
            out.append({
                "interval": pulse["interval"] * split, "freq_mhz": rng.uniform(f_lo, f_hi),
                "pw": rng.uniform(*ranges.get("pw_range", (0.5, 50.0))), "rise": rng.uniform(*ranges.get("rise_range", (5.0, 150.0))),
                "fall": rng.uniform(*ranges.get("fall_range", (5.0, 200.0))), "bw": rng.uniform(*ranges.get("bandwidth_range", (1.0, 100.0))),
                "mode": pulse["mode"], "spurious": True, "base": pulse["base"], "missed_before": 0,
            })
            pulse = dict(pulse)
            pulse["interval"] *= (1 - split)
            pulse["spurious_before"] = 1  # its interval now starts at the spurious pulse
        out.append(pulse)
    return out[:n_pulses]


def generate_noise_track(n_pulses: int, rng: np.random.RandomState) -> List[Dict[str, Any]]:
    """An unstructured 'track' (no coherent emitter): every pulse random — the sequence
    counterpart of the classification dataset's noise rows."""
    out = []
    for _ in range(n_pulses):
        freq = rng.uniform(4000.0, 30000.0)
        pp = noise_pw_pri(rng)
        out.append({"interval": pp["pri"], "freq_mhz": freq, "pw": pp["pw"], "rise": rng.uniform(1.0, 20.0),
                    "fall": rng.uniform(1.0, 30.0), "bw": rng.uniform(0.1, 300.0), "mode": "n/a (unstructured noise)",
                    "spurious": False, "base": None, "missed_before": 0})
    return out


def build_sequence_dataset_rows(num_classes: int, holdout_class_ids: List[int], tracks_per_class: int,
                                pulses_per_track: int, dataset_seed: int,
                                ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                                missed_pct: float = 0.02, spurious_pct: float = 0.02,
                                noise_track_pct: float = 0.1, track_seed: Optional[int] = None,
                                include_pgri: bool = False, balanced_patterns: bool = False,
                                reveal_class_ids: Optional[List[int]] = None,
                                extra_features: Optional[List[Dict[str, Any]]] = None,
                                capture_date: Optional[date] = None) -> List[Dict[str, Any]]:
    """Rows of a sequence dataset: tracks_per_class tracks per known family, a quarter of that
    per held-out family (true_class_id -1), and noise tracks (true_class_id -1).

    `dataset_seed` fixes the families; `track_seed` the individual tracks drawn from them
    (default: derived from dataset_seed). A test set passes a fresh track_seed, so it gets new
    tracks of the same families rather than a copy of the training tracks.

    `reveal_class_ids`: held-out families that are labelled like known ones (a full share of tracks,
    their own RADAR_FAMILY name) — families a deployed model has since been taught incrementally.
    The library (and the round-robin PRI types) still follows `holdout_class_ids`, so the families
    themselves are exactly the ones the original model was trained without.

    `extra_features`: user-defined columns, each {name, kind: numeric|categorical, min, max, noise_pct
    | values}; every family gets its own value, pulses scatter around it. `capture_date`: the date
    the first tracks were captured (default today); tracks are spread over the following 30 days."""
    library = build_sequence_library(num_classes, dataset_seed, ranges, include_pgri,
                                     holdout_class_ids if balanced_patterns else None)
    rng = aux_rng_for(dataset_seed if track_seed is None else track_seed, "sequence_tracks")
    rows: List[Dict[str, Any]] = []
    track_no = 0
    extras = [dict(e, name=str(e["name"]).strip()) for e in (extra_features or [])]
    mods = family_modulation(num_classes, dataset_seed)
    extra_base = _extra_bases(extras, num_classes, dataset_seed)
    # Own streams (not the track stream): modulation / SNR / dates never shift the pulse values.
    side = aux_rng_for(dataset_seed if track_seed is None else track_seed, "track_side")
    first_day = capture_date or date.today()

    def extra_value(e, c):
        return extra_pulse_value(e, None if c is None else extra_base[e["name"]][c], side)

    def emit(pulses, class_id, label, pattern, agility, family=None):
        nonlocal track_no
        track_no += 1
        stamp = (first_day + timedelta(days=int(side.randint(0, 30)))).strftime("%Y%m%d")
        instance = f"{label}_{stamp}_{track_no:04d}"
        add_side_columns(pulses, mods[family] if family is not None else None, side)
        for idx, pl in enumerate(pulses):
            spur = pl["spurious"]
            row = {
                "Frequency": round(pl["freq_mhz"], 6),
                "PW": round(pl["pw"], 4),
                "PRI": round(pl["interval"], 3) if idx > 0 else None,  # first pulse: no previous pulse
                "Modlntype": pl["mod_type"],
                "Modlnval": round(pl["mod_val"], 4),
                "snr": round(pl["snr"], 2),
                "rise": round(pl["rise"], 1), "fall": round(pl["fall"], 1),
                "timeBW": round(pl["bw"] * MHZ),
            }
            for e in extras:
                row[e["name"]] = extra_value(e, None if spur else family)
            row.update({
                "instance": instance, "class": label,
                "pulse_idx": idx, "true_class_id": class_id,
                "pri_pattern": pattern, "freq_agility": agility,
                "operating_mode": pl["mode"], "is_spurious": int(pl["spurious"]),
                # Ground truth for checking the PRI pattern (never model inputs): this mode's PRI before
                # the pattern, the pattern's deviation from it, and pulses missed just before this one
                # (a missed pulse makes the PRI the sum of two intervals).
                "pri_mode_base_us": round(pl["base"], 3) if pl.get("base") is not None else None,
                "pri_deviation_pct": round((pl["interval"] - pl["base"]) / pl["base"] * 100, 2)
                                     if idx > 0 and pl.get("base") is not None else None,
                "missed_before": pl.get("missed_before", 0),
                "spurious_before": pl.get("spurious_before", 0),
            })
            rows.append(row)

    for c in range(num_classes):
        held_out = c in holdout_class_ids and c not in (reveal_class_ids or [])
        n_tracks = max(1, tracks_per_class // 4) if held_out else tracks_per_class
        for _ in range(n_tracks):
            pulses = generate_track(library[c], pulses_per_track, rng, missed_pct, spurious_pct, ranges)
            emit(pulses, -1 if held_out else c, "UNKNOWN_HOLDOUT" if held_out else f"RADAR_FAMILY_{c:02d}",
                 library[c]["pri_pattern"], library[c]["seq"]["freq_agility"], family=c)
    known = num_classes - len(set(holdout_class_ids) - set(reveal_class_ids or []))
    for _ in range(int(round(known * tracks_per_class * noise_track_pct))):
        emit(generate_noise_track(pulses_per_track, rng), -1, "UNKNOWN_NOISE_OOD",
             "n/a (unstructured noise)", "n/a (unstructured noise)")
    return rows
