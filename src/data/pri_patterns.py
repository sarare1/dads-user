import numpy as np
from typing import Optional

PRI_PATTERN_TYPES = ["constant", "jittered", "staggered", "sliding", "wobulated", "dwell_switch"]
"""The six standard radar PRI modulation types from the ESM/PRI-recognition literature. The
previous generator only ever produced 'jittered' (a fixed mean + Gaussian noise) — this module
adds the other five, which is what PRI modulation recognition research actually
discriminates between."""


# Pulse Group Repetition Interval: the emitter transmits GROUPS of pulses — a few pulses at a short
# intra-group PRI, then a long gap — and the groups repeat every PGRI (= the family's base PRI here).
# Kept out of PRI_PATTERN_TYPES so seeds generated before it existed keep their patterns; datasets
# opt in (include_pgri) and some families are then switched to it (see build_emitter_library).
PGRI = "pgri"


def pgri_params(rng: np.random.RandomState) -> dict:
    """Group size 2–8 pulses; intra-group PRI 4–20% of the group interval, capped so the gap
    before the next group stays ≥ 40% of the group interval."""
    n = int(rng.randint(2, 9))
    frac = float(rng.uniform(0.04, min(0.2, 0.6 / (n - 1))))
    return {"group_size": n, "intra_frac": frac}


PGRI_MAX_INTRA_DUTY = 0.5  # pulses inside a group must still fit: PW ≤ 50% of the intra-group PRI


def fit_pgri_to_pulse_width(shape: dict, pw_us: float, pri_us: float, worst_pw_per_pri: float) -> Optional[dict]:
    """A group shape that is physically possible for a family with this pulse width: the short
    intra-group PRI (pri · mult · intra_frac) must exceed PW / PGRI_MAX_INTRA_DUTY in every
    operating mode (`worst_pw_per_pri` = the largest pw_mult / pri_mult). Shapes that already fit
    are returned unchanged; otherwise intra_frac is raised to the minimum and the group shortened
    so the gap stays at least twice the intra-group PRI. None when even a 2-pulse group can't fit
    (the family then keeps its own pattern)."""
    need = pw_us * worst_pw_per_pri / (pri_us * PGRI_MAX_INTRA_DUTY)
    if shape["intra_frac"] >= need:
        return shape
    n_max = int(np.floor(1.0 / need - 1.0)) if need > 0 else shape["group_size"]
    if n_max < 2:
        return None
    return {"group_size": min(shape["group_size"], n_max), "intra_frac": float(need)}


def pgri_interval(position_in_group: int, base_pri: float, params: dict) -> float:
    """Interval BEFORE a pulse: the gap for the group's first pulse, the short intra-group PRI otherwise."""
    n, frac = params["group_size"], params["intra_frac"]
    return base_pri * (1 - (n - 1) * frac) if position_in_group == 0 else base_pri * frac
