import zlib
import numpy as np
from typing import Dict, Tuple

from src.data.pri_patterns import PRI_PATTERN_TYPES, PGRI, fit_pgri_to_pulse_width, pgri_params
from src.data.operating_modes import OPERATING_MODES


DEFAULT_DUTY_RANGE = (0.001, 0.10)   # family duty cycle PW / PRI: 0.1% … 10%
NOISE_DUTY_RANGE = (0.0005, 0.30)    # noise pulses: much wider, still physical


def family_pulse_width(u: float, pri_us: float, duty_range: Tuple[float, float], pw_range: Tuple[float, float]) -> float:
    """PW = duty × PRI, the duty cycle log-uniform in duty_range (u ∈ [0, 1) picks where). Then
    kept inside pw_range where possible; if pw_range and the duty range can't both hold for this
    PRI, the duty range wins (a pulse longer than its interval is impossible, a PW outside the
    configured range is merely unusual)."""
    lo, hi = sorted(float(v) for v in duty_range)
    lo, hi = max(lo, 1e-5), min(max(hi, max(lo, 1e-5)), 1.0)
    duty = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    pw = float(np.clip(duty * pri_us, *pw_range))
    return float(np.clip(pw, lo * pri_us, hi * pri_us))


def noisy_pulse_width(nominal_us: float, rng: np.random.RandomState) -> float:
    """Measurement noise on PW: N(0, 0.2 µs), but at most 10% of the pulse (a 0.2 µs pulse can't
    be measured as negative), floored at 0.02 µs."""
    return max(0.02, nominal_us + rng.normal(0, min(0.2, 0.1 * nominal_us)))


def build_emitter_library(
    num_classes: int,
    seed: int = 42,
    freq_range: Tuple[float, float] = (2000.0, 18000.0),
    pw_range: Tuple[float, float] = (0.5, 50.0),
    pri_range: Tuple[float, float] = (10.0, 1000.0),
    rise_range: Tuple[float, float] = (5.0, 150.0),
    bandwidth_range: Tuple[float, float] = (1.0, 100.0),
    fall_range: Tuple[float, float] = (5.0, 200.0),
    include_pgri: bool = False,
    duty_range: Tuple[float, float] = DEFAULT_DUTY_RANGE,
) -> Dict[int, Dict[str, float]]:
    """Builds base PDW centroids for `num_classes` radar families. Shared by the track
    generator and the Testing/Inference simulators so they draw from the same
    underlying emitter population. `seed` controls which specific archetypes get drawn —
    pass a fresh seed each run for genuinely different training populations, not the same
    20 families every time.

    Each family also gets a `pri_pattern` (one of the six standard radar PRI modulation
    types — constant/jittered/staggered/sliding/wobulated/dwell_switch), so different
    families are distinguishable not just by their mean PRI but by the *shape* of its
    distribution — a real, literature-backed ESM discriminant.

    Pulse width follows from a per-family DUTY CYCLE (PW / PRI), drawn log-uniformly from
    `duty_range` (fractions, default 0.1%–10%) — real radars are designed around their duty
    cycle, and drawing PW and PRI independently gave families whose pulses were longer than the
    interval between them. `pw_range` stays an outer limit (see family_pulse_width).

    Bandwidth and fall time were added after the original measurements, so they are drawn from
    their own random stream (derived from `seed`): a given seed still produces exactly the
    frequency/PW/PRI/rise values it always did, with the new columns added alongside.
    """
    rng = np.random.RandomState(seed)
    library = {}
    for c in range(num_classes):
        library[c] = {
            "freq": rng.uniform(*freq_range),
            "duty_u": rng.uniform(),             # position in the duty-cycle range (PW set below)
            "pri": rng.uniform(*pri_range),      # Delta ToA / PRI in us
            "rise": rng.uniform(*rise_range),    # Rise time in ns
            "pri_pattern": rng.choice(PRI_PATTERN_TYPES),
        }
    for c in range(num_classes):
        lib = library[c]
        lib["pw"] = family_pulse_width(lib.pop("duty_u"), lib["pri"], duty_range, pw_range)  # Pulse Width in us
    extra_rng = aux_rng_for(seed, "library")
    bandwidths = extra_rng.uniform(*bandwidth_range, size=num_classes)  # MHz
    falls = extra_rng.uniform(*fall_range, size=num_classes)            # ns
    for c in range(num_classes):
        library[c]["bw"] = float(bandwidths[c])
        library[c]["fall"] = float(falls[c])
    if include_pgri:
        # From its own random stream, so all other draws for this seed are unchanged: each family
        # becomes PGRI with probability 1/7 (one of seven patterns), with its own group shape —
        # fitted to its pulse width, so pulses inside a group never overlap (a family whose pulses
        # are too wide for any short intra-group PRI keeps its own pattern).
        pgri_rng = aux_rng_for(seed, "pgri")
        worst = max(m["pw_mult"] / m["pri_mult"] for m in OPERATING_MODES)
        for c in range(num_classes):
            switch, shape = pgri_rng.random() < 1.0 / 7, pgri_params(pgri_rng)
            if switch:
                shape = fit_pgri_to_pulse_width(shape, library[c]["pw"], library[c]["pri"], worst)
            if switch and shape is not None:
                library[c]["pri_pattern"] = PGRI
                library[c]["pgri"] = shape
    return library


def aux_rng_for(seed: int, purpose: str) -> np.random.RandomState:
    """A random stream derived from `seed`, separate from the main one. Used for values added
    after the original generator (bandwidth, fall time), so drawing them never shifts the main
    stream — existing seeds keep producing identical frequency/PW/PRI/rise values."""
    return np.random.RandomState((seed * 1_000_003 + zlib.crc32(purpose.encode())) % (2 ** 32))


def noise_pw_pri(rng: np.random.RandomState) -> Dict[str, float]:
    """PW and PRI of an unstructured noise pulse: PRI 2–60 µs, duty cycle log-uniform over
    NOISE_DUTY_RANGE (wide, but never a pulse longer than its interval)."""
    u = rng.uniform()
    pri = rng.uniform(2.0, 60.0)
    lo, hi = NOISE_DUTY_RANGE
    pw = float(np.clip(pri * np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))), 0.05, hi * pri))
    return {"pw": pw, "pri": pri}


def frequency_to_band(freq_mhz: float) -> str:
    """Maps a carrier frequency to its standard radar frequency band letter designation."""
    if freq_mhz < 2000: return "L-BAND"
    if freq_mhz < 4000: return "S-BAND"
    if freq_mhz < 8000: return "C-BAND"
    if freq_mhz < 12000: return "X-BAND"
    if freq_mhz < 18000: return "KU-BAND"
    if freq_mhz < 27000: return "K-BAND"
    return "KA-BAND"
