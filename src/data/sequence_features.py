"""Sequence-model inputs: per-pulse features computed along each track, their scaling, and
cutting tracks into windows of k pulses.

A window is X = [x_1 … x_k], one row per pulse in arrival order, each row the selected
features of that pulse. Δ features are differences to the PREVIOUS pulse of the same track
(never across tracks), so they expose how a parameter changes pulse to pulse — the PRI
modulation pattern (ΔPRI) or the frequency agility (Δfreq).
"""
import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ZERO_IS_MISSING_SEQ = {"rise", "fall"}


@dataclass(frozen=True)
class SeqFeature:
    label: str
    unit: str
    columns: Tuple[str, ...]
    description: str
    scale: float = 1.0  # the model is fed column × scale (frequency MHz → Hz, bandwidth Hz → MHz)


SEQ_FEATURES: "OrderedDict[str, SeqFeature]" = OrderedDict([
    ("pri", SeqFeature("PRI", "µs", ("PRI",), "Pulse repetition interval: time since this emitter's previous pulse.")),
    ("d_pri", SeqFeature("ΔPRI", "µs", ("PRI",), "PRI_i − PRI_(i−1): how the interval changes pulse to pulse — exposes stagger cycles, ramps, jitter.")),
    ("freq", SeqFeature("Carrier Frequency", "MHz", ("Frequency",), "Carrier frequency in MHz (fed to the model scaled to Hz).", scale=1e6)),
    ("d_freq", SeqFeature("ΔFrequency", "MHz", ("Frequency",), "freq_i − freq_(i−1): frequency agility — 0 for fixed, large for hopping, a cycle for stepped.", scale=1e6)),
    ("pw", SeqFeature("Pulse Width", "µs", ("PW",), "Pulse width.")),
    ("d_pw", SeqFeature("ΔPulse Width", "µs", ("PW",), "PW_i − PW_(i−1): pulse-width changes, mainly at operating-mode switches.")),
    ("bw", SeqFeature("Bandwidth", "Hz", ("timeBW",), "Instantaneous bandwidth of the pulse, in Hz (fed to the model scaled to MHz).", scale=1e-6)),
    ("snr", SeqFeature("SNR", "dB", ("snr",), "Signal-to-noise ratio of the pulse (a capture condition, not an emitter property - useful mainly to see that the model does not lean on it).")),
    ("modtype", SeqFeature("Modulation Type", "", ("Modlntype",), "Intra-pulse modulation type (text: None, LFM, NLFM, Barker, BPSK, Frank ...), fed as a number (its position in the modulation list).")),
    ("modval", SeqFeature("Modulation Value", "", ("Modlnval",), "Parameter of the intra-pulse modulation: chirp rate (MHz/us) for FM, code length for phase codes, 0 for none.")),
    ("rise", SeqFeature("Rise Time", "ns", ("rise",), "Rise time (0 = not measured).")),
    ("fall", SeqFeature("Fall Time", "ns", ("fall",), "Fall time (0 = not measured).")),
    ("duty_cycle", SeqFeature("Duty Cycle", "", ("PW", "PRI"), "PW / PRI.")),
    ("pw_x_pri", SeqFeature("PW × PRI", "µs²", ("PW", "PRI"), "PW · PRI.")),
])
DEFAULT_SEQ_FEATURES = list(SEQ_FEATURES)
DEFAULT_WINDOW = 32
WINDOW_MODES = ("windows", "track")  # "windows": overlapping k-pulse windows; "track": ONE input per track (its first N pulses)
DEFAULT_TRACK_CAP = 128              # "track" mode: pulses of a track used at most (the Training form's default)
MAX_TRACK_PULSES = 512               # hard ceiling of "track" mode, also with no cap (attention cost grows with length²)

# Extra columns of an uploaded or generated track file become features too:
#   "col:<column>"   a numeric column's value per pulse
#   "dcol:<column>"  its change from the previous pulse of the same track (Δ)
#   "cat:<column>"   a text column, fed as a number (see category_code)
DATASET_VALUE, DATASET_DELTA, DATASET_CATEGORY = "col:", "dcol:", "cat:"
MODULATION_TYPES = ["None", "LFM", "NLFM", "Barker", "BPSK", "Frank"]  # Modlntype -> its position (same list as the generator)
SEQ_RESERVED_COLUMNS = {
    "instance", "pulse_idx", "Frequency", "PW", "PRI", "Modlntype", "Modlnval", "snr", "rise", "fall", "timeBW",
    "true_class_id", "class", "pri_pattern", "freq_agility",
    # names used before the dataset columns were renamed (still read, see LEGACY_COLUMNS)
    "track_id", "true_label", "carrier_freq_mhz", "carrier_freq_hz", "pulse_width_us", "pri_us", "rise_time_ns",
    "fall_time_ns", "bandwidth_hz", "bandwidth_mhz",
    "operating_mode", "is_spurious", "toa_us", "toa_ns", "toa",
    "pri_mode_base_us", "pri_deviation_pct", "missed_before", "spurious_before",  # ground-truth pattern references
}


PRI_COLUMN = "PRI"  # the first pulse of a track has none (no previous pulse)
MIN_TRAIN_CROP = 2      # shortest training crop (a decision needs a PRI, i.e. ≥ 2 pulses)
SHORT_LENGTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)  # τ_d calibrated per length below k


_UNSAFE_TEXT = re.compile(r"[<>&\"'`]")
_UNSAFE_NAME = re.compile(r"[^\w.\- ()%/+µΔ]", re.UNICODE)


_CLEAN_MARK = "dads_sanitized"  # df.attrs key: the column names a frame was last checked with


def _unsafe_text_column(s: pd.Series) -> Optional[dict]:
    """{dirty value: cleaned value} of a text column, or None when it holds no markup characters.
    Only the distinct values are checked: a label column of a million rows has a few dozen of them."""
    dirty = {}
    for v in pd.unique(s.dropna()):
        text = str(v)
        if _UNSAFE_TEXT.search(text):
            dirty[v] = _UNSAFE_TEXT.sub("", text)
    return dirty or None


def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Uploaded files are untrusted: column names and class/text labels end up in the web pages, so
    markup characters are removed from them. Numeric data is untouched; clean frames pass through as is.

    A checked frame is marked (df.attrs, carried along by copies and slices), so the cached datasets and
    the per-track calls of the inference path are not scanned again. The mark holds the column names:
    a frame that gained or renamed columns since is checked again."""
    cols = tuple(str(c) for c in df.columns)
    if df.attrs.get(_CLEAN_MARK) == cols:
        return df
    names = [_UNSAFE_NAME.sub("_", c) for c in cols]
    if names != list(cols):
        df = df.copy()
        df.columns = names
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            dirty = _unsafe_text_column(s)
            if dirty:
                df = df.copy()
                df[col] = s.map(lambda v: dirty.get(v, v))
    df.attrs[_CLEAN_MARK] = tuple(str(c) for c in df.columns)
    return df


# Columns of files written before the dataset used the standard names (Frequency in MHz, PW, PRI, rise,
# fall, timeBW in Hz, instance, class): old name -> (new name, factor to the new unit).
LEGACY_COLUMNS = {
    "carrier_freq_mhz": ("Frequency", 1.0), "carrier_freq_hz": ("Frequency", 1e-6),
    "pulse_width_us": ("PW", 1.0), "pri_us": ("PRI", 1.0),
    "rise_time_ns": ("rise", 1.0), "fall_time_ns": ("fall", 1.0),
    "bandwidth_hz": ("timeBW", 1.0), "bandwidth_mhz": ("timeBW", 1e6),
    "track_id": ("instance", None), "true_label": ("class", None),
}
# Standard column order of a track file; anything else (extra features) sits between timeBW and instance.
COLUMN_ORDER = ["Frequency", "PW", "PRI", "Modlntype", "Modlnval", "snr", "rise", "fall", "timeBW"]
COLUMN_TAIL = ["instance", "class", "pulse_idx", "true_class_id"]


def upgrade_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency is in MHz (`Frequency`) and bandwidth in Hz (`timeBW`). Files written with the
    earlier column names or the other unit (carrier_freq_hz, bandwidth_mhz ...) are converted; other
    files pass through untouched."""
    for old, (new, factor) in LEGACY_COLUMNS.items():
        if old in df.columns and new not in df.columns:
            df = df.copy()
            df[new] = df[old] if factor is None else pd.to_numeric(df[old], errors="coerce") * factor
    legacy = [c for c in LEGACY_COLUMNS if c in df.columns]
    if legacy:
        df = df.drop(columns=legacy)
    return sanitize_frame(df)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns in the standard order: the measurements, extra features, instance, class, then the
    bookkeeping (pulse_idx, true_class_id) and any other columns (simulator ground truth) last."""
    head = [c for c in COLUMN_ORDER if c in df.columns]
    tail = [c for c in COLUMN_TAIL if c in df.columns]
    used = set(head) | set(tail)
    rest = [c for c in df.columns if c not in used]
    extras = [c for c in rest if c not in SIMULATOR_INFO]
    others = [c for c in rest if c in SIMULATOR_INFO]
    return df[head + extras + tail + others]


# Simulator ground truth: never a model input, only for checking and the analysis pages.
SIMULATOR_INFO = {"pri_pattern", "freq_agility", "operating_mode", "is_spurious", "pri_mode_base_us",
                  "pri_deviation_pct", "missed_before", "spurious_before"}


def category_code(value: Any, column: str = "") -> float:
    """A text value as the number the model is fed. Modlntype: its position in MODULATION_TYPES
    (unknown text -> after the last one). Any other text column: a stable fraction in [0, 1) derived
    from the text itself (crc32), so the same category gets the same number in every dataset and no
    category list has to be stored with the model."""
    text = str(value).strip()
    if column == "Modlntype":
        return float(MODULATION_TYPES.index(text)) if text in MODULATION_TYPES else float(len(MODULATION_TYPES))
    return (zlib.crc32(text.encode("utf-8")) % 100_003) / 100_003.0


def is_delta_feature(name: str) -> bool:
    """Δ features (change from the previous pulse): heavy-tailed, so they get a signed-log scaling."""
    return name.startswith("d_") or name.startswith(DATASET_DELTA)


def dataset_feature_column(name: str) -> Optional[str]:
    """The file column behind a dataset feature name, or None for a built-in feature."""
    for prefix in (DATASET_DELTA, DATASET_VALUE, DATASET_CATEGORY):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return None


def sequence_dataset_columns(df: pd.DataFrame) -> List[str]:
    """Extra numeric columns of a track file that can be used as features (file order)."""
    out = []
    for c in df.columns:
        if c in SEQ_RESERVED_COLUMNS or str(c).startswith("Unnamed"):
            continue
        values = pd.to_numeric(df[c], errors="coerce")
        if values.notna().mean() >= 0.5 and values.nunique(dropna=True) > 1:
            out.append(str(c))
    return out


def sequence_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Extra text columns of a track file (2-50 distinct values, not the class/instance labels) that can
    be used as categorical features."""
    out = []
    for c in df.columns:
        if c in SEQ_RESERVED_COLUMNS or str(c).startswith("Unnamed") or c in SIMULATOR_INFO:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if 2 <= s.nunique(dropna=True) <= 50:
            out.append(str(c))
    return out


def seq_feature_label(name: str) -> str:
    col = dataset_feature_column(name)
    if col is None:
        return SEQ_FEATURES[name].label if name in SEQ_FEATURES else name
    return f"Δ{col}" if name.startswith(DATASET_DELTA) else col


def seq_feature_columns(name: str) -> Tuple[str, ...]:
    col = dataset_feature_column(name)
    return (col,) if col is not None else SEQ_FEATURES[name].columns


def validate_seq_features(names: Sequence[str], dataset_columns: Optional[Sequence[str]] = None) -> List[str]:
    """Built-in features in registry order, then dataset-column features in the given order —
    a fixed input layout. Dataset features must name a column in `dataset_columns` (when given)."""
    names = list(dict.fromkeys(names))
    unknown = []
    for n in names:
        col = dataset_feature_column(n)
        if col is None:
            if n not in SEQ_FEATURES:
                unknown.append(n)
        elif dataset_columns is not None and col not in dataset_columns:
            unknown.append(n)
    if unknown:
        raise ValueError(f"Unknown sequence feature(s) {unknown}: not a built-in feature "
                         f"({list(SEQ_FEATURES)}) or a numeric column of the sequence dataset.")
    if not names:
        raise ValueError("Select at least one sequence feature.")
    return [n for n in SEQ_FEATURES if n in names] + [n for n in names if dataset_feature_column(n) is not None]


def required_seq_columns(names: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for n in names:
        cols += [c for c in seq_feature_columns(n) if c not in cols]
    return cols


def _clean_column(values: np.ndarray, col: str) -> np.ndarray:
    """Missing values of one track's column (blank, or 0 rise/fall time). Leading gaps — above all
    the first pulse's PRI, which doesn't exist — take the first measured value (the nearest pulse,
    known as soon as pulse 2 arrives, so no later pulse is used and the second pulse's Δ is 0, "no
    change"). Gaps further in take the track's median — a per-track estimate that never mixes in
    other emitters. A column with no value at all → zeros."""
    v = values.astype(np.float64).copy()
    if col in ZERO_IS_MISSING_SEQ:
        v[v == 0] = np.nan
    missing = np.isnan(v)
    if missing.all():
        return np.zeros_like(v)
    first = int(np.argmax(~missing))
    v[:first] = v[first]
    v[np.isnan(v)] = np.nanmedian(v)
    return v


def has_pri(track: pd.DataFrame) -> bool:
    """A track (or the pulses of it seen so far) has at least one PRI value — i.e. ≥ 2 pulses."""
    return PRI_COLUMN in track.columns and pd.to_numeric(track[PRI_COLUMN], errors="coerce").notna().any()


def causal_window(track: pd.DataFrame, names: Sequence[str], n_seen: int, k: int, whole: bool = False) -> np.ndarray:
    """The window a decision after `n_seen` pulses is based on: the last ≤ k of those pulses, with
    features computed from pulses already seen only (plus the one before the window, so its first
    Δ is the real change). Live Simulation, the Inference page's "decision as pulses arrive" and
    the accuracy-vs-pulses curve all use exactly this. `whole` (track mode): the FIRST ≤ k pulses
    instead — a whole-track model always sees the track from its start."""
    if whole:
        return track_feature_matrix(track.iloc[:min(n_seen, k)], names)
    return track_feature_matrix(track.iloc[max(0, n_seen - k - 1):n_seen], names)[-k:]


def track_feature_matrix(track: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """[n_pulses × n_features] for ONE track, rows in arrival order. The first pulse (no
    previous pulse) has no PRI and no Δ values; its PRI is filled from the second pulse's (see
    _clean_column) and its Δ values are 0 ("no change")."""
    track = upgrade_columns(track)
    categorical = {dataset_feature_column(n) for n in names if n.startswith(DATASET_CATEGORY)} | \
                  ({"Modlntype"} if "modtype" in names else set())
    cols = {}
    for c in required_seq_columns(names):
        if c in categorical:  # text column: each value -> its number (missing -> the track's most common value)
            raw = track[c]
            fill = raw.mode().iloc[0] if raw.notna().any() else ""
            cols[c] = np.array([category_code(v, c) for v in raw.fillna(fill)], dtype=np.float64)
        else:
            cols[c] = _clean_column(pd.to_numeric(track[c], errors="coerce").to_numpy(), c)

    def delta(v):
        d = np.diff(v, prepend=v[0])
        return d

    out = []
    for n in names:
        col = dataset_feature_column(n)
        if col is not None:
            out.append(delta(cols[col]) if n.startswith(DATASET_DELTA) else cols[col])
        elif n == "pri": out.append(cols["PRI"])
        elif n == "d_pri": out.append(delta(cols["PRI"]))
        elif n == "freq": out.append(cols["Frequency"] * 1e6)  # the model sees Hz, as it always did
        elif n == "d_freq": out.append(delta(cols["Frequency"] * 1e6))
        elif n == "pw": out.append(cols["PW"])
        elif n == "d_pw": out.append(delta(cols["PW"]))
        elif n == "bw": out.append(cols["timeBW"] / 1e6)  # the model sees MHz
        elif n == "snr": out.append(cols["snr"])
        elif n == "modtype": out.append(cols["Modlntype"])
        elif n == "modval": out.append(cols["Modlnval"])
        elif n == "rise": out.append(cols["rise"])
        elif n == "fall": out.append(cols["fall"])
        elif n == "duty_cycle": out.append(np.where(cols["PRI"] > 0, cols["PW"] / np.maximum(cols["PRI"], 1e-12), 0.0))
        elif n == "pw_x_pri": out.append(cols["PW"] * cols["PRI"])
    return np.stack(out, axis=1)


def ordered_tracks(df: pd.DataFrame) -> List[Tuple[Any, pd.DataFrame]]:
    """Tracks in file order, each sorted by pulse_idx when present (else row order)."""
    tracks = []
    for tid, t in df.groupby("instance", sort=False):
        if "pulse_idx" in t.columns:
            t = t.sort_values("pulse_idx", kind="mergesort")
        tracks.append((tid, t))
    return tracks


def windows_of(matrix: np.ndarray, k: int, stride: int, whole: bool = False) -> List[np.ndarray]:
    """Consecutive windows of k pulses (stride apart). A track shorter than k gives none.
    `whole` (track mode): ONE input per track, its first ≤ k pulses - a shorter track is kept
    (padded later); a track needs 2 pulses (the first has no PRI)."""
    n = len(matrix)
    if whole:
        return [matrix[:k]] if n >= 2 else []
    return [matrix[s:s + k] for s in range(0, n - k + 1, max(1, stride))]


def pad_stack(xs: Sequence[np.ndarray], n_features: Optional[int] = None) -> np.ndarray:
    """Stack windows of (possibly) different lengths into one array. All the same length -> plain
    [N, L, F]. Otherwise padded at the end to the longest, with a mask as ONE extra last channel
    ([N, L, F + 1], 1 = real pulse): the model splits it off (SequenceDualHeadModel.forward)."""
    lengths = {len(w) for w in xs}
    if not xs:
        return np.zeros((0, 1, n_features or 1), np.float32)
    if len(lengths) == 1:
        return np.stack(xs)
    L, f = max(lengths), xs[0].shape[1]
    out = np.zeros((len(xs), L, f + 1), dtype=np.asarray(xs[0]).dtype)
    for i, w in enumerate(xs):
        out[i, :len(w), :f] = w
        out[i, :len(w), f] = 1.0
    return out


@dataclass
class SequenceSpec:
    """The sequence model's input contract, saved with the model: feature names, per-feature
    scaling (fitted on the training tracks' pulses) and the training window length k (the
    model itself accepts any length, see sequence_models.py)."""
    names: List[str]
    mean: np.ndarray
    std: np.ndarray
    window: int
    # Δ features: x → sign(x)·ln(1 + |x| / c_j) before the z-score, c_j = median |Δ| over the
    # training pulses. Δ values are heavy-tailed (a hopping family's Δfrequency is ~100× a stepped
    # family's), and a plain z-score would squash the small-but-meaningful changes to ~0.
    # 0 = no log step (non-Δ features); None = models trained before this existed.
    delta_scale: Optional[np.ndarray] = None
    # "windows": the model is trained and applied on windows of k = `window` pulses (a track = many
    # windows, voted). "track": ONE input per track = its first `window` pulses (a cap; shorter tracks
    # are padded and masked), no windowing, no vote.
    mode: str = "windows"

    @property
    def whole(self) -> bool:
        return self.mode == "track"

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def columns(self) -> List[str]:
        return required_seq_columns(self.names)

    @classmethod
    def fit(cls, names: List[str], pulses: np.ndarray, window: int, mode: str = "windows") -> "SequenceSpec":
        scale = np.zeros(len(names))
        for j, n in enumerate(names):
            if is_delta_feature(n):
                mag = np.abs(pulses[:, j])
                mag = mag[mag > 0]
                scale[j] = float(np.median(mag)) if len(mag) and np.median(mag) > 0 else 1.0
        z = cls._log_delta(pulses, scale)
        std = z.std(axis=0)
        std[std == 0] = 1.0
        return cls(list(names), z.mean(axis=0), std, int(window), delta_scale=scale, mode=mode)

    @staticmethod
    def _log_delta(x: np.ndarray, scale: Optional[np.ndarray]) -> np.ndarray:
        if scale is None or not np.any(scale > 0):
            return x
        x = np.array(x, dtype=np.float64, copy=True)
        for j in np.nonzero(scale > 0)[0]:
            x[..., j] = np.sign(x[..., j]) * np.log1p(np.abs(x[..., j]) / scale[j])
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((self._log_delta(x, self.delta_scale) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        d = {"kind": "sequence", "names": list(self.names), "mean": self.mean.tolist(),
             "std": self.std.tolist(), "window": self.window}
        if self.delta_scale is not None:
            d["delta_scale"] = self.delta_scale.tolist()
        d["mode"] = self.mode
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SequenceSpec":
        scale = np.asarray(d["delta_scale"], dtype=np.float64) if d.get("delta_scale") is not None else None
        return cls(list(d["names"]), np.asarray(d["mean"], dtype=np.float64),
                   np.asarray(d["std"], dtype=np.float64), int(d["window"]), delta_scale=scale,
                   mode=d.get("mode", "windows"))


def track_windows(df: pd.DataFrame, names: List[str], k: int, stride: int, whole: bool = False
                  ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """All windows of every track: (raw windows, label per window, track id per window)."""
    xs, ys, ts = [], [], []
    for tid, t in ordered_tracks(df):
        m = track_feature_matrix(t, names)
        label = int(t["true_class_id"].iloc[0])
        for w in windows_of(m, k, stride, whole):
            xs.append(w); ys.append(label); ts.append(tid)
    return xs, np.asarray(ys, dtype=np.int64), np.asarray(ts)


def split_track_ids(df: pd.DataFrame, split_seed: int = 0, val_fraction: float = 0.2,
                    cal_fraction: float = 0.1) -> Tuple[set, set, set, set]:
    """(train, validation, calibration, open-set) track ids — the split training uses, reproducible
    from the seed (incremental training rebuilds a model's split from it).

    Split PER FAMILY, so every family has training tracks (its prototypes are built from them —
    a family without any would always look unknown) and validation tracks to score it. A further share is held
    back purely to calibrate the distance threshold: used neither for training nor for the
    prototypes, so the calibrated τ_d accepts about the requested share of known windows on NEW
    tracks too (calibrating on the validation windows would set τ_d too tight)."""
    track_labels = df.groupby("instance", sort=False)["true_class_id"].first()
    rng = np.random.RandomState(split_seed)
    val_ids, cal_ids, train_ids = set(), set(), set()
    for c in sorted(track_labels[track_labels >= 0].unique()):
        ids = track_labels[track_labels == c].index.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_val = max(1, int(round(len(ids) * val_fraction))) if len(ids) >= 2 else 0
        n_cal = int(len(ids) * cal_fraction) if len(ids) - n_val - int(len(ids) * cal_fraction) >= 1 else 0
        val_ids.update(ids[:n_val].tolist())
        cal_ids.update(ids[n_val:n_val + n_cal].tolist())
        train_ids.update(ids[n_val + n_cal:].tolist())
    open_ids = set(track_labels[track_labels < 0].index.tolist())
    return train_ids, val_ids, cal_ids, open_ids


def cv_split(df: pd.DataFrame, split_seed: int = 0, test_pct: float = 20.0, folds: int = 4,
             cal_fraction: float = 0.1, include_test: bool = False) -> Tuple[set, List[Tuple[set, set, set, set]]]:
    """Cross-validation split by TRACK, stratified per family. Returns (test ids, one
    (train, validation, calibration, open-set) tuple of track ids per fold).

    test_pct % of every known family's tracks (and of the unknown tracks) are set aside as the TEST
    split. The remaining known tracks of a family are cut into `folds` equal buckets; fold f validates on
    bucket f and trains on the other buckets (e.g. test 20 % and 4 folds = 20:20:20:20:20 - three buckets
    train, one validates, the fifth is the test split), so every track validates exactly once. With
    `include_test` the test tracks join every fold's training pool (a model for deployment; the test
    split is then no independent test). A further `cal_fraction` of each fold's training pool is held back
    to calibrate the distance threshold, as in split_track_ids. The unknown tracks (held-out families,
    noise) are never trained on: those outside the test split form the open-set pool of every fold."""
    if folds < 2:
        raise ValueError("Cross-validation needs at least 2 folds.")
    if not 0 <= test_pct < 100:
        raise ValueError("The test share must be 0-99 %.")
    track_labels = df.groupby("instance", sort=False)["true_class_id"].first()
    rng = np.random.RandomState(split_seed)
    test_ids: set = set()
    buckets: List[List[set]] = [[] for _ in range(folds)]
    for c in sorted(track_labels[track_labels >= 0].unique()):
        ids = track_labels[track_labels == c].index.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_test = int(round(len(ids) * test_pct / 100.0))
        if test_pct > 0 and n_test == 0 and len(ids) > folds + 1:
            n_test = 1
        if len(ids) - n_test < folds + 1:
            raise ValueError(f"Family {c} has {len(ids)} tracks - too few for a {test_pct:g} % test split and {folds} folds "
                             f"(it needs at least {folds + 1} tracks left for training and validation). Use more tracks, fewer folds or a smaller test share.")
        test_ids.update(ids[:n_test].tolist())
        for f, part in enumerate(np.array_split(ids[n_test:], folds)):
            buckets[f].append(set(part.tolist()))
    unknown = track_labels[track_labels < 0].index.to_numpy()
    unknown = unknown[rng.permutation(len(unknown))]
    n_test_unknown = int(round(len(unknown) * test_pct / 100.0))
    test_ids.update(unknown[:n_test_unknown].tolist())
    test_unknown = set(unknown[:n_test_unknown].tolist())
    # The open-set pool that validates every fold (and takes part in choosing the kept fold) leaves out
    # the test split's unknown tracks, so the test result's open-set AUROC stays independent. With
    # include_test the test split is no independent test anyway, and every unknown track is used.
    open_ids = set(unknown.tolist()) if include_test else set(unknown.tolist()) - test_unknown
    fold_splits = []
    for f in range(folds):
        val = set().union(*buckets[f])
        pool = set().union(*[b for g, bs in enumerate(buckets) if g != f for b in bs])
        if include_test:
            pool |= {t for t in test_ids if track_labels[t] >= 0}
        cal, train = set(), set()
        for c in sorted(track_labels[track_labels >= 0].unique()):
            fam = sorted((t for t in pool if track_labels[t] == c), key=str)
            fam = [fam[i] for i in rng.permutation(len(fam))]
            n_cal = int(len(fam) * cal_fraction) if len(fam) - int(len(fam) * cal_fraction) >= 1 else 0
            cal.update(fam[:n_cal])
            train.update(fam[n_cal:])
        fold_splits.append((train, val, cal, open_ids))
    return test_ids, fold_splits


def load_sequence_dataset_for_training(df: pd.DataFrame, names: List[str], k: int,
                                       split_seed: int = 0, val_fraction: float = 0.2,
                                       spec: Optional[SequenceSpec] = None, cal_fraction: float = 0.1,
                                       split: Optional[Tuple[set, set, set, set]] = None,
                                       mode: str = "windows") -> Dict[str, Any]:
    """Splits by TRACK (never by window: overlapping windows of one track would leak between
    training and validation), per family: 20% validation, 10% calibration, the rest training; unknown tracks
    (true_class_id −1: held-out families + noise) → open-set pool. Training windows overlap
    by half (stride k/2) for more examples; validation/open-set windows don't overlap.
    Scaling is fitted on training tracks' pulses only (unless an existing `spec` is given).

    mode="track": no windowing - ONE input per track, its first k pulses (k = the cap; k <= 0 = no cap,
    up to MAX_TRACK_PULSES). Shorter tracks are kept: padded at the end with a mask channel (pad_stack)."""
    if spec is not None:
        mode, k = spec.mode, spec.window
    if mode not in WINDOW_MODES:
        raise ValueError(f"Unknown window mode '{mode}'; choose from {list(WINDOW_MODES)}.")
    whole = mode == "track"
    track_lengths = df.groupby("instance", sort=False).size()
    if whole:
        k = min(int(k) if k and k > 0 else int(track_lengths.max()), MAX_TRACK_PULSES)
    track_labels = df.groupby("instance", sort=False)["true_class_id"].first()
    train_ids, val_ids, cal_ids, open_ids = split if split is not None else split_track_ids(df, split_seed, val_fraction, cal_fraction)

    def part(ids, stride):
        sub = df[df["instance"].isin(ids)]
        return track_windows(sub, names, k, stride, whole) if len(sub) else ([], np.zeros(0, np.int64), np.zeros(0))

    tr_x, tr_y, _ = part(train_ids, max(1, k // 2))
    va_x, va_y, va_t = part(val_ids, k)
    op_x, op_y, _ = part(open_ids, k)
    cal_x, _, _ = part(cal_ids, k)
    # Decisions on the first L < k pulses of a track (Live Simulation, Inference's progression) sit
    # farther from prototypes built from k-pulse windows, so τ_d is also calibrated per length L on
    # exactly those windows: the first L pulses of each calibration track (validation tracks when
    # there are too few), computed from those pulses only.
    short_ids = cal_ids if len(cal_ids) >= 10 else val_ids
    short_tracks = ordered_tracks(df[df["instance"].isin(short_ids)])
    cal_short_raw = {L: [causal_window(t, names, L, k, whole) for _, t in short_tracks if len(t) >= L and has_pri(t.iloc[:L])]
                     for L in SHORT_LENGTHS if L < k}
    if not tr_x:
        raise ValueError(f"No training windows: every known track is shorter than the window length k={k}.")
    if spec is None:
        train_pulses = np.concatenate([track_feature_matrix(t, names) for _, t in ordered_tracks(df[df["instance"].isin(train_ids)])])
        spec = SequenceSpec.fit(names, train_pulses, k, mode)

    def stack(xs):
        if not xs:
            return np.zeros((0, k, len(names)), np.float32)
        if whole:  # tracks of different lengths: scale each, then pad + mask
            return pad_stack([spec.transform(w) for w in xs])
        return spec.transform(np.stack(xs))

    known_ids = set(track_labels[track_labels >= 0].index)
    too_short = int(sum(1 for t in known_ids if track_lengths[t] < (2 if whole else k)))  # tracks that give no input at all
    known_class_ids = sorted(set(track_labels[track_labels >= 0].astype(int).tolist()))
    return {
        "train_x": stack(tr_x), "train_y": tr_y,
        "val_x": stack(va_x), "val_y": va_y, "val_tracks": va_t,
        "openset_x": stack(op_x), "openset_y": op_y,
        "known_class_ids": known_class_ids,
        "holdout_class_ids": [],
        "feature_spec": spec,
        "val_track_ids": sorted(val_ids, key=str),
        "train_track_ids": sorted(train_ids, key=str), "cal_track_ids": sorted(cal_ids, key=str),
        "cal_x": stack(cal_x), "num_cal_tracks": len(cal_ids),
        "cal_short_x": {L: spec.transform(np.stack(ws)) for L, ws in cal_short_raw.items() if len(ws) >= 5},
        "cal_short_source": "calibration" if short_ids is cal_ids else "validation",
        # Half of the training batches use a random shorter crop of the windows (MIN_TRAIN_CROP … k
        # pulses), so the model also learns to decide from the few pulses a new track starts with.
        # Track mode crops the FIRST pulses (the padding sits at the end, a decision starts at the track's start).
        "train_crop": {"prob": 0.5, "min": MIN_TRAIN_CROP, "prefix": whole},
        "window_mode": mode, "window": k, "known_tracks_without_input": too_short,
    }
