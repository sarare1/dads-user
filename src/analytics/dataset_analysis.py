"""Dataset analysis shared by the Dataset page (quick health check) and the Analytics page
(deep analysis): one canonical "analysis frame" of every PDW feature, plus the per-feature
distributions, feature-usefulness ranking, radar/ESM views and data-quality summaries both
pages draw from — so the two pages never disagree about what's in a dataset.

Rise/fall time values of exactly 0 mean "not measured" (see sequence_features.ZERO_IS_MISSING_SEQ), so
they're excluded from distributions/statistics here and reported in data quality instead.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.sequence_features import ZERO_IS_MISSING_SEQ as ZERO_IS_MISSING, sequence_dataset_columns
from src.data.generator import frequency_to_band

# Canonical analysis names -> the dataset column carrying that measurement.
MEASURED_COLUMNS = {
    "frequency_mhz": "Frequency",
    "pulse_width": "PW",
    "pri_us": "PRI",
    "rise_time_ns": "rise",
    "fall_time_ns": "fall",
    "bandwidth_hz": "timeBW",
}
DERIVED = ["duty_cycle_pct", "prf_hz", "pw_x_pri", "tb_product"]

# Dataset-generation range fields (meta.json "ranges") for each measured feature.
RANGE_FIELDS = {
    "frequency_mhz": ("freq_min_mhz", "freq_max_mhz"),
    "pulse_width": ("pw_min_us", "pw_max_us"),
    "pri_us": ("pri_min_us", "pri_max_us"),
    "rise_time_ns": ("rise_min_ns", "rise_max_ns"),
    "fall_time_ns": ("fall_min_ns", "fall_max_ns"),
    "bandwidth_hz": ("bw_min_mhz", "bw_max_mhz"),  # configured in MHz, see RANGE_SCALE
}

RANGE_SCALE = {"bandwidth_hz": 1e6}  # configured ranges are stored in MHz; the column is in Hz

BAND_EDGES_MHZ = [("L-BAND", 1000, 2000), ("S-BAND", 2000, 4000), ("C-BAND", 4000, 8000),
                  ("X-BAND", 8000, 12000), ("KU-BAND", 12000, 18000), ("K-BAND", 18000, 27000),
                  ("KA-BAND", 27000, 40000)]

GROUP_KNOWN, GROUP_HOLDOUT, GROUP_NOISE = "known", "holdout", "noise"


def dataset_column_features(df: pd.DataFrame) -> List[str]:
    """Extra numeric columns of a track file (beyond the measured ones) that can be analysed."""
    return [c for c in sequence_dataset_columns(df)
            if c not in MEASURED_COLUMNS.values() and c != "Frequency"]


def analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every numeric PDW feature under its canonical analysis name: measured columns, derived
    radar quantities (duty cycle %, PRF, PW×PRI, time-bandwidth product) and any extra numeric
    columns of the track file. Rise/fall zeros become NaN (not measured)."""
    sub = pd.DataFrame(index=df.index)
    for name, col in MEASURED_COLUMNS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].astype(float)
            sub[name] = values.mask(values == 0) if col in ZERO_IS_MISSING else values
    with np.errstate(divide="ignore", invalid="ignore"):
        if {"pulse_width", "pri_us"} <= set(sub.columns):
            sub["duty_cycle_pct"] = (sub["pulse_width"] / sub["pri_us"] * 100).replace([np.inf, -np.inf], np.nan)
            sub["prf_hz"] = (1e6 / sub["pri_us"]).replace([np.inf, -np.inf], np.nan)
            sub["pw_x_pri"] = sub["pulse_width"] * sub["pri_us"]
        if {"pulse_width", "bandwidth_hz"} <= set(sub.columns):
            # µs × MHz = dimensionless time-bandwidth product: ≈1 for a simple pulse, ≫1 for a
            # pulse-compressed (chirped/coded) waveform.
            sub["tb_product"] = sub["pulse_width"] * sub["bandwidth_hz"] / 1e6
    for col in dataset_column_features(df):
        sub[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return sub


def row_groups(df: pd.DataFrame, label_col: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    """(class label per row as str, group per row: known / holdout / noise). Unknown rows
    (true_class_id = -1) are split by their label into held-out classes vs. unstructured noise."""
    labels = df[label_col].astype(str) if label_col else pd.Series(["all"] * len(df), index=df.index)
    if "true_class_id" in df.columns:
        unknown = df["true_class_id"].astype(int) < 0
        noise = labels.str.contains("NOISE", case=False)
        groups = np.where(~unknown, GROUP_KNOWN, np.where(noise, GROUP_NOISE, GROUP_HOLDOUT))
    else:
        groups = np.full(len(df), GROUP_KNOWN)
    return labels, pd.Series(groups, index=df.index)


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 1:
        return f"{x:.3g}"
    return f"{x:.2g}"


def _edges(values: np.ndarray, bins: int, log: bool) -> Optional[np.ndarray]:
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return None
    if log:
        if (v <= 0).any():
            return None
        lo, hi = v.min(), v.max()
        if hi <= lo:
            hi = lo * 1.01
        return np.geomspace(lo, hi, bins + 1)
    lo, hi = v.min(), v.max()
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1)


def _counts(values: np.ndarray, edges: np.ndarray) -> List[int]:
    v = values[~np.isnan(values)]
    return [int(c) for c in np.histogram(v, bins=edges)[0]]


def _histogram_block(values: np.ndarray, labels: pd.Series, groups: pd.Series, top_classes: List[str],
                     bins: int, log: bool) -> Optional[dict]:
    edges = _edges(values, bins, log)
    if edges is None:
        return None
    lab, grp = labels.to_numpy(), groups.to_numpy()
    block = {
        "labels": [f"{_fmt(edges[i])}–{_fmt(edges[i + 1])}" for i in range(len(edges) - 1)],
        "edges": [float(e) for e in edges],
        "total": _counts(values, edges),
        "by_class": {c: _counts(values[lab == c], edges) for c in top_classes},
        "by_group": {},
    }
    for g in (GROUP_KNOWN, GROUP_HOLDOUT, GROUP_NOISE):
        mask = grp == g
        if mask.any():
            block["by_group"][g] = _counts(values[mask], edges)
    return block


def feature_distributions(sub: pd.DataFrame, labels: pd.Series, groups: pd.Series,
                          max_classes: int = 10, bins: int = 30) -> dict:
    """Per feature: linear and (all-positive features only) log-binned histograms — overall,
    per class (the `max_classes` most populous known classes) and per known/holdout/noise group.
    Log bins matter for PRI, PRF, PW and bandwidth, which span orders of magnitude."""
    known_labels = labels[groups == GROUP_KNOWN]
    top_classes = known_labels.value_counts().index[:max_classes].tolist()
    out = {}
    for col in sub.columns:
        values = sub[col].to_numpy(dtype=float)
        out[col] = {
            "linear": _histogram_block(values, labels, groups, top_classes, bins, log=False),
            "log": _histogram_block(values, labels, groups, top_classes, bins, log=True),
        }
    return {
        "features": out,
        "classes": top_classes,
        "classes_truncated": known_labels.nunique() > max_classes,
        "class_totals": {c: int((labels == c).sum()) for c in top_classes},
        "group_totals": {g: int((groups == g).sum()) for g in (GROUP_KNOWN, GROUP_HOLDOUT, GROUP_NOISE)},
    }


def feature_ranking(sub: pd.DataFrame, labels: pd.Series, groups: pd.Series,
                    max_rows: int = 6000) -> List[dict]:
    """How useful each feature is on its own, computed on its measured rows:
      - anova_f: between-class vs within-class variance (higher = class means further apart);
      - mi_norm: mutual information with the class label / ln(#classes), 0..1 — also catches
        non-linear, multi-modal separation the F-score misses;
      - ks_unknown: Kolmogorov-Smirnov distance between known and unknown (holdout + noise)
        rows, 0..1 — how differently unknown emitters look on this feature (open-set value).
    Sorted by mi_norm, highest first."""
    from scipy.stats import ks_2samp
    from sklearn.feature_selection import f_classif, mutual_info_classif

    known = groups == GROUP_KNOWN
    unknown = ~known
    rows = []
    for col in sub.columns:
        values = sub[col]
        k_mask = known & values.notna()
        y = labels[k_mask]
        x = values[k_mask]
        entry = {"feature": col, "coverage": round(float(values.notna().mean()), 4),
                 "anova_f": None, "mi_norm": None, "ks_unknown": None}
        n_classes = y.nunique()
        if n_classes >= 2 and len(x) >= 10 and x.nunique() > 1:
            if len(x) > max_rows:
                idx = x.sample(n=max_rows, random_state=0).index
                x, y = x.loc[idx], y.loc[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f, _ = f_classif(x.to_numpy().reshape(-1, 1), y.to_numpy())
                mi = mutual_info_classif(x.to_numpy().reshape(-1, 1), y.to_numpy(), random_state=0)
            if np.isfinite(f[0]):
                entry["anova_f"] = round(float(f[0]), 2)
            entry["mi_norm"] = round(float(min(1.0, mi[0] / np.log(n_classes))), 4)
        u = values[unknown & values.notna()]
        kn = values[known & values.notna()]
        if len(u) >= 5 and len(kn) >= 5:
            entry["ks_unknown"] = round(float(ks_2samp(kn, u).statistic), 4)
        rows.append(entry)
    return sorted(rows, key=lambda r: (r["mi_norm"] is None, -(r["mi_norm"] or 0)))


def rf_spectrum(sub: pd.DataFrame, bins: int = 120) -> Optional[dict]:
    """Pulse count vs carrier frequency in fine bins, each bin tagged with its radar band —
    the classic RF spectrum occupancy view."""
    if "frequency_mhz" not in sub.columns:
        return None
    values = sub["frequency_mhz"].to_numpy(dtype=float)
    edges = _edges(values, bins, log=False)
    if edges is None:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "labels": [_fmt(c) for c in centers],
        "counts": _counts(values, edges),
        "bands": [frequency_to_band(c) for c in centers],
        "band_edges": [{"band": b, "min": lo, "max": hi} for b, lo, hi in BAND_EDGES_MHZ],
    }


def mode_by_class(df: pd.DataFrame, labels: pd.Series, groups: pd.Series, max_classes: int = 15) -> Optional[dict]:
    """Operating-mode mix (search/acquisition/track/illumination) per known class."""
    if "operating_mode" not in df.columns:
        return None
    known = groups == GROUP_KNOWN
    top = labels[known].value_counts().index[:max_classes].tolist()
    sel = known & labels.isin(top)
    table = pd.crosstab(labels[sel], df.loc[sel, "operating_mode"].astype(str))
    if table.empty:
        return None
    table = table.reindex(top).fillna(0).astype(int)
    return {"classes": top, "modes": list(table.columns), "counts": table.to_numpy().T.tolist()}


def _outlier_count(values: np.ndarray) -> int:
    v = values[~np.isnan(values)]
    if len(v) < 4:
        return 0
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())


def data_quality(df: pd.DataFrame, labels: pd.Series, groups: pd.Series, max_classes: int = 15) -> dict:
    """Per measured/extra column: missing (blank), zero, and Tukey outlier counts, plus each
    column's missing-or-zero rate per class — to spot measurement gaps that hit some classes
    more than others (which a model could otherwise learn as a false class signal)."""
    columns = [c for c in MEASURED_COLUMNS.values() if c in df.columns] + dataset_column_features(df)
    n = len(df)
    overall = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        missing = int(np.isnan(values).sum())
        zeros = int((values == 0).sum())
        measured = values.copy()
        if col in ZERO_IS_MISSING:
            measured[measured == 0] = np.nan
        valid = measured[~np.isnan(measured)]
        overall.append({
            "column": col,
            "missing": missing, "missing_pct": round(missing / n, 4) if n else 0.0,
            "zeros": zeros, "zero_pct": round(zeros / n, 4) if n else 0.0,
            "zero_is_missing": col in ZERO_IS_MISSING,
            "outliers": _outlier_count(measured), "outlier_pct": round(_outlier_count(measured) / max(1, len(valid)), 4),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
        })

    affected = [q["column"] for q in overall if q["missing"] or (q["zero_is_missing"] and q["zeros"])]
    top = labels[groups == GROUP_KNOWN].value_counts().index[:max_classes].tolist()
    row_keys = top + [g for g in (GROUP_HOLDOUT, GROUP_NOISE) if (groups == g).any()]
    by_class = []
    for key in row_keys:
        mask = (groups == key) if key in (GROUP_HOLDOUT, GROUP_NOISE) else ((labels == key) & (groups == GROUP_KNOWN))
        rates = {}
        for col in affected:
            values = pd.to_numeric(df.loc[mask, col], errors="coerce")
            gap = values.isna() | ((values == 0) if col in ZERO_IS_MISSING else False)
            rates[col] = round(float(gap.mean()), 4) if len(values) else None
        by_class.append({"class": key, "rows": int(mask.sum()), "rates": rates})
    return {"overall": overall, "affected_columns": affected, "by_class": by_class}


def dataset_overview(df: pd.DataFrame, label_col: Optional[str], meta: Optional[dict], bins: int = 20) -> dict:
    """The Dataset page's health check: composition, a small histogram + min/median/max per
    feature (with the configured generation range when known), and data quality."""
    labels, groups = row_groups(df, label_col)
    sub = analysis_frame(df)
    ranges = (meta or {}).get("ranges", {})
    tiles = []
    for col in sub.columns:
        values = sub[col].to_numpy(dtype=float)
        valid = values[~np.isnan(values)]
        edges = _edges(values, bins, log=False)
        configured = None
        if col in RANGE_FIELDS and ranges.get(RANGE_FIELDS[col][0]) is not None:
            scale = RANGE_SCALE.get(col, 1.0)
            configured = [ranges[RANGE_FIELDS[col][0]] * scale, ranges[RANGE_FIELDS[col][1]] * scale]
        tiles.append({
            "feature": col,
            "derived": col in DERIVED,
            "count": int(len(valid)),
            "min": float(valid.min()) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "configured_range": configured,
            "histogram": {"labels": [f"{_fmt(edges[i])}–{_fmt(edges[i + 1])}" for i in range(len(edges) - 1)],
                          "counts": _counts(values, edges)} if edges is not None else None,
        })
    known = groups == GROUP_KNOWN
    return {
        "composition": {
            "total_rows": int(len(df)),
            "known_classes": int(labels[known].nunique()) if label_col else None,
            "known_rows": int(known.sum()),
            "holdout_rows": int((groups == GROUP_HOLDOUT).sum()),
            "noise_rows": int((groups == GROUP_NOISE).sum()),
        },
        "tiles": tiles,
        "quality": data_quality(df, labels, groups)["overall"],
    }
