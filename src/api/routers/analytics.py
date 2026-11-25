"""Dataset analysis for the Analytics and Dataset pages."""

import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.analytics.dataset_analysis import (
    analysis_frame, data_quality, dataset_overview, feature_distributions, feature_ranking, mode_by_class,
    rf_spectrum, row_groups,
)
from src.data.generator import frequency_to_band
from src.data.sequence_features import upgrade_columns
from src.storage.training_store import get_run
from src.api.core import _public_error, datasets
from src.api.common import _dataset_kind, _ist, _kept_sequence_runs, _model_dataset
router = APIRouter()


_LABEL_COLUMN_CANDIDATES = ["class", "class_id"]
CORRELATION_HIDDEN = {"prf_hz", "pw_x_pri", "tb_product"}

def _pri_pattern_check(df: pd.DataFrame) -> Optional[dict]:
    """Does each family's PRI behave like its pri_pattern label? Uses the generator's ground-truth
    pri_deviation_pct (PRI relative to the operating mode's base PRI), on clean pulses of labelled
    families only (no missed / spurious pulse involved). None for files without these columns."""
    if not {"pri_pattern", "pri_deviation_pct"} <= set(df.columns):
        return None
    clean = pd.to_numeric(df["pri_deviation_pct"], errors="coerce").notna()
    for col in ("missed_before", "spurious_before", "is_spurious"):
        if col in df.columns:
            clean &= pd.to_numeric(df[col], errors="coerce").fillna(0) == 0
    if "class" in df.columns:
        clean &= ~df["class"].astype(str).str.contains("NOISE", case=False)
    sub = df[clean]
    if sub.empty:
        return None
    dev = pd.to_numeric(sub["pri_deviation_pct"])
    edges = np.arange(-40, 42, 2.0)
    patterns = []
    for pat, g in dev.groupby(sub["pri_pattern"].astype(str)):
        counts, _ = np.histogram(g.clip(-40, 40), bins=edges)
        patterns.append({"pattern": pat, "pulses": int(len(g)), "std": round(float(g.std(ddof=0)), 2),
                         "min": round(float(g.min()), 2), "max": round(float(g.max()), 2),
                         "share": [round(float(c / len(g)), 4) for c in counts]})
    label_col = "class" if "class" in sub.columns else None
    families = []
    if label_col:
        # by label AND pattern: held-out rows share one label but come from several families
        for (lab, pat), g in sub.groupby([sub[label_col].astype(str), sub["pri_pattern"].astype(str)]):
            d = pd.to_numeric(g["pri_deviation_pct"])
            families.append({"label": lab, "pattern": pat, "pulses": int(len(g)),
                             "std": round(float(d.std(ddof=0)), 2), "min": round(float(d.min()), 2),
                             "max": round(float(d.max()), 2), "levels": int(d.round(0).nunique())})
    return {"bins": [round(float((a + b) / 2), 1) for a, b in zip(edges[:-1], edges[1:])],
            "patterns": patterns, "families": families,
            "excluded_disturbed": int((~clean).sum())}


def _pri_traces(df: pd.DataFrame, max_groups: int = 12, max_pulses: int = 160) -> Optional[dict]:
    """PRI against pulse number for one example track per PRI pattern (per class label when the file has
    no pattern column): what the pattern looks like in time, and where missed / spurious pulses make
    the PRI jump (flagged). Labelled families only — noise tracks have no pattern."""
    if not {"instance", "PRI"} <= set(df.columns):
        return None
    key = "pri_pattern" if "pri_pattern" in df.columns else ("class" if "class" in df.columns else None)
    if key is None:
        return None
    first = df.groupby("instance", sort=False).first()
    if "true_class_id" in first.columns:
        first = first[pd.to_numeric(first["true_class_id"], errors="coerce").fillna(-1) >= 0]
    traces = []
    for group, tracks in first.groupby(first[key].astype(str), sort=True):
        if len(traces) >= max_groups:
            break
        tid = tracks.index[0]
        t = df[df["instance"] == tid]
        t = (t.sort_values("pulse_idx") if "pulse_idx" in t.columns else t).head(max_pulses)
        flags = np.zeros(len(t), dtype=bool)
        for col in ("missed_before", "spurious_before", "is_spurious"):
            if col in t.columns:
                flags |= pd.to_numeric(t[col], errors="coerce").fillna(0).to_numpy() > 0
        label = str(tracks["class"].iloc[0]) if "class" in tracks.columns else str(group)
        pri = pd.to_numeric(t["PRI"], errors="coerce")
        dev = pd.to_numeric(t["pri_deviation_pct"], errors="coerce") if "pri_deviation_pct" in t.columns else None
        traces.append({"group": group, "label": label, "track_id": str(tid),
                       "pri_us": [None if pd.isna(v) else round(float(v), 3) for v in pri],
                       "deviation_pct": None if dev is None else [None if pd.isna(v) else round(float(v), 2) for v in dev],
                       "disturbed": [int(i) for i in np.flatnonzero(flags)]})
    return {"by": key, "traces": traces} if traces else None


def _analysis_source(source: str, filename: Optional[str]):
    """(dataframe, display name, meta) for an Analytics/Dataset-page selection, or a JSON error response.
    source "dataset": a named dataset (blank = the active one); source "model": the dataset of kept run <filename>.
    The old source names "sequence" / "sequence_uploaded" still mean "a named dataset"."""
    if source == "model":
        try:
            r = get_run(int(filename))
        except (TypeError, ValueError):
            r = None
        if r is None or r.get("model_kind") != "sequence":
            return JSONResponse({"status": "error", "message": "Model not found."}, status_code=404)
        try:
            df, meta = _model_dataset(r)
        except ValueError as e:
            return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=409)
        if df is None:
            return JSONResponse({"status": "error", "message": f"Run #{r['id']} was trained on a dataset that is no longer stored."}, status_code=404)
        return df, meta.get("name") or f"run #{r['id']}", meta
    name = filename or datasets.get_active()
    if not name or not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "No dataset found — generate or upload one on the Dataset page."}, status_code=404)
    df, meta = datasets.load(name)
    return df, name, meta


@router.get("/api/analytics/sources")
def analytics_sources():
    """Everything Analytics can show: every stored dataset (generated or uploaded) and the dataset behind each kept model."""
    sources = []
    for m in datasets.list():
        sources.append({"source": "dataset", "filename": m["name"], "group": "Datasets",
                        "label": f"{m['name']} — {_dataset_kind(m)}, {m.get('total_tracks', '?')} tracks",
                        "modified": _ist(m.get("created_at_utc"))})
    for r in _kept_sequence_runs():
        name = r.get("dataset_name") or f"regenerated from seed {r.get('dataset_seed')}"
        title = r.get("saved_name") or f"run #{r['id']}"
        sources.append({"source": "model", "filename": str(r["id"]), "group": "Dataset of a trained model",
                        "label": f"{title} ({r['backbone_type']}) → {name}", "modified": r.get("timestamp_ist", "")})
    return JSONResponse({"status": "success", "sources": sources, "active": datasets.get_active()})


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Single shared candidate list for the ground-truth/class label column — shared by
    inspect_dataset and dataset_analytics."""
    return next((c for c in _LABEL_COLUMN_CANDIDATES if c in df.columns), None)


@router.get("/api/dataset/inspect")
def inspect_dataset(source: str = "dataset", filename: str = None):
    found = _analysis_source(source, filename)
    if isinstance(found, JSONResponse):
        return found
    df, name, meta = found
    df = upgrade_columns(df)  # older track files: Hz frequency column -> MHz
    label_col = _detect_label_column(df)
    class_counts = df[label_col].value_counts().to_dict() if label_col else {}
    # A generated dataset's meta carries the configured generation ranges, shown next to each
    # feature's actual spread so a wrong range setting is obvious at a glance. An uploaded
    # file was not generated from ranges, so it has none to show.
    ranges_meta = meta if meta.get("source") == "generated" else None
    return JSONResponse({
        "status": "success",
        "filename": name,
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "label_column": label_col,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        **dataset_overview(df, label_col, ranges_meta),
    })


def _tukey_box_stats(values: np.ndarray) -> Optional[dict]:
    """Standard Tukey boxplot: box at Q1/median/Q3, whiskers extended to the furthest point
    still within 1.5*IQR of the box, everything beyond flagged as an outlier."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = values[(values >= lo_fence) & (values <= hi_fence)]
    whisker_lo = float(inliers.min()) if len(inliers) else float(values.min())
    whisker_hi = float(inliers.max()) if len(inliers) else float(values.max())
    outliers = sorted(values[(values < whisker_lo) | (values > whisker_hi)].tolist())
    return {
        "min": whisker_lo, "q1": float(q1), "median": float(median), "q3": float(q3), "max": whisker_hi,
        "outliers": outliers[:25],
    }


def _pca_2d(matrix: np.ndarray, labels: Optional[List[str]] = None, max_points: int = 800) -> Optional[dict]:
    """Standardizes and projects `matrix` to 2D via PCA — the standard first-look for whether
    classes are actually separable in a given feature space, rather than assuming it from
    summary statistics alone. Subsamples for plotting if there are many more rows than
    max_points (PCA itself still fits on the full matrix)."""
    valid = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid]
    if labels is not None:
        labels = [l for l, v in zip(labels, valid) if v]
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / std
    n_components = min(2, matrix.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(standardized)
    if n_components == 1:
        projected = np.concatenate([projected, np.zeros_like(projected)], axis=1)

    idx = np.arange(len(projected))
    if len(idx) > max_points:
        idx = np.random.RandomState(0).choice(idx, size=max_points, replace=False)

    points = [{"x": float(projected[i, 0]), "y": float(projected[i, 1]),
               "label": str(labels[i]) if labels is not None else None} for i in idx]
    explained = pca.explained_variance_ratio_.tolist()
    return {"points": points, "explained_variance_ratio": [round(float(v), 4) for v in explained]}


SILHOUETTE_MAX_ROWS = 5000


def _silhouette(matrix: np.ndarray, labels: List[str]) -> Optional[float]:
    """Single-number complement to the PCA scatter: how well-separated classes actually are
    in a feature space (+1 = tight, well-separated clusters; 0 = overlapping; negative =
    likely mislabeled/worse than random). Needs at least 2 classes with 2+ samples each."""
    valid = ~np.isnan(matrix).any(axis=1)
    matrix, labels = matrix[valid], [l for l, v in zip(labels, valid) if v]
    unique = set(labels)
    if len(unique) < 2 or matrix.shape[0] < 4:
        return None
    counts = {u: labels.count(u) for u in unique}
    if any(c < 2 for c in counts.values()):
        return None
    # Exact silhouette is O(n²): fine for a few thousand rows, minutes for a 100k-pulse sequence
    # dataset. Above SILHOUETTE_MAX_ROWS it's estimated on a fixed random sample (same seed each
    # time, so the number is reproducible).
    sample = {"sample_size": SILHOUETTE_MAX_ROWS, "random_state": 0} if matrix.shape[0] > SILHOUETTE_MAX_ROWS else {}
    try:
        return round(float(silhouette_score(matrix, labels, **sample)), 4)
    except Exception:
        return None


@router.get("/api/analytics")
def dataset_analytics(source: str = "dataset", filename: str = None,
                      max_boxplot_classes: int = Query(15, ge=1, le=50), max_scatter_rows: int = Query(600, ge=1, le=5000)):
    # plain def (not async): FastAPI runs it in a worker thread, so a big file doesn't stall the server
    """One-shot EDA payload for the Analytics page: summary statistics, a Pearson correlation
    matrix, per-feature histograms, per-class Tukey boxplots, and a row sample for the scatter
    explorer — computed once per dataset selection so the page's plot buttons just toggle
    which already-fetched view is shown, rather than re-hitting the server per click."""
    found = _analysis_source(source, filename)
    if isinstance(found, JSONResponse):
        return found
    df, filename, _meta = found
    df = upgrade_columns(df)  # older track files: Hz frequency column -> MHz
    label_col = _detect_label_column(df)
    class_counts_full = df[label_col].value_counts() if label_col else pd.Series(dtype=int)

    # Every measured, derived (duty cycle, PRF, PW×PRI, time-bandwidth product) and extra
    # (dataset/synthetic) feature; rise/fall zeros are "not measured" and excluded here —
    # they're reported in data_quality instead.
    sub = analysis_frame(df)
    if sub.empty:
        return JSONResponse({"status": "error", "message": "No recognized numeric PDW columns in this file."}, status_code=400)
    labels, groups = row_groups(df, label_col)

    describe = {}
    for col in sub.columns:
        vals = sub[col].to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        describe[col] = {
            "count": int(len(valid)), "missing": int(len(vals) - len(valid)),
            "mean": float(np.mean(valid)), "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)), "p25": float(np.percentile(valid, 25)),
            "median": float(np.median(valid)), "p75": float(np.percentile(valid, 75)),
            "max": float(np.max(valid)),
        }

    # The derived columns that only restate PRI / PW (PRF = 1/PRI, PW×PRI, time-bandwidth product) are left
    # out of the heatmap: they would show up as near-perfect correlations with their own inputs.
    corr_cols = [c for c in sub.columns if c not in CORRELATION_HIDDEN]
    corr_df = sub[corr_cols].corr(numeric_only=True).round(4)
    correlation = {
        "columns": list(corr_df.columns),
        "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in corr_df.to_numpy()],
    }

    distributions = feature_distributions(sub, labels, groups)
    ranking = feature_ranking(sub, labels, groups) if label_col else []
    quality = data_quality(df, labels, groups)
    spectrum = rf_spectrum(sub)
    modes = mode_by_class(df, labels, groups) if label_col else None
    boxplot = {}
    classes_truncated = False
    if label_col:
        top_classes = class_counts_full.index.tolist()
        if len(top_classes) > max_boxplot_classes:
            top_classes = top_classes[:max_boxplot_classes]
            classes_truncated = True
        label_values = df[label_col].to_numpy()
        for col in sub.columns:
            col_values = sub[col].to_numpy(dtype=float)
            classes_here, stats_here = [], []
            for cls in top_classes:
                stats = _tukey_box_stats(col_values[label_values == cls])
                if stats:
                    classes_here.append(str(cls))
                    stats_here.append(stats)
            if stats_here:
                boxplot[col] = {"classes": classes_here, "stats": stats_here}

    sample_df = sub.copy()
    if label_col:
        sample_df["_label"] = df[label_col].astype(str).to_numpy()
    if len(sample_df) > max_scatter_rows:
        sample_df = sample_df.sample(n=max_scatter_rows, random_state=0)
    # object dtype first: on float columns pandas turns None back into NaN, which isn't valid JSON
    scatter_sample = sample_df.astype(object).where(pd.notnull(sample_df), None).to_dict("records")

    # Frequency band breakdown — the classic first-pass ESM/EW analysis (how much of the
    # intercepted spectrum falls in which band), using the same L/S/C/X/Ku/K/Ka mapping the
    # training pipeline uses for its prototype metadata.
    band_distribution = None
    if "frequency_mhz" in sub.columns:
        bands = sub["frequency_mhz"].dropna().apply(frequency_to_band)
        band_distribution = {str(k): int(v) for k, v in bands.value_counts().items()}

    # PRI modulation pattern breakdown — only meaningful now that the generator actually
    # produces more than one pattern (see src/data/pri_patterns.py); shows which of the six
    # standard types (constant/jittered/staggered/sliding/wobulated/dwell_switch) the current
    # dataset's families/emitters are using.
    pri_pattern_distribution = None
    if "pri_pattern" in df.columns:
        pri_pattern_distribution = {str(k): int(v) for k, v in df["pri_pattern"].value_counts().items()}
    pri_pattern_check = _pri_pattern_check(df)
    pri_traces = _pri_traces(df)

    # Class balance as a single number, not just the bar chart — Shannon entropy (normalized
    # to [0,1], 1.0 = perfectly even) and the max/min count ratio, standard dataset-health
    # metrics for any classification task.
    class_diversity = None
    if label_col and len(class_counts_full) > 1:
        counts = class_counts_full.to_numpy(dtype=float)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        max_entropy = float(np.log(len(counts)))
        class_diversity = {
            "shannon_entropy_normalized": round(entropy / max_entropy, 4) if max_entropy > 0 else None,
            "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
            "num_classes": int(len(counts)),
        }

    # PCA + silhouette in the RAW physical feature space — directly visualizes/quantifies
    # whether classes are actually separable before any model gets involved, the standard
    # first question in any classification EDA.
    # Features missing for more than 20% of rows (e.g. rise time recorded as 0) are left out
    # here, since PCA needs complete rows and would otherwise drop most of the dataset.
    label_values_list = df[label_col].astype(str).tolist() if label_col else None
    pca_columns = [c for c in sub.columns if sub[c].isna().mean() <= 0.2]
    pca_matrix = sub[pca_columns].to_numpy(dtype=float)
    pca_raw = _pca_2d(pca_matrix, label_values_list)
    if pca_raw is not None:
        pca_raw["columns"] = pca_columns
    silhouette_raw = _silhouette(pca_matrix, label_values_list) if label_values_list else None

    return JSONResponse({
        "status": "success",
        "filename": filename,
        "total_rows": int(len(df)),
        "label_column": label_col,
        "class_counts": {str(k): int(v) for k, v in class_counts_full.items()},
        "features": list(sub.columns),
        "describe": describe,
        "correlation": correlation,
        "distributions": distributions,
        "feature_ranking": ranking,
        "data_quality": quality,
        "rf_spectrum": spectrum,
        "mode_by_class": modes,
        "boxplot": boxplot,
        "boxplot_classes_truncated": classes_truncated,
        "scatter_sample": scatter_sample,
        "band_distribution": band_distribution,
        "pri_pattern_distribution": pri_pattern_distribution,
        "pri_pattern_check": pri_pattern_check,
        "pri_traces": pri_traces,
        "class_diversity": class_diversity,
        "pca_raw": pca_raw,
        "silhouette_raw": silhouette_raw,
    })
