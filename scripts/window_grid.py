"""Window-length x backbone comparison on ONE dataset, ONE seed, the SAME splits.

For every (input setting, backbone) it trains one model per cross-validation fold and evaluates it on the
SAME held-out test tracks, so the cells are directly comparable. Input settings:
    k<N>        overlapping windows of N pulses, voted per track          (e.g. k16)
    track<N>    no windows: ONE input per track = its first N pulses       (e.g. track128)
    track0      no windows, no cap (whole track, up to 512 pulses)
Writes results.csv / results.json / results.md to data/benchmark_results/window_grid_<time>/.

Metrics are compared at the TRACK level (window-level numbers are not comparable across window lengths).
All numbers are mean ± std over the folds; "test" = the held-out test split, never trained on.

Examples (run from the project root):
    python scripts/window_grid.py                                  # defaults: active dataset, 4 backbones, 7 settings
    python scripts/window_grid.py --windows k16,k32,track128 --backbones seq_cnn,seq_transformer --epochs 10 --folds 2
    python scripts/window_grid.py --quick                          # tiny smoke run (1 fold, 3 epochs)
"""
import argparse, copy, json, os, sys, time
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.sequence_features import (DEFAULT_SEQ_FEATURES, cv_split, load_sequence_dataset_for_training, required_seq_columns,
                                         upgrade_columns)
from src.models.sequence_eval import calibrate_distance_threshold, evaluate_tracks
from src.models.sequence_models import SEQ_BACKBONES, SequenceDualHeadModel
from src.storage.dataset_store import DatasetStore
from src.training.trainer import build_distance_fn, train_model

DEFAULT_SETTINGS = "k4,k8,k16,k32,k64,track128,track0"
METRICS = ["val_window_acc", "test_track_acc", "test_unknown_detected", "test_auroc_distance", "val_auroc_rpl",
           "test_acc_at_8_pulses", "train_seconds"]


def parse_setting(tag: str):
    """'k16' -> ('windows', 16); 'track128' -> ('track', 128); 'track0' -> ('track', 0)."""
    tag = tag.strip().lower()
    if tag.startswith("track"):
        return "track", int(tag[5:] or 0)
    if tag.startswith("k"):
        return "windows", int(tag[1:])
    raise SystemExit(f"Unknown setting '{tag}' (use k<N> or track<N>).")


def curve_value(curve, pulses):
    for e in curve or []:
        if e["pulses"] == pulses:
            return e["accuracy"]
    return None


def run_cell(df, names, mode, window, backbone, test_ids, fold_splits, seed, cfg, epochs, folds_to_run, num_classes):
    distance_fn = build_distance_fn(cfg["hyperparameters"]["distance_metric"])
    tau_c = cfg["hyperparameters"]["classifier_confidence_threshold"]
    test_df = df[df["instance"].isin(test_ids)]
    rows = []
    for f in folds_to_run:
        torch.manual_seed(seed + f)
        np.random.seed(seed + f)
        t0 = time.time()
        ds = load_sequence_dataset_for_training(df, names, window, seed, 0.2, None, 0.1, fold_splits[f], mode)
        spec = ds["feature_spec"]
        model = SequenceDualHeadModel(backbone, spec.dim, cfg["model"]["embedding_dim"], num_classes)
        m = train_model(model, ds, cfg, epochs, 64)
        centroids = m.pop("centroids")
        use_cal = len(ds["cal_x"]) >= 5
        cal = calibrate_distance_threshold(model, ds["cal_x"] if use_cal else ds["val_x"], centroids, distance_fn, 0.95,
                                           ds.get("cal_short_x"))
        seconds = time.time() - t0
        ev = evaluate_tracks(model, spec, test_df, centroids, distance_fn, tau_c, cal["ood_distance_threshold"])
        rows.append({
            "fold": f + 1, "resolved_window": spec.window,
            "val_window_acc": m["closed_set_accuracy"], "val_auroc_rpl": m.get("open_set_auroc_rpl"),
            "test_track_acc": ev.get("track_accuracy"), "test_unknown_detected": ev.get("track_unknown_detection"),
            "test_auroc_distance": ev.get("open_set_auroc"), "test_acc_at_8_pulses": curve_value(ev.get("pulses_curve"), 8),
            "train_seconds": round(seconds, 1), "known_tracks_without_input": ds.get("known_tracks_without_input", 0),
        })
        print(f"    fold {f + 1}: val {rows[-1]['val_window_acc']}  test track acc {rows[-1]['test_track_acc']}  "
              f"AUROC {rows[-1]['test_auroc_distance']}  {rows[-1]['train_seconds']}s", flush=True)
    return rows


def summarize(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def fmt(mean, std, pct=True):
    if mean is None:
        return "—"
    return f"{mean * 100:.1f}±{std * 100:.1f}" if pct else f"{mean:.1f}±{std:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="dataset name in data/sequence_datasets (default: the active one)")
    ap.add_argument("--backbones", default=",".join(SEQ_BACKBONES))
    ap.add_argument("--windows", default=DEFAULT_SETTINGS, help="comma list of k<N> / track<N> (default %(default)s)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=4, help="cross-validation folds (one model per fold per cell)")
    ap.add_argument("--test-pct", type=float, default=20.0)
    ap.add_argument("--features", help="comma list of feature names (default: all)")
    ap.add_argument("--seed", type=int, help="split/training seed (default: the dataset's own seed)")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--quick", action="store_true", help="smoke run: 3 epochs, the first fold only")
    args = ap.parse_args()

    store = DatasetStore(os.path.join(ROOT, "data", "sequence_datasets"))
    df, meta = store.load(args.dataset)
    if df is None:
        raise SystemExit("No sequence dataset found - generate one on the Dataset page first.")
    df = upgrade_columns(df)
    # default: every standard feature the dataset has the columns for (e.g. a file without SNR has no 'snr' feature)
    names = [n.strip() for n in args.features.split(",")] if args.features else         [n for n in DEFAULT_SEQ_FEATURES if all(c in df.columns for c in required_seq_columns([n]))]
    with open(os.path.join(ROOT, "config", "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    num_classes = meta["num_classes"]
    cfg = copy.deepcopy(cfg)
    cfg["model"]["num_classes"] = num_classes
    cfg["hyperparameters"].setdefault("rpl_loss_weight", 0.1)
    seed = args.seed if args.seed is not None else int(meta["dataset_seed"])
    epochs = 3 if args.quick else args.epochs
    test_ids, fold_splits = cv_split(df, seed, args.test_pct, args.folds, 0.1, False)
    folds_to_run = [0] if args.quick else list(range(args.folds))
    settings = [s.strip() for s in args.windows.split(",") if s.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for b in backbones:
        if b not in SEQ_BACKBONES:
            raise SystemExit(f"Unknown backbone '{b}'.")

    out_dir = args.out or os.path.join(ROOT, "data", "benchmark_results", "window_grid_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    lengths = df.groupby("instance").size()
    print(f"dataset {meta['name']}  seed {seed}  tracks {df['instance'].nunique()} (pulses/track {lengths.min()}–{lengths.max()})  "
          f"features {len(names)}  epochs {epochs}  folds {len(folds_to_run)}/{args.folds}  test tracks {len(test_ids)}")
    print(f"grid: {len(settings)} settings x {len(backbones)} backbones = {len(settings) * len(backbones)} cells -> {out_dir}\n", flush=True)

    results = []
    for tag in settings:
        mode, window = parse_setting(tag)
        for bb in backbones:
            print(f"[{tag}] {bb}", flush=True)
            try:
                rows = run_cell(df, names, mode, window, bb, test_ids, fold_splits, seed, cfg, epochs, folds_to_run, num_classes)
            except Exception as e:  # one failing cell must not lose the others
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                results.append({"setting": tag, "mode": mode, "window": window, "backbone": bb, "error": f"{type(e).__name__}: {e}"})
                continue
            cell = {"setting": tag, "mode": mode, "window": window, "backbone": bb, "folds": rows,
                    "resolved_window": rows[0]["resolved_window"],
                    "known_tracks_without_input": rows[0]["known_tracks_without_input"]}
            for k in METRICS:
                cell[k + "_mean"], cell[k + "_std"] = summarize(rows, k)
            results.append(cell)
            with open(os.path.join(out_dir, "results.json"), "w") as fh:  # saved after every cell
                json.dump({"dataset": meta["name"], "seed": seed, "epochs": epochs, "folds": len(folds_to_run),
                           "features": names, "results": results}, fh, indent=1, default=str)

    ok = [r for r in results if "error" not in r]
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "folds"} for r in ok])
    if len(flat):
        flat.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    lines = [f"# Window length x backbone — {meta['name']}", "",
             f"Seed {seed}, {epochs} epochs, {len(folds_to_run)} fold(s), same test split ({len(test_ids)} tracks) for every cell. "
             "Mean±std over folds, in %. Track-level metrics on the held-out test tracks.", ""]
    for title, key, pct in (("Track accuracy (known tracks, test split)", "test_track_acc", True),
                            ("Unknown tracks detected (test split)", "test_unknown_detected", True),
                            ("Open-set AUROC, distance (test split)", "test_auroc_distance", True),
                            ("Accuracy after only 8 pulses (test split)", "test_acc_at_8_pulses", True),
                            ("Training time per fold, seconds", "train_seconds", False)):
        lines += [f"## {title}", "", "| setting | " + " | ".join(backbones) + " |", "|---|" + "---|" * len(backbones)]
        for tag in settings:
            cells = []
            for bb in backbones:
                r = next((x for x in ok if x["setting"] == tag and x["backbone"] == bb), None)
                cells.append(fmt(r.get(key + "_mean"), r.get(key + "_std"), pct) if r else "failed")
            res = next((x["resolved_window"] for x in ok if x["setting"] == tag), "")
            lines.append(f"| {tag} ({res} pulses) | " + " | ".join(cells) + " |")
        lines.append("")
    dropped = {r["setting"]: r["known_tracks_without_input"] for r in ok}
    lines += ["Known tracks that give no input at all (shorter than the window / 2 pulses): " +
              ", ".join(f"{k}: {v}" for k, v in dropped.items()), ""]
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
