"""Track (whole-track) mode end to end on tracks of DIFFERENT lengths (exercises padding + mask):
loader -> training -> evaluation -> SequenceClassifier. Self-contained (generates its own dataset).
Run: python tests/test_track_mode.py  (or pytest)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, torch, yaml
from src.data.sequence_features import (DEFAULT_SEQ_FEATURES, cv_split, load_sequence_dataset_for_training, required_seq_columns,
                                         upgrade_columns, SequenceSpec)
from src.data.sequence_generator import build_sequence_dataset_rows
from src.models.sequence_eval import SequenceClassifier, calibrate_distance_threshold, evaluate_tracks
from src.models.sequence_models import SequenceDualHeadModel
from src.training.trainer import build_distance_fn, seed_training, train_model

ROOT = os.path.join(os.path.dirname(__file__), "..")
NUM_CLASSES = 4


def _variable_length_dataset() -> pd.DataFrame:
    df = upgrade_columns(pd.DataFrame(build_sequence_dataset_rows(NUM_CLASSES, [3], 10, 90, 7, None, 0.02, 0.02, 0.2,
                                                                  None, False, True)))
    rng = np.random.RandomState(1)
    keep = [t.sort_values("pulse_idx").iloc[:rng.randint(20, 81)]   # every track cut to a random length 20..80
            for _, t in df.groupby("instance", sort=False)]
    return pd.concat(keep, ignore_index=True)


def test_track_mode_end_to_end(verbose: bool = False):
    df = _variable_length_dataset()
    names = [n for n in DEFAULT_SEQ_FEATURES if all(c in df.columns for c in required_seq_columns([n]))]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    cfg["model"]["num_classes"] = NUM_CLASSES
    cfg["model"]["embedding_dim"] = 32
    cfg["hyperparameters"].setdefault("rpl_loss_weight", 0.1)
    test_ids, folds = cv_split(df, 7, 20.0, 3, 0.1, False)
    dist = build_distance_fn("cosine")

    for mode, cap in (("track", 128), ("track", 0), ("track", 40), ("windows", 16)):
        ds = load_sequence_dataset_for_training(df, names, cap, 7, 0.2, None, 0.1, folds[0], mode)
        spec = ds["feature_spec"]
        padded = ds["train_x"].shape[-1] == len(names) + 1
        seed_training(0)
        model = SequenceDualHeadModel("seq_hybrid", spec.dim, cfg["model"]["embedding_dim"], NUM_CLASSES)
        m = train_model(model, ds, cfg, 2, 64, None, 0)
        cents = m.pop("centroids")
        cal = calibrate_distance_threshold(model, ds["cal_x"] if len(ds["cal_x"]) >= 5 else ds["val_x"], cents, dist, 0.95,
                                           ds.get("cal_short_x"))
        ev = evaluate_tracks(model, spec, df[df["instance"].isin(test_ids)], cents, dist, 0.88, cal["ood_distance_threshold"])
        clf = SequenceClassifier(model, SequenceSpec.from_dict(spec.to_dict()), cents, dist, 0.88, cal["ood_distance_threshold"],
                                 cal.get("by_length"), cal["ood_distance_threshold"])
        r = clf.classify_track(df[df["instance"] == sorted(test_ids, key=str)[0]])
        if verbose:
            print(f"{mode:7s} cap={cap:3d} -> L={spec.window:3d} padded={padded} train {tuple(ds['train_x'].shape)} "
                  f"windows/track in classify={r['num_windows']} track_acc={ev['track_accuracy']}")
        if mode == "track":
            assert r["num_windows"] == 1 and spec.mode == "track" and spec.window <= 512
            assert padded, "tracks of different lengths must be padded with a mask channel"
        else:
            assert not padded and r["num_windows"] >= 1


if __name__ == "__main__":
    test_track_mode_end_to_end(verbose=True)
    print("OK")
