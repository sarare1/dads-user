"""Named track datasets on disk: data/sequence_datasets/<name>.csv + <name>.meta.json.

Every dataset — generated, uploaded for training, or uploaded for testing — is one such pair, so
the Dataset, Training, Testing and Analytics pages all pick from the same list and every trained
model can record which dataset it came from. One dataset is "active" (the Training page's default).

Lineage: a dataset's name is permanent. A deleted dataset's name is retired (_retired.json) and never
given to a new dataset, and names that training runs still refer to are never reused either. Each
dataset records the SHA-256 of its CSV (`sha256` in its meta), training runs store that hash, and
`verify` tells whether the stored file is still byte-for-byte the one a model was trained on.
"""
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.data.sequence_features import upgrade_columns

_ACTIVE_FILE = "_active.json"
_RETIRED_FILE = "_retired.json"


class DatasetStore:
    _CACHE_SIZE = 2  # parsed datasets kept in memory

    def __init__(self, directory: str, *,
                 reserved_names: Optional[Callable[[], Iterable[str]]] = None):
        self.dir = directory
        self._cache: "OrderedDict" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._write_lock = threading.Lock()   # choosing a name and claiming it is one step
        self._hashes: Dict[tuple, str] = {}   # (name, mtime_ns, size) -> sha256, so an unchanged file is hashed once
        self._reserved_names = reserved_names  # names still referenced elsewhere (training runs' dataset_name)
        os.makedirs(directory, exist_ok=True)

    # ------------------------------------------------------------------ paths / names
    def csv_path(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.csv")

    def meta_path(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.meta.json")

    def exists(self, name: Optional[str]) -> bool:
        return bool(name) and self.valid_name(name) and os.path.isfile(self.csv_path(name)) and os.path.isfile(self.meta_path(name))

    @staticmethod
    def valid_name(name: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}", name or ""))

    @staticmethod
    def slug(text: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", (text or "").strip()).strip("._-")
        return s[:60] or "dataset"

    def unique_name(self, wanted: str) -> str:
        """`wanted` (as a slug), or with _2, _3 ... appended: never the name of a stored, deleted (retired)
        or still-referenced dataset, so a name always means the same data."""
        # Compared without case: Windows file names ignore it, so 'Tracks' and 'tracks' are one file there.
        stored = {f[:-len(suffix)].lower() for f in os.listdir(self.dir)
                  for suffix in (".csv", ".meta.json") if f.endswith(suffix)}
        taken = stored | set(self._retired()) | {n.lower() for n in (self._reserved_names() if self._reserved_names else []) if n}
        base, n, name = self.slug(wanted), 2, self.slug(wanted)
        while name.lower() in taken:
            name, n = f"{base}_{n}", n + 1
        return name

    # ------------------------------------------------------------------ content hash / lineage
    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def content_hash(self, name: str) -> Optional[str]:
        """SHA-256 of the dataset's CSV as it is on disk now (cached while the file is unchanged)."""
        if not self.exists(name):
            return None
        path = self.csv_path(name)
        st = os.stat(path)
        key = (name, st.st_mtime_ns, st.st_size)
        digest = self._hashes.get(key)
        if digest is None:
            digest = self._sha256_file(path)
            with self._cache_lock:
                for old in [k for k in self._hashes if k[0] == name]:
                    del self._hashes[old]
                self._hashes[key] = digest
        return digest

    def verify(self, name: str, expected_sha256: Optional[str]) -> bool:
        """True when the stored dataset is byte-for-byte the one with `expected_sha256` (or nothing was
        recorded to compare with, as for runs from before hashes were kept)."""
        return not expected_sha256 or self.content_hash(name) == expected_sha256

    def _retired(self) -> Dict[str, Any]:
        try:
            with open(os.path.join(self.dir, _RETIRED_FILE)) as f:
                return {k.lower(): v for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------------ read / write
    def save(self, df: pd.DataFrame, meta: Dict[str, Any], wanted_name: str, activate: bool = True) -> str:
        """Stores a NEW dataset (never overwrites one) and returns the name it got."""
        meta = dict(meta, created_at_utc=meta.get("created_at_utc") or datetime.now(timezone.utc).isoformat())
        meta.update(self._counts(df))
        # One save at a time: the name is chosen and both files written before another save can pick a name.
        # The CSV is written first; the dataset only "exists" once its meta file is there.
        with self._write_lock:
            name = self.unique_name(wanted_name)
            df.to_csv(self.csv_path(name), index=False)
            meta.update(name=name, sha256=self._sha256_file(self.csv_path(name)))
            tmp = self.meta_path(name) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, self.meta_path(name))
        if activate:
            self.set_active(name)
        return name

    @staticmethod
    def _counts(df: pd.DataFrame) -> Dict[str, Any]:
        if "instance" not in df.columns:
            return {"total_tracks": None, "total_pulses": int(len(df))}
        first = df.groupby("instance", sort=False)["true_class_id"].first() if "true_class_id" in df.columns else None
        return {"total_tracks": int(df["instance"].nunique()), "total_pulses": int(len(df)),
                "known_tracks": int((first >= 0).sum()) if first is not None else None,
                "unknown_tracks": int((first < 0).sum()) if first is not None else None}

    def load(self, name: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        """(dataframe, meta) of the named dataset — the active one when `name` is None."""
        name = name or self.get_active()
        if not self.exists(name):
            return None, None
        with open(self.meta_path(name)) as f:
            meta = json.load(f)
        meta.setdefault("name", name)
        # A big CSV took seconds to parse on every request; keep the last few parsed frames, keyed by
        # file mtime + size so an overwritten/re-uploaded dataset is never served stale.
        path = self.csv_path(name)
        st = os.stat(path)
        key = (name, st.st_mtime_ns, st.st_size)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is None:
            hit = upgrade_columns(pd.read_csv(path))
            with self._cache_lock:
                for old in [k for k in self._cache if k[0] == name]:
                    del self._cache[old]
                self._cache[key] = hit
                while len(self._cache) > self._CACHE_SIZE:
                    self._cache.popitem(last=False)
        return hit.copy(), meta  # callers may modify their frame

    def meta(self, name: str) -> Optional[dict]:
        if not self.exists(name):
            return None
        with open(self.meta_path(name)) as f:
            meta = json.load(f)
        meta.setdefault("name", name)
        changed = False
        if "total_tracks" not in meta:  # written before counts were stored
            meta.update(self._counts(upgrade_columns(pd.read_csv(self.csv_path(name)))))
            changed = True
        if not meta.get("sha256"):      # stored before content hashes were kept: hash it now, once
            meta["sha256"] = self.content_hash(name)
            changed = True
        if changed:
            with open(self.meta_path(name), "w") as f:
                json.dump(meta, f, indent=2)
        return meta

    def list(self) -> List[dict]:
        """Every dataset, newest first, with the facts the pages show (no pulse data)."""
        out = []
        for f in os.listdir(self.dir):
            if f.endswith(".meta.json"):
                name = f[:-len(".meta.json")]
                m = self.meta(name)
                if m is None:
                    continue
                m["size_bytes"] = os.path.getsize(self.csv_path(name))
                out.append(m)
        out.sort(key=lambda m: m.get("created_at_utc") or m.get("generated_at_utc") or "", reverse=True)
        active = self.get_active()
        for m in out:
            m["is_active"] = m["name"] == active
        return out

    def delete(self, name: str) -> None:
        """Removes the dataset's files and retires its name for good (recorded with the hash it had)."""
        with self._write_lock:
            retired_path = os.path.join(self.dir, _RETIRED_FILE)
            try:
                with open(retired_path) as f:
                    retired = json.load(f)
            except (OSError, ValueError):
                retired = {}
            retired[name] = {"sha256": (self.meta(name) or {}).get("sha256"),
                             "deleted_at_utc": datetime.now(timezone.utc).isoformat()}
            tmp = retired_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(retired, f, indent=2)
            os.replace(tmp, retired_path)
        for p in (self.csv_path(name), self.meta_path(name)):
            if os.path.exists(p):
                os.remove(p)
        if self.get_active() == name:
            self._write_active(None)

    # ------------------------------------------------------------------ active dataset
    def _write_active(self, name: Optional[str]) -> None:
        with open(os.path.join(self.dir, _ACTIVE_FILE), "w") as f:
            json.dump({"name": name}, f)

    def set_active(self, name: str) -> None:
        self._write_active(name)

    def get_active(self) -> Optional[str]:
        """The Training page's default dataset: the one set last, else the newest."""
        try:
            with open(os.path.join(self.dir, _ACTIVE_FILE)) as f:
                name = json.load(f).get("name")
            if self.exists(name):
                return name
        except (OSError, ValueError):
            pass
        names = [m["name"] for m in self.list_unsorted()]
        return names[0] if names else None

    def list_unsorted(self) -> List[dict]:
        out = []
        for f in os.listdir(self.dir):
            if f.endswith(".meta.json"):
                try:
                    with open(os.path.join(self.dir, f)) as fh:
                        m = json.load(fh)
                    m.setdefault("name", f[:-len(".meta.json")])
                    out.append(m)
                except (OSError, ValueError):
                    continue
        out.sort(key=lambda m: m.get("created_at_utc") or m.get("generated_at_utc") or "", reverse=True)
        return out
