"""Integrity and provenance for downloadable model packages.

Every package carries MANIFEST.json: the SHA-256 and size of every other file in it, plus the run, the DADS
release and the dataset hash it came from. When the `cryptography` library is installed, the server also signs
the manifest with its Ed25519 key (MANIFEST.json.sig). The key pair lives in <data>/signing/; the private key never
leaves the server, and the public key (GET /api/package-key) is what a receiver pins as trusted.

Verification (verify_package_files) needs only the standard library for the hashes; checking the signature
needs `cryptography`. This file is copied into every package as verify_package.py (standard library only, apart
from the optional signature check), so a receiver can check a package without DADS:
    python verify_package.py [package.zip | folder] --trusted-key <hex>
"""
import base64
import hashlib
import json
import os
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

MANIFEST = "MANIFEST.json"
SIGNATURE = "MANIFEST.json.sig"
MANIFEST_FORMAT = "dads-package-manifest/1"

try:  # optional: without it packages are hashed but not signed
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    HAVE_CRYPTO = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_CRYPTO = False


def key_fingerprint(public_raw: bytes) -> str:
    """Short, human-comparable identity of a public key: SHA-256 of the raw 32 bytes, first 16 bytes in hex."""
    return hashlib.sha256(public_raw).hexdigest()[:32]


class PackageSigner:
    """The server's package-signing key. Created on first use; private key file is readable by the owner only."""

    def __init__(self, data_dir: str):
        self.dir = os.path.join(data_dir, "signing")
        self.private_path = os.path.join(self.dir, "package_ed25519.pem")
        self.public_path = os.path.join(self.dir, "package_ed25519.pub")
        self._key = None

    @property
    def available(self) -> bool:
        return HAVE_CRYPTO

    def _load(self):
        if self._key is not None:
            return self._key
        if not HAVE_CRYPTO:
            raise RuntimeError("Package signing needs the 'cryptography' package.")
        if os.path.exists(self.private_path):
            with open(self.private_path, "rb") as f:
                self._key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            os.makedirs(self.dir, exist_ok=True)
            key = Ed25519PrivateKey.generate()
            pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
            fd = os.open(self.private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
            with open(self.public_path, "w") as f:
                f.write(self._public_raw(key).hex() + "\n")
            self._key = key
        return self._key

    @staticmethod
    def _public_raw(key) -> bytes:
        return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def public_key(self) -> Optional[Dict[str, str]]:
        if not HAVE_CRYPTO:
            return None
        raw = self._public_raw(self._load())
        return {"algorithm": "ed25519", "public_key_hex": raw.hex(), "fingerprint": key_fingerprint(raw)}

    def sign(self, data: bytes) -> bytes:
        return self._load().sign(data)


def build_manifest(files: Dict[str, bytes], provenance: Dict[str, Any],
                   signer: Optional[PackageSigner]) -> Tuple[bytes, Optional[bytes]]:
    """MANIFEST.json bytes (and its signature, if the signer can sign) for a package made of `files`."""
    manifest = {
        "format": MANIFEST_FORMAT,
        "provenance": provenance,
        "signer": signer.public_key() if signer is not None and signer.available else None,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                  for name, data in sorted(files.items())},
    }
    body = json.dumps(manifest, indent=2, sort_keys=True).encode()
    sig = signer.sign(body) if manifest["signer"] else None
    return body, (base64.b64encode(sig) + b"\n") if sig else None


def verify_package_files(read: Callable[[str], Optional[bytes]], names: Iterable[str],
                         trusted_key_hex: Optional[str] = None, allow_extra: bool = False) -> Tuple[bool, str]:
    """Checks a package given `read(name) -> bytes or None` and the list of its file names.
    Passes only if every file is listed in the manifest with the right hash and nothing listed is missing; with
    `trusted_key_hex`, the manifest must also carry a valid signature by exactly that key. allow_extra=True ignores
    files the manifest doesn't list (an unpacked folder may hold the user's own results or a TensorRT engine)."""
    manifest_bytes = read(MANIFEST)
    if manifest_bytes is None:
        return False, "No MANIFEST.json: the package was made before v1.2.0 or has been altered."
    try:
        manifest = json.loads(manifest_bytes)
        listed = manifest["files"]
    except (ValueError, KeyError, TypeError):
        return False, "MANIFEST.json is unreadable."
    present = {n for n in names if n not in (MANIFEST, SIGNATURE) and not n.endswith("/")}
    extra = sorted(present - set(listed))
    if extra and not allow_extra:
        return False, f"Files not covered by the manifest: {', '.join(extra)}"
    for name, want in listed.items():
        data = read(name)
        if data is None:
            return False, f"Missing file: {name}"
        if hashlib.sha256(data).hexdigest() != want.get("sha256"):
            return False, f"Hash mismatch: {name} has been changed."
    signer = manifest.get("signer")
    if trusted_key_hex is None:
        note = f" Signed by key {signer['fingerprint']} (not checked: no trusted key given)." if signer else " Unsigned."
        return True, f"All {len(listed)} files match the manifest.{note}"
    if not HAVE_CRYPTO:
        return False, "Checking the signature needs the 'cryptography' package (pip install cryptography)."
    sig_b64 = read(SIGNATURE)
    if not signer or sig_b64 is None:
        return False, "The package is not signed."
    if signer.get("public_key_hex", "").lower() != trusted_key_hex.strip().lower():
        return False, f"Signed by a different key ({signer.get('fingerprint')}), not the trusted one."
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_key_hex.strip())).verify(
            base64.b64decode(sig_b64.strip()), manifest_bytes)
    except Exception:  # InvalidSignature or a malformed key/signature
        return False, "Invalid signature: the manifest was changed after signing."
    return True, f"All {len(listed)} files match and the signature by key {signer['fingerprint']} is valid."


def verify_package_path(path: str, trusted_key_hex: Optional[str] = None, allow_extra: bool = False) -> Tuple[bool, str]:
    """verify_package_files for a .zip file or an unpacked package folder."""
    if os.path.isdir(path):
        names = [os.path.relpath(os.path.join(root, f), path).replace(os.sep, "/")
                 for root, _, fs in os.walk(path) for f in fs if "__pycache__" not in root]

        def read(n):
            p = os.path.join(path, *n.split("/"))
            if not os.path.isfile(p):
                return None
            with open(p, "rb") as fh:
                return fh.read()
        return verify_package_files(read, names, trusted_key_hex, allow_extra)
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        return verify_package_files(lambda n: z.read(n) if n in names else None, names, trusted_key_hex, allow_extra)


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Check a DADS model package (.zip or unpacked folder).")
    ap.add_argument("package", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--trusted-key", help="the DADS server's public key (hex, from GET /api/package-key)")
    a = ap.parse_args()
    ok, msg = verify_package_path(a.package, a.trusted_key, allow_extra=os.path.isdir(a.package))
    print(("OK: " if ok else "FAILED: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
