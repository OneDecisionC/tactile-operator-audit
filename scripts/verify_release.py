"""Verify explicit release files, local documentation links and obvious private artifacts."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {".git", ".venv", "venv", "outputs", "build", "dist", ".pytest_cache", "__pycache__"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    "token": re.compile(r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9]{20,}"),
    "private path/IP": re.compile(r"(?:[A-Z]:[\\/]Users[\\/]|/mnt/" r"data/|172\.31\.\d+\.\d+)"),
}
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npz", ".npy", ".h5", ".hdf5", ".pem", ".key", ".zip", ".exe"}


def files():
    result = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in GENERATED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in release: {relative}")
        if path.is_file() and path.name != "RELEASE_FILES.json":
            result.append(path)
    return sorted(result)


def inventory():
    entries = []
    for path in files():
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS or path.name.startswith(".env") or ".local." in path.name:
            raise ValueError(f"Private/binary/local artifact: {relative}")
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"Potential {label} in {relative}; inspect locally, do not publish")
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", text):
                if link.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = link.split("#")[0]
                if target and not (path.parent / target).exists():
                    raise ValueError(f"Broken local link: {relative} -> {target}")
        entries.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Explicitly refresh hashes after reviewed changes")
    args = parser.parse_args()
    actual = inventory()
    manifest = ROOT / "RELEASE_FILES.json"
    if args.write_manifest:
        manifest.write_text(json.dumps({"version": "0.2.0", "files": actual,
            "note": "Manifest excludes itself; patterns are a heuristic scan, not a guarantee of absence of secrets."}, indent=2) + "\n", encoding="utf-8")
    else:
        expected = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        if actual != expected:
            raise ValueError("Release files changed, are missing, or were added; review before explicitly regenerating manifest")
    print(json.dumps({"status": "PASS", "files": len(actual), "checks": ["hashes", "local links", "private-artifact patterns"]}))


if __name__ == "__main__":
    main()
