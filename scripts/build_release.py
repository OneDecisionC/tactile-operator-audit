"""Build a source ZIP from the verified explicit release manifest, never from the whole folder."""
import argparse
import json
import zipfile
from pathlib import Path

from verify_release import ROOT, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = ROOT / "RELEASE_FILES.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    if entries != inventory():
        raise ValueError("Release manifest mismatch")
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Put the ZIP outside the source directory")
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(ROOT / entry["path"], f"tactile-operator-audit-0.2.0/{entry['path']}")
        archive.write(manifest, "tactile-operator-audit-0.2.0/RELEASE_FILES.json")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP verification failed")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
