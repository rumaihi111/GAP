#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True, help="Path to multi-view GAP dir (e.g. output/gaps/supplement_mv)")
    ap.add_argument("--out", default="multiview_manifest.json")
    args = ap.parse_args()

    pkg = Path(args.package_dir)
    if not pkg.exists():
        raise SystemExit(f"Not found: {pkg}")

    manifest = {
        "asset_name": pkg.name,
        "views": {}
    }

    for view_dir in sorted(pkg.glob("view_*")):
        angle = view_dir.name.split("_")[1]
        manifest["views"][angle] = {
            "reference": str(view_dir / "canonical" / "reference.png"),
            "reference_no_bg": str(view_dir / "canonical" / "reference_no_bg.png"),
            "depth": str(view_dir / "geometry" / "depth_000.png"),
            "mask": str(view_dir / "geometry" / "mask_000.png"),
            "identity": str(view_dir / "identity_embedding.pt"),
            "metadata": str(view_dir / "metadata.json"),
        }

    out_path = pkg / args.out
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Manifest written: {out_path}")

if __name__ == "__main__":
    main()