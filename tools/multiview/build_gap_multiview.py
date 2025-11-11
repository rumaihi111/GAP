#!/usr/bin/env python3
"""
Build a multi-view GAP package from 8 photos (000..315 at 45° increments).

Usage:
  python tools/multiview/build_gap_multiview.py \
    --input-dir "8 photos" \
    --asset-name supplement_mv
"""
import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Tuple

ANGLES = ["000","045","090","135","180","225","270","315"]
ANGLE_REGEX = re.compile(r"(?:^|[^0-9])(0{0,2}(?:0|45|90|135|180|225|270|315))(?:[^0-9]|$)")
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def find_photo_to_gap_cmd() -> List[str]:
    # Prefer module (core.photo_to_gap); else tools/photo_to_gap.py
    try:
        __import__("core.photo_to_gap")
        return [sys.executable, "-m", "core.photo_to_gap"]
    except Exception:
        script = Path(__file__).resolve().parents[2] / "tools" / "photo_to_gap.py"
        if script.exists():
            return [sys.executable, str(script)]
    print("Error: photo_to_gap not found (core.photo_to_gap or tools/photo_to_gap.py).", file=sys.stderr)
    sys.exit(1)

def infer_angles(files: List[Path]) -> List[Tuple[str, Path]]:
    # Map by angle token in filename if present; otherwise assign by sorted order.
    mapping = {}
    for f in files:
        m = ANGLE_REGEX.search(f.stem)
        if m:
            ang = f"{int(m.group(1)):03d}"
            if ang in ANGLES and ang not in mapping:
                mapping[ang] = f
    ordered = [(a, mapping[a]) for a in ANGLES if a in mapping]
    if len(ordered) == len(ANGLES):
        return ordered
    remaining = [f for f in files if f not in [p for _, p in ordered]]
    remaining_sorted = sorted(remaining)
    need = [a for a in ANGLES if a not in [x for x,_ in ordered]]
    for a, f in zip(need, remaining_sorted):
        ordered.append((a, f))
    return sorted(ordered, key=lambda x: ANGLES.index(x[0]))

def main():
    ap = argparse.ArgumentParser(description="Build multi-view GAP package from 8 photos.")
    ap.add_argument("--input-dir", required=True, help="Directory with 8 photos")
    ap.add_argument("--asset-name", required=True, help="Logical asset name (e.g., supplement_mv)")
    ap.add_argument("--out-dir", default="output/gaps", help="Base output directory")
    ap.add_argument("--skip-existing", action="store_true", help="Skip views that already have outputs")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Extra args passed to photo_to_gap")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    files = [p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if len(files) < 8:
        print(f"Need at least 8 images, found {len(files)} in {in_dir}", file=sys.stderr)
        sys.exit(1)

    angle_files = infer_angles(files)
    out_base = Path(args.out_dir) / args.asset_name
    out_base.mkdir(parents=True, exist_ok=True)

    cmd_base = find_photo_to_gap_cmd()
    ok, total = 0, len(angle_files)

    for ang, src in angle_files:
        view_dir = out_base / f"view_{ang}"
        view_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and all((view_dir / fn).exists() for fn in ["depth_map.png","normal_map.png","reference_image.png","metadata.json"]):
            print(f"[{ang}] Skipping (already exists)")
            ok += 1
            continue

        cmd = cmd_base + [
            "--input", str(src),
            "--output", str(view_dir),
            "--name", args.asset_name,   # CHANGED: was --asset-name
        ]
        if args.extra_args:
            cmd += args.extra_args

        print(f"[{ang}] {src.name}")
        print("  " + " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode == 0:
            ok += 1
        else:
            print(f"View {ang} failed (exit {res.returncode}).", file=sys.stderr)

    print(f"Done. {ok}/{total} views built at: {out_base}")
    print("Verify each view contains: depth_map.png, normal_map.png, reference_image.png, metadata.json")

if __name__ == "__main__":
    main()