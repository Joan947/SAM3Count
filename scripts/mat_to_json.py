#!/usr/bin/env python3
"""
Convert all .mat files in a directory (recursively) to JSON.

Supports:
- Standard MATLAB .mat files via scipy.io.loadmat
- MATLAB v7.3 .mat files (HDF5-based) via h5py fallback

Special handling for ShanghaiTech crowd-counting GT files:
- Attempts to extract point annotations from `image_info`
- Stores `points` and `count` if detected

Usage examples:
    python mat_to_json_all.py --input_dir ./part_B/test_data/ground-truth --output_dir ./part_B/test_data/gt_json --mode per_file
    python mat_to_json_all.py --input_dir ./part_B/test_data/ground-truth --output_file ./part_B/test_data/annotations_B.json --mode combined
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ----------------------------
# JSON conversion helpers
# ----------------------------

def to_jsonable(obj: Any, max_string_len: int = 10000) -> Any:
    """
    Recursively convert Python/numpy/scipy-loaded MATLAB objects into JSON-serializable types.
    """
    # Primitive types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str) and len(obj) > max_string_len:
            return obj[:max_string_len] + "...<truncated>"
        return obj

    # Bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            s = obj.decode("utf-8", errors="replace")
            if len(s) > max_string_len:
                s = s[:max_string_len] + "...<truncated>"
            return s
        except Exception:
            return repr(obj)

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # NumPy scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        # Object arrays: recurse element-wise
        if obj.dtype == object:
            return [to_jsonable(x, max_string_len=max_string_len) for x in obj.tolist()]

        # Structured arrays
        if obj.dtype.names is not None:
            return [
                {name: to_jsonable(row[name], max_string_len=max_string_len) for name in obj.dtype.names}
                for row in obj
            ]

        # Normal numeric/string arrays
        try:
            return obj.tolist()
        except Exception:
            return repr(obj)

    # Python containers
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            out[key] = to_jsonable(v, max_string_len=max_string_len)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x, max_string_len=max_string_len) for x in obj]

    # Fallback for unknown objects (e.g., scipy mat_struct)
    # Try inspecting __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {
                "__class__": obj.__class__.__name__,
                **{k: to_jsonable(v, max_string_len=max_string_len) for k, v in vars(obj).items()}
            }
        except Exception:
            pass

    # Final fallback
    return repr(obj)


# ----------------------------
# ShanghaiTech-specific parsing
# ----------------------------

def infer_image_name_from_gt(gt_name: str) -> Optional[str]:
    """
    Example: GT_IMG_1.mat -> IMG_1.jpg
    """
    m = re.match(r"GT_(IMG_\d+)\.mat$", gt_name, flags=re.IGNORECASE)
    if m:
        return m.group(1) + ".jpg"
    return None


def try_extract_shanghaitech_points(mat_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Try common ShanghaiTech indexing patterns for `image_info`.
    Returns Nx2 ndarray if successful, else None.
    """
    if "image_info" not in mat_data:
        return None

    image_info = mat_data["image_info"]

    patterns = [
        lambda x: x[0, 0][0, 0][0],   # most common
        lambda x: x[0, 0][0, 0],      # sometimes directly Nx2
        lambda x: x[0][0][0][0][0],   # extra nesting
        lambda x: x[0][0],            # fallback
    ]

    for fn in patterns:
        try:
            pts = np.array(fn(image_info)).squeeze()
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts.astype(float)
        except Exception:
            continue

    return None


# ----------------------------
# .mat loaders
# ----------------------------

def load_mat_with_scipy(mat_path: str) -> Dict[str, Any]:
    import scipy.io as sio
    data = sio.loadmat(mat_path)
    # Remove MATLAB metadata keys if present
    return {k: v for k, v in data.items() if not k.startswith("__")}


def load_mat_with_h5py(mat_path: str) -> Dict[str, Any]:
    import h5py

    def h5_to_py(obj):
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            if isinstance(arr, bytes):
                try:
                    return arr.decode("utf-8", errors="replace")
                except Exception:
                    return repr(arr)
            return to_jsonable(arr)

        if isinstance(obj, h5py.Group):
            return {k: h5_to_py(obj[k]) for k in obj.keys()}

        return repr(obj)

    out = {}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            out[k] = h5_to_py(f[k])
    return out


def load_any_mat(mat_path: str) -> Dict[str, Any]:
    """
    Try scipy first, then h5py fallback.
    Returns a dictionary of loaded content.
    Raises exception if both fail.
    """
    scipy_err = None
    try:
        data = load_mat_with_scipy(mat_path)
        data["_loader"] = "scipy.io.loadmat"
        return data
    except Exception as e:
        scipy_err = e

    h5_err = None
    try:
        data = load_mat_with_h5py(mat_path)
        data["_loader"] = "h5py"
        return data
    except Exception as e:
        h5_err = e

    raise RuntimeError(
        f"Failed to load MAT file with both scipy and h5py.\n"
        f"scipy error: {scipy_err}\n"
        f"h5py error: {h5_err}"
    )


# ----------------------------
# Conversion logic
# ----------------------------

def convert_mat_file(mat_path: str, input_root: str) -> Dict[str, Any]:
    """
    Convert a single .mat file into a JSON-friendly dictionary.
    Includes:
    - file metadata
    - parsed raw content
    - ShanghaiTech convenience fields if detected
    """
    mat_data = load_any_mat(mat_path)

    rel_path = os.path.relpath(mat_path, input_root)
    fname = os.path.basename(mat_path)

    result: Dict[str, Any] = {
        "file_name": fname,
        "relative_path": rel_path,
        "absolute_path": os.path.abspath(mat_path),
        "format": "mat",
    }

    # ShanghaiTech convenience extraction
    sh_points = None
    try:
        sh_points = try_extract_shanghaitech_points(mat_data)
    except Exception:
        sh_points = None

    if sh_points is not None:
        result["shanghaitech"] = {
            "image_name": infer_image_name_from_gt(fname),
            "count": int(len(sh_points)),
            "points": sh_points.tolist(),
        }

    # Raw content (JSON-serializable conversion)
    result["content"] = to_jsonable(mat_data)

    return result


def find_mat_files(input_dir: str) -> List[str]:
    mat_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".mat"):
                mat_files.append(os.path.join(root, f))
    mat_files.sort()
    return mat_files


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert all .mat files and their content to JSON.")
    parser.add_argument("--input_dir", required=True, help="Root directory to search for .mat files recursively")
    parser.add_argument(
        "--mode",
        choices=["per_file", "combined"],
        default="per_file",
        help="per_file: one JSON for each .mat | combined: one JSON containing all .mat contents"
    )
    parser.add_argument("--output_dir", default=None, help="Output directory for per_file mode")
    parser.add_argument("--output_file", default=None, help="Output JSON file path for combined mode")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    parser.add_argument(
        "--skip_raw_content",
        action="store_true",
        help="If set, only store metadata + ShanghaiTech parsed fields (smaller JSONs)"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    mat_files = find_mat_files(input_dir)
    if not mat_files:
        print(f"No .mat files found under: {input_dir}")
        return

    print(f"Found {len(mat_files)} .mat files.")

    # Validate mode/output args
    if args.mode == "per_file":
        if not args.output_dir:
            raise ValueError("--output_dir is required when --mode per_file")
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        if not args.output_file:
            raise ValueError("--output_file is required when --mode combined")
        output_file = os.path.abspath(args.output_file)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    converted_items = []
    num_ok = 0
    num_fail = 0

    for idx, mat_path in enumerate(mat_files, 1):
        try:
            item = convert_mat_file(mat_path, input_dir)

            if args.skip_raw_content:
                # Keep only metadata + shanghaitech extracted fields if present
                item = {
                    k: v for k, v in item.items()
                    if k in {"file_name", "relative_path", "absolute_path", "format", "shanghaitech"}
                }

            if args.mode == "per_file":
                # Preserve relative directory structure in output_dir
                rel_mat_path = os.path.relpath(mat_path, input_dir)
                rel_json_path = os.path.splitext(rel_mat_path)[0] + ".json"
                out_path = os.path.join(output_dir, rel_json_path)
                write_json(out_path, item, indent=args.indent)
                msg = f"[{idx}/{len(mat_files)}] OK  {rel_mat_path} -> {rel_json_path}"
            else:
                converted_items.append(item)
                msg = f"[{idx}/{len(mat_files)}] OK  {os.path.relpath(mat_path, input_dir)}"

            # Nice summary if ShanghaiTech points were extracted
            if "shanghaitech" in item and isinstance(item["shanghaitech"], dict):
                cnt = item["shanghaitech"].get("count", None)
                if cnt is not None:
                    msg += f"  (count={cnt})"

            print(msg)
            num_ok += 1

        except Exception as e:
            rel = os.path.relpath(mat_path, input_dir)
            print(f"[{idx}/{len(mat_files)}] FAIL {rel} | {e}")
            num_fail += 1

            if args.mode == "combined":
                converted_items.append({
                    "file_name": os.path.basename(mat_path),
                    "relative_path": rel,
                    "absolute_path": os.path.abspath(mat_path),
                    "format": "mat",
                    "error": str(e),
                })

    if args.mode == "combined":
        payload = {
            "input_dir": input_dir,
            "num_files_found": len(mat_files),
            "num_converted_ok": num_ok,
            "num_failed": num_fail,
            "items": converted_items,
        }
        write_json(output_file, payload, indent=args.indent)
        print(f"\nSaved combined JSON: {output_file}")

    print(f"\nDone. Success: {num_ok} | Failed: {num_fail} | Total: {len(mat_files)}")


if __name__ == "__main__":
    main()