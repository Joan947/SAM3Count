#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys
import multiprocessing as mp
from datetime import datetime
import shutil

from sam3count_eval import (
    count_objects_in_image,
    build_sam3_image_predictor,
)

def load_carpk_dataset(
    dataset_root: str,
    split: str = "test",
) -> List[Dict]:
    
    root        = Path(dataset_root)
    images_dir  = root / "Images"
    ann_dir     = root / "Annotations"
    imageset    = root / "ImageSets" / "Main" / f"{split}.txt"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {ann_dir}")

    if imageset.exists():
        print(f"  Using split file: {imageset}")
        with open(imageset) as f:
            stems = [line.strip() for line in f if line.strip()]
    else:
        print(f" No ImageSets file found at {imageset}")
        print(f"       Falling back: scanning Images/ for all .png/.jpg files")
        stems = [p.stem for p in sorted(images_dir.iterdir())
                 if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    samples = []
    missing  = 0
    for uid, stem in enumerate(stems):
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = str(candidate)
                break

        ann_path = ann_dir / (stem + ".txt")

        if img_path is None:
            print(f" Image not found for stem: {stem}")
            missing += 1
            continue
        if not ann_path.exists():
            print(f" Annotation not found: {ann_path}")
            missing += 1
            continue

        samples.append({
            "image_id":  uid,
            "image_path": img_path,
            "ann_path":  str(ann_path),
            "filename":  Path(img_path).name,
            "split":     split,
        })

    print(f"  {len(samples)} samples loaded  "
          f"(split={split}, missing={missing})")
    return samples


def read_gt_count_txt(ann_path: str) -> int:
    
    try:
        count = 0
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 4:   # at minimum x1 y1 x2 y2
                        count += 1
        return count
    except Exception as e:
        print(f" Could not read {ann_path}: {e}")
        return 0


def load_progress(progress_file: str) -> Dict:
    if not os.path.exists(progress_file):
        return {
            "processed_images": [],
            "results": {
                "per_image": [], "errors": [],
                "absolute_errors": [], "squared_errors": [],
            },
        }
    try:
        with open(progress_file) as f:
            data = json.load(f)
        print(f"  Loaded progress: {len(data['processed_images'])} images")
        return data
    except json.JSONDecodeError as e:
        print(f" Progress file corrupted: {e}")
        backup = progress_file + ".corrupted"
        try:
            shutil.copy(progress_file, backup)
        except Exception:
            pass
        return {
            "processed_images": [],
            "results": {
                "per_image": [], "errors": [],
                "absolute_errors": [], "squared_errors": [],
            },
        }


def save_progress(progress_file: str, data: Dict):
    tmp = progress_file + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        if os.path.exists(progress_file):
            os.replace(tmp, progress_file)
        else:
            os.rename(tmp, progress_file)
    except Exception as e:
        print(f" Failed to save progress: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

def worker_process(
    gpu_id:        int,
    sample_subset: List[Dict],
    output_dir:    str,
    args:          Dict,
    resume:        bool = False,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    print(f"[gpu {gpu_id}] building model")
    try:
        model, device_obj = build_sam3_image_predictor(device, args.get("bpe_path"))
        print(f"[gpu {gpu_id}] Model ready on {device_obj}")
    except Exception as e:
        print(f"[gpu {gpu_id}] Model load failed: {e}")
        import traceback; traceback.print_exc()
        return None
  
    out_file      = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
    progress_file = os.path.join(output_dir, f"gpu_{gpu_id}_progress.json")

    print(f"[gpu {gpu_id}] Assigned {len(sample_subset)} images")

    processed_ids = set()
    results = {
        "per_image": [], "errors": [],
        "absolute_errors": [], "squared_errors": [],
    }

    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                old = json.load(f)
            results["per_image"]       = old.get("per_image", [])
            results["errors"]          = old.get("errors", [])
            results["absolute_errors"] = old.get("absolute_errors", [])
            results["squared_errors"]  = old.get("squared_errors", [])
            for img in results["per_image"]:
                processed_ids.add(img["image_id"])
            print(f"[gpu {gpu_id}] 📂 Loaded {len(results['per_image'])} old results")
        except Exception as e:
            print(f"[gpu {gpu_id}]Could not load old results: {e}")

    import sam3count_eval as count_module
    _orig_build = count_module.build_sam3_image_predictor

    def _use_prebuilt(device_str=None, bpe_path=None):
        return model, device_obj

    count_module.build_sam3_image_predictor = _use_prebuilt
    
    text_prompt     = args.get("text_prompt", "car")
    processed_count = 0
    failed_count    = 0

    try:
        for idx, sample in enumerate(sample_subset):
            img_id = sample["image_id"]

            if img_id in processed_ids:
                print(f"[gpu {gpu_id}] ⏭  Skipping image_id={img_id}")
                continue

            image_path = sample["image_path"]
            ann_path   = sample["ann_path"]
            filename   = sample["filename"]

            if not os.path.exists(image_path):
                print(f"[gpu {gpu_id}]Missing: {image_path}")
                failed_count += 1
                continue

            gt_count = read_gt_count_txt(ann_path)

            try:
                pred_count, detections, _ = count_module.count_objects_in_image(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    device_str=device,
                    confidence_threshold=args["confidence_threshold"],
                    min_obj_area=args.get("min_obj_area", 0),
                    iom_threshold=args["iom_threshold"],
                    apply_nms=True,
                    bpe_path=args.get("bpe_path"),
                    visualizer=None,
                )

                error     = pred_count - gt_count
                abs_error = abs(error)

                clean_dets = [
                    {
                        "box":   det.get("box", []),
                        "score": float(det.get("score", 0.0)),
                        "area":  int(det.get("area", 0)),
                    }
                    for det in detections
                ]

                results["per_image"].append({
                    "image_id":   int(img_id),
                    "filename":   filename,
                    "gt_count":   int(gt_count),
                    "pred_count": int(pred_count),
                    "error":      int(error),
                    "abs_error":  int(abs_error),
                    "detections": clean_dets,
                })
                results["errors"].append(float(error))
                results["absolute_errors"].append(float(abs_error))
                results["squared_errors"].append(float(error ** 2))

                processed_ids.add(int(img_id))
                processed_count += 1

                print(
                    f"[gpu {gpu_id}] [{idx+1}/{len(sample_subset)}] "
                    f"{filename}  GT={gt_count}  Pred={pred_count}  AE={abs_error}"
                )

            except Exception as e:
                print(f"[gpu {gpu_id}]  Error on {filename}: {e}")
                failed_count += 1
                import traceback; traceback.print_exc()
                continue

            if (idx + 1) % 10 == 0 and results["absolute_errors"]:
                mae  = np.mean(results["absolute_errors"])
                rmse = np.sqrt(np.mean(results["squared_errors"]))
                print(
                    f"[gpu {gpu_id}] [{idx+1}/{len(sample_subset)}] "
                    f"Total={len(results['per_image'])} | MAE={mae:.2f} RMSE={rmse:.2f}"
                )
                save_progress(progress_file, {
                    "processed_images": list(processed_ids),
                    "results":          results,
                    "timestamp":        datetime.now().isoformat(),
                    "gpu_id":           gpu_id,
                })

    finally:
        count_module.build_sam3_image_predictor = _orig_build

    save_progress(progress_file, {
        "processed_images": list(processed_ids),
        "results":          results,
        "timestamp":        datetime.now().isoformat(),
        "gpu_id":           gpu_id,
    })

    print(f"\n[gpu {gpu_id}] Done — new={processed_count}  "
          f"failed={failed_count}  total={len(results['per_image'])}")

    if results["absolute_errors"]:
        mae  = np.mean(results["absolute_errors"])
        rmse = np.sqrt(np.mean(results["squared_errors"]))
        results["summary"] = {
            "gpu_id":     gpu_id,
            "mae":        float(mae),
            "rmse":       float(rmse),
            "num_images": len(results["per_image"]),
        }
        print(f"[gpu {gpu_id}]    MAE={mae:.2f}  RMSE={rmse:.2f}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gpu {gpu_id}] {out_file}")

    if processed_count == len(sample_subset) and os.path.exists(progress_file):
        os.remove(progress_file)

    return out_file


def get_all_processed_images_global(
    output_dir:        str,
    num_possible_gpus: int = 16,
) -> Tuple[Set[int], Dict[int, int]]:
    all_processed    = set()
    gpu_image_counts = {}
    for gid in range(num_possible_gpus):
        path = os.path.join(output_dir, f"gpu_{gid}_results.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                ids = [img["image_id"] for img in data.get("per_image", [])]
                all_processed.update(ids)
                gpu_image_counts[gid] = len(ids)
                print(f"gpu {gid}: {len(ids)} images")
            except Exception as e:
                print(f" gpu {gid} result unreadable: {e}")
    return all_processed, gpu_image_counts


def merge_results(
    output_files:   List[str],
    final_output:   str,
    summary_output: str,
) -> Dict:
    print(f"\n{'='*80}")
    print(f"MERGING RESULTS FROM {len(output_files)} WORKERS")
    print(f"{'='*80}")

    all_results = {
        "per_image": [], "errors": [],
        "absolute_errors": [], "squared_errors": [],
    }
    seen_ids        = set()
    duplicate_count = 0

    for gid, fpath in enumerate(output_files):
        if not os.path.exists(fpath):
            print(f" Not found: {fpath}")
            continue
        with open(fpath) as f:
            worker = json.load(f)
        n = len(worker.get("per_image", []))
        print(f"  gpu {gid}: {n} images")

        for img in worker.get("per_image", []):
            iid = img["image_id"]
            if iid in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(iid)
                all_results["per_image"].append(img)
                all_results["errors"].append(float(img["error"]))
                all_results["absolute_errors"].append(float(img["abs_error"]))
                all_results["squared_errors"].append(float(img["error"] ** 2))

    if duplicate_count:
        print(f" {duplicate_count} duplicates removed")

    if not all_results["absolute_errors"]:
        print("\n No data to aggregate.")
        with open(final_output, "w") as f:
            json.dump(all_results, f, indent=2)
        return all_results

    errors     = np.array(all_results["errors"])
    abs_errors = np.array(all_results["absolute_errors"])

    mae        = float(abs_errors.mean())
    rmse       = float(np.sqrt((errors ** 2).mean()))
    mean_error = float(errors.mean())
    under_5  = int((abs_errors <  5).sum())
    under_10 = int((abs_errors < 10).sum())
    under_20 = int((abs_errors < 20).sum())
    total_n  = len(abs_errors)

    all_results["summary"] = {
        "mae":          mae,
        "rmse":         rmse,
        "mean_error":   mean_error,
        "num_images":   total_n,
        "unique_images": len(seen_ids),
        "timestamp":    datetime.now().isoformat(),
    }

    summary_report = {
        "overall_metrics": {
            "mae":          mae,
            "rmse":         rmse,
            "mean_error":   mean_error,
            "total_images": total_n,
            "timestamp":    datetime.now().isoformat(),
        },
        "error_distribution": {
            "abs_error_lt_5":  under_5,
            "abs_error_lt_10": under_10,
            "abs_error_lt_20": under_20,
            "pct_lt_5":  round(100 * under_5  / total_n, 2),
            "pct_lt_10": round(100 * under_10 / total_n, 2),
            "pct_lt_20": round(100 * under_20 / total_n, 2),
        },
        # Worst-performing images for debugging
        "worst_10_images": sorted(
            [
                {
                    "filename":   img["filename"],
                    "gt_count":   img["gt_count"],
                    "pred_count": img["pred_count"],
                    "abs_error":  img["abs_error"],
                }
                for img in all_results["per_image"]
            ],
            key=lambda x: x["abs_error"],
            reverse=True,
        )[:10],
    }

    with open(summary_output, "w") as f:
        json.dump(summary_report, f, indent=2)
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("CARPK final results")
    print(f"\nSummary : {summary_output}")
    print(f"Detailed: {final_output}")

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="multi-gpu CARPK evaluation with SAM3Count"
    )

    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to CARPK_devkit/data  "
                             "(contains Images/, Annotations/, ImageSets/)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--text_prompt", type=str, default="car",
                        help="SAM3 text prompt (default: 'car')")

    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of gpus (default: 4)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="Specific gpu IDs (e.g. --gpu_ids 0 2 3)")

    parser.add_argument("--output_json",  type=str,
                        default="carpk_results.json",
                        help="Path for per-image results JSON")
    parser.add_argument("--summary_json", type=str,
                        default="carpk_summary.json",
                        help="Path for summary JSON")
    parser.add_argument("--temp_dir",     type=str,
                        default="./temp_carpk_dense_sampt",
                        help="Temp dir for per-gpu files")

    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous run")

    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Initial SAM3 confidence threshold (default: 0.5)")
    parser.add_argument("--iom_threshold", type=float, default=0.5,
                        help="IoM NMS threshold (default: 0.5)")
    parser.add_argument("--min_obj_area",  type=int,   default=0,
                        help="Min pixel area to keep a detection (default: 0)")
    parser.add_argument("--bpe_path", type=str, default=None)

    # debug
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process only N images (for smoke-testing)")

    args = parser.parse_args()

    gpu_ids  = args.gpu_ids if args.gpu_ids is not None else list(range(args.num_gpus))
    num_gpus = len(gpu_ids)

    os.makedirs(args.temp_dir, exist_ok=True)
    print("Loading dataset …")
    all_samples = load_carpk_dataset(args.dataset_root, args.split)

    if args.max_images:
        all_samples = all_samples[:args.max_images]

    total = len(all_samples)
    if total == 0:
        print(" No samples found. Check --dataset_root.")
        sys.exit(1)

    globally_processed: Set[int] = set()
    if args.resume:
        print(f"\nresume: scanning existing gpu result files …")
        globally_processed, gpu_counts = get_all_processed_images_global(args.temp_dir)
        if globally_processed:
            print(f"  {len(globally_processed)} images already processed")
        else:
            print(" No prior progress found — starting fresh")

    if len(globally_processed) >= total:
        print("\nAll images already processed — merging …")
        out_files = [
            os.path.join(args.temp_dir, f"gpu_{g}_results.json")
            for g in range(16)
            if os.path.exists(os.path.join(args.temp_dir, f"gpu_{g}_results.json"))
        ]
        if out_files:
            merge_results(out_files, args.output_json, args.summary_json)
        return

    remaining = [s for s in all_samples if s["image_id"] not in globally_processed]

    print(f"\n{'='*80}")
    print(f"  Total: {total}  |  Done: {len(globally_processed)}  |  Remaining: {len(remaining)}")
    print(f"{'='*80}\n")

    if not remaining:
        print("Nothing left to process.")
        return

    # split across gpus 
    base    = len(remaining) // num_gpus
    rem     = len(remaining) %  num_gpus
    subsets = []
    start   = 0
    for i in range(num_gpus):
        size = base + (1 if i < rem else 0)
        subsets.append(remaining[start:start + size])
        start += size
        print(f"  gpu {gpu_ids[i]}: {len(subsets[-1])} images")

    assert sum(len(s) for s in subsets) == len(remaining)

    worker_args = {
        "confidence_threshold": args.confidence_threshold,
        "iom_threshold":        args.iom_threshold,
        "min_obj_area":         args.min_obj_area,
        "text_prompt":          args.text_prompt,
        "bpe_path":             args.bpe_path,
    }


    with mp.Pool(processes=num_gpus) as pool:
        async_results = []
        for gpu_id, subset in zip(gpu_ids, subsets):
            ar = pool.apply_async(
                worker_process,
                args=(gpu_id, subset, args.temp_dir, worker_args, args.resume),
            )
            async_results.append(ar)

        out_files = []
        for i, ar in enumerate(async_results):
            try:
                path = ar.get(timeout=7200)
                if path:
                    out_files.append(path)
            except Exception as e:
                print(f" Worker {i} failed: {e}")
                import traceback; traceback.print_exc()

    if out_files:
        merge_results(out_files, args.output_json, args.summary_json)
        print("\nCARPK evaluation complete!")
    else:
        print("\n No output files to merge.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()