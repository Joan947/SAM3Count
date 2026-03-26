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
import scipy.io

from  sam3count_eval import (
    count_objects_in_image,
    build_sam3_image_predictor,
)

def load_dataset(
    data_r: str,
    part: str = "A",
    split: str = "test",
) -> List[Dict]:
    parts  = ["A", "B"]  if part  == "both" else [part]
    splits = ["train", "test"] if split == "both" else [split]
    samples = []
    uid     = 0

    for p in parts:
        for s in splits:
            img_dir = Path(data_r) / f"part_{p}" / f"{s}_data" / "images"
            gt_dir  = Path(data_r) / f"part_{p}" / f"{s}_data" / "ground_truth"

            if not img_dir.exists():
                print(f" Image dir not found: {img_dir}")
                continue
            if not gt_dir.exists():
                print(f" GT dir not found: {gt_dir}")
                continue

            for img_path in sorted(img_dir.glob("IMG_*.jpg")):
                stem    = img_path.stem                   # e.g. IMG_1
                gt_file = gt_dir / f"GT_{stem}.mat"
                if not gt_file.exists():
                    print(f" GT not found for {img_path.name}, skipping")
                    continue

                samples.append({
                    "image_id":   uid,
                    "image_path": str(img_path),
                    "gt_path":    str(gt_file),
                    "filename":   img_path.name,
                    "part":       p,
                    "split":      s,
                })
                uid += 1

    print(f" Loaded {len(samples)} samples  "
          f"(part={part}, split={split})")
    return samples


def read_gt_count(gt_mat_path: str) -> int:
    try:
        mat = scipy.io.loadmat(gt_mat_path)
        # Standard layout: image_info[0,0]['location'][0,0] -> Nx2
        locs = mat["image_info"][0, 0]["location"][0, 0]
        return int(locs.shape[0])
    except Exception as e:
        print(f" Could not read {gt_mat_path}: {e}")
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
        print(f" Loaded progress: {len(data['processed_images'])} images")
        return data
    except json.JSONDecodeError as e:
        print(f" Progress file corrupted: {e}")
        backup = progress_file + ".corrupted"
        try:
            shutil.copy(progress_file, backup)
            print(f" Backed up to: {backup}")
        except Exception:
            pass
        return {
            "processed_images": [],
            "results": {
                "per_image": [], "errors": [],
                "absolute_errors": [], "squared_errors": [],
            },
        }

def save_progress(progress_file: str, progress_data: Dict):
    temp = progress_file + ".tmp"
    try:
        with open(temp, "w") as f:
            json.dump(progress_data, f, indent=2)
        if os.path.exists(progress_file):
            os.replace(temp, progress_file)
        else:
            os.rename(temp, progress_file)
    except Exception as e:
        print(f" Failed to save progress: {e}")
        if os.path.exists(temp):
            os.remove(temp)


def worker_process(
    gpu_id:        int,
    sample_subset: List[Dict],
    output_dir:    str,
    args:          Dict,
    resume:        bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    print(f"[GPU {gpu_id}] building model")
    try:
        model, device_obj = build_sam3_image_predictor(device, args.get("bpe_path"))
        print(f"[GPU {gpu_id}] model ready on {device_obj}")
    except Exception as e:
        print(f"[GPU {gpu_id}] model load failed: {e}")
        import traceback; traceback.print_exc()
        return None
    
    worker_output_file   = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
    worker_progress_file = os.path.join(output_dir, f"gpu_{gpu_id}_progress.json")

    print(f"\n[GPU {gpu_id}] Assigned {len(sample_subset)} images")
    processed_ids = set()
    results = {
        "per_image": [], "errors": [],
        "absolute_errors": [], "squared_errors": [],
    }

    if os.path.exists(worker_output_file):
        try:
            with open(worker_output_file) as f:
                old = json.load(f)
            results["per_image"]       = old.get("per_image", [])
            results["errors"]          = old.get("errors", [])
            results["absolute_errors"] = old.get("absolute_errors", [])
            results["squared_errors"]  = old.get("squared_errors", [])
            for img in results["per_image"]:
                processed_ids.add(img["image_id"])
            print(f"[GPU {gpu_id}] 📂 Loaded {len(results['per_image'])} old results")
        except Exception as e:
            print(f"[GPU {gpu_id}] ⚠️  Could not load old results: {e}")

    import  sam3count_eval as count_module
    _orig_build = count_module.build_sam3_image_predictor

    def _use_prebuilt(device_str, bpe_path=None):
        return model, device_obj

    count_module.build_sam3_image_predictor = _use_prebuilt
    
    processed_count = 0
    failed_count    = 0
    text_prompt     = args.get("text_prompt", "person")

    try:
        for idx, sample in enumerate(sample_subset):
            img_id = sample["image_id"]

            if img_id in processed_ids:
                print(f"[GPU {gpu_id}] Skipping already-processed image_id:{img_id}")
                continue

            image_path = sample["image_path"]
            gt_path    = sample["gt_path"]
            filename   = sample["filename"]
            part_tag   = f"Part_{sample['part']}"

            if not os.path.exists(image_path):
                print(f"[GPU {gpu_id}] ⚠️  Image not found: {image_path}")
                failed_count += 1
                continue

            gt_count = read_gt_count(gt_path)

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
                    "part":       part_tag,
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
                    f"[GPU {gpu_id}] [{idx+1}/{len(sample_subset)}] "
                    f"{filename}  GT={gt_count}  Pred={pred_count}  "
                    f"AE={abs_error}"
                )

            except Exception as e:
                print(f"[GPU {gpu_id}] Error on {filename}: {e}")
                failed_count += 1
                import traceback; traceback.print_exc()
                continue

            # save progress
            if (idx + 1) % 10 == 0 and results["absolute_errors"]:
                mae  = np.mean(results["absolute_errors"])
                rmse = np.sqrt(np.mean(results["squared_errors"]))
                print(
                    f"[GPU {gpu_id}] progress {idx+1}/{len(sample_subset)} | "
                    f"Total={len(results['per_image'])} | "
                    f"MAE={mae:.2f} RMSE={rmse:.2f}"
                )
                save_progress(worker_progress_file, {
                    "processed_images": list(processed_ids),
                    "results":          results,
                    "timestamp":        datetime.now().isoformat(),
                    "gpu_id":           gpu_id,
                })

    finally:
        count_module.build_sam3_image_predictor = _orig_build

    save_progress(worker_progress_file, {
        "processed_images": list(processed_ids),
        "results":          results,
        "timestamp":        datetime.now().isoformat(),
        "gpu_id":           gpu_id,
    })

    print(f"\n[GPU {gpu_id}] finished")
    print(f"[GPU {gpu_id}]  new: {processed_count}  Failed: {failed_count}  "
          f"Total: {len(results['per_image'])}")

    if results["absolute_errors"]:
        mae  = np.mean(results["absolute_errors"])
        rmse = np.sqrt(np.mean(results["squared_errors"]))
        results["summary"] = {
            "gpu_id": gpu_id,
            "mae":    float(mae),
            "rmse":   float(rmse),
            "num_images": len(results["per_image"]),
        }
        print(f"[GPU {gpu_id}]    MAE={mae:.2f}  RMSE={rmse:.2f}")

    with open(worker_output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] 💾 Saved → {worker_output_file}")

    if processed_count == len(sample_subset) and os.path.exists(worker_progress_file):
        os.remove(worker_progress_file)

    return worker_output_file


def get_all_processed_images_global(
    output_dir:       str,
    num_possible_gpus: int = 16,
) -> Tuple[Set[int], Dict[int, int]]:
    all_processed   = set()
    gpu_image_counts = {}

    for gpu_id in range(num_possible_gpus):
        result_file = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                ids = [img["image_id"] for img in data.get("per_image", [])]
                all_processed.update(ids)
                gpu_image_counts[gpu_id] = len(ids)
                print(f" GPU {gpu_id}: {len(ids)} images in results")
            except Exception as e:
                print(f"  Could not read GPU {gpu_id} results: {e}")

    return all_processed, gpu_image_counts


def merge_results(
    output_files:  List[str],
    final_output:  str,
    summary_output: str,
) -> Dict:
    print(f"\n{'='*80}")
    print(f"merging results from {len(output_files)} worker")
    print(f"{'='*80}")

    all_results = {
        "per_image": [], "errors": [],
        "absolute_errors": [], "squared_errors": [],
    }
    seen_ids       = set()
    duplicate_count = 0

    for gpu_id, fpath in enumerate(output_files):
        if not os.path.exists(fpath):
            print(f" {fpath} not found")
            continue
        with open(fpath) as f:
            worker = json.load(f)
        n = len(worker.get("per_image", []))
        print(f"GPU {gpu_id}: {n} images")

        for img in worker.get("per_image", []):
            iid = img["image_id"]
            if iid in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(iid)
                all_results["per_image"].append(img)
                all_results["errors"].append(img["error"])
                all_results["absolute_errors"].append(img["abs_error"])
                all_results["squared_errors"].append(float(img["error"] ** 2))

    if duplicate_count:
        print(f"{duplicate_count} duplicates skipped")

    if not all_results["absolute_errors"]:
        print("\n no processed images to aggregate.")
        with open(final_output, "w") as f:
            json.dump(all_results, f, indent=2)
        return all_results

    errors  = np.array(all_results["errors"])
    abs_errors = np.array(all_results["absolute_errors"])
    mae  = float(abs_errors.mean())
    rmse   = float(np.sqrt((errors ** 2).mean()))
    mean_error = float(errors.mean())

    part_stats: Dict[str, Dict] = {}
    for img in all_results["per_image"]:
        p = img.get("part", "unknown")
        part_stats.setdefault(p, {"abs_errors": [], "errors": []})
        part_stats[p]["abs_errors"].append(img["abs_error"])
        part_stats[p]["errors"].append(img["error"])

    per_part_metrics = {}
    for p, data in part_stats.items():
        ae = np.array(data["abs_errors"])
        er = np.array(data["errors"])
        per_part_metrics[p] = {
            "mae":        float(ae.mean()),
            "rmse":       float(np.sqrt((er ** 2).mean())),
            "mean_error": float(er.mean()),
            "num_images": len(ae),
        }

    all_results["summary"] = {
        "mae":          mae,
        "rmse":         rmse,
        "mean_error":   mean_error,
        "num_images":   len(all_results["per_image"]),
        "unique_images": len(seen_ids),
        "timestamp":    datetime.now().isoformat(),
    }

    summary_report = {
        "overall_metrics": {
            "mae":           mae,
            "rmse":          rmse,
            "mean_error":    mean_error,
            "total_images":  len(all_results["per_image"]),
            "unique_images": len(seen_ids),
            "timestamp":     datetime.now().isoformat(),
        },
        "per_part_metrics": per_part_metrics,
    }

    with open(summary_output, "w") as f:
        json.dump(summary_report, f, indent=2)
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("final results")
    print(f"\nSummary : {summary_output}")
    print(f"Detailed: {final_output}")

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="multi-GPU ShanghaiTech evaluation with SAM3Count"
    )
    # dataset
    parser.add_argument("--data_r", type=str, required=True,
                        help="Root of the ShanghaiTech dataset")
    parser.add_argument("--part",  type=str, default="A",
                        choices=["A", "B", "both"],
                        help="Which part to evaluate (default: A)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "both"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--text_prompt", type=str, default="person",
                        help="text prompt (default: 'person')")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="Specific GPU IDs")
    parser.add_argument("--output_json",  type=str,
                        default="partB_results.json",
                        help="Path for per-image results")
    parser.add_argument("--summary_json", type=str,
                        default="partB_summary.json ",
                        help="lightweight summary")
    parser.add_argument("--temp_dir",     type=str,
                        default="./temp_shanghaitech",
                        help="temporary dir for per-GPU result files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous (partial) run")

    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--iom_threshold", type=float, default=0.5)
    parser.add_argument("--min_obj_area",  type=int,   default=0)
    parser.add_argument("--bpe_path", type=str, default=None,
                        help="BPE vocab path (optional)")

    parser.add_argument("--max_images", type=int, default=None,
                        help="cap on total images (for quick testing)")

    args = parser.parse_args()

    gpu_ids  = args.gpu_ids if args.gpu_ids is not None else list(range(args.num_gpus))
    num_gpus = len(gpu_ids)

    os.makedirs(args.temp_dir, exist_ok=True)

    print("Loading ShanghaiTech dataset …")
    all_samples = load_dataset(args.data_r, args.part, args.split)

    if args.max_images:
        all_samples = all_samples[:args.max_images]

    total = len(all_samples)
    if total == 0:
        print("No samples found")
        sys.exit(1)

    globally_processed: Set[int] = set()
    if args.resume:
        globally_processed, gpu_counts = get_all_processed_images_global(args.temp_dir)
        if globally_processed:
            print(f"  found {len(globally_processed)} already-processed images")
            for gid, cnt in sorted(gpu_counts.items()):
                print(f"  GPU {gid}: {cnt} images")
        else:
            print("  no prior progress found, starting fresh")

    if len(globally_processed) >= total:
        print("\n all images already processed — merging existing results …")
        output_files = [
            os.path.join(args.temp_dir, f"gpu_{g}_results.json")
            for g in range(16)
            if os.path.exists(os.path.join(args.temp_dir, f"gpu_{g}_results.json"))
        ]
        if output_files:
            merge_results(output_files, args.output_json, args.summary_json)
        else:
            print(" no GPU result files found to merge.")
        return

    remaining = [s for s in all_samples if s["image_id"] not in globally_processed]


    print(f"  Total samples   : {total}")
    print(f"  Already done    : {len(globally_processed)}")
    print(f"  Remaining       : {len(remaining)}")

    if not remaining:
        print("nothing left to process.")
        return

    base     = len(remaining) // num_gpus
    rem      = len(remaining) %  num_gpus
    subsets  = []
    start    = 0
    for i in range(num_gpus):
        size  = base + (1 if i < rem else 0)
        subsets.append(remaining[start:start + size])
        start += size
        print(f"  GPU {gpu_ids[i]}: {len(subsets[-1])} images assigned")

    assert sum(len(s) for s in subsets) == len(remaining), "split mismatch!"

    worker_args = {
        "confidence_threshold": args.confidence_threshold,
        "iom_threshold":        args.iom_threshold,
        "min_obj_area":         args.min_obj_area,
        "text_prompt":          args.text_prompt,
        "bpe_path":             args.bpe_path,
    }

    print(f"\n{'='*80}")
    print(f"LAUNCHING {num_gpus} WORKER PROCESSES")
    print(f"{'='*80}")

    with mp.Pool(processes=num_gpus) as pool:
        async_results = []
        for i, (gpu_id, subset) in enumerate(zip(gpu_ids, subsets)):
            ar = pool.apply_async(
                worker_process,
                args=(gpu_id, subset, args.temp_dir, worker_args, args.resume),
            )
            async_results.append(ar)

        output_files = []
        for i, ar in enumerate(async_results):
            try:
                path = ar.get(timeout=7200)   # 2-hour per-GPU timeout
                if path:
                    output_files.append(path)
            except Exception as e:
                print(f" Worker {i} failed: {e}")
                import traceback; traceback.print_exc()

    # ── Merge ─────────────────────────────────────────────────────────────────
    if output_files:
        merge_results(output_files, args.output_json, args.summary_json)
        print("\n evaluation complete!")
    else:
        print("\n no output files to merge.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()