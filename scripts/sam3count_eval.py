#!/usr/bin/env python3

import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# SAM3 repo imports
from sam3.sam3 import build_sam3_img_model
from sam3.sam3.model.sam3_img_processor import Sam3Processor
from sam3.sam3.model.box_ops import box_xywh_to_cxcywh, box_cxcywh_to_xyxy
from sam3.sam3.visualization_utils import normalize_bbox


ENABLE_VISUALIZATION  = True
USE_MERGING           = False
STRICT_NMS_THRESHOLD  = 0.23

MODEL_CAP             = 90    
CAP_TRIGGER_RATIO     = 1.0    
CAP_STOP_RATIO        = 0.80   
CONF_BUMP             = 0.15   
MIN_TILE_AREA_RATIO   = 0.08   
MED_OVRLP_RATIO       = 0.30  


def gen_med_tile(
    img_w: int,
    img_h: int,
    min_t_size: int = 97,
    max_t_size: int = 1024,
) -> Tuple[List[Dict], int, int]:
   
    cols, rows   = 4, 2
    ov_lp_ratio = MED_OVRLP_RATIO

    t_size = int(max(img_w / cols, img_h / rows))
    t_size = max(min_t_size, min(t_size, max_t_size))
    ov_lp   = int(t_size * ov_lp_ratio)
    stde    = max(1, t_size - ov_lp)

    n_t_y = max(1, int(np.ceil(img_h / stde)))
    n_t_x = max(1, int(np.ceil(img_w  / stde)))

    t, seen = [], set()
    for ty in range(n_t_y):
        for tx in range(n_t_x):
            y     = min(ty * stde, max(0, img_h - t_size))
            x     = min(tx * stde, max(0, img_w  - t_size))
            y_end = min(y + t_size, img_h)
            x_end = min(x + t_size, img_w)
            key   = (x, y, x_end, y_end)
            if key not in seen:
                seen.add(key)
                t.append({"x": x, "y": y, "x_end": x_end, "y_end": y_end})

    print(f"\n{'='*70}")
    print(f"  initial pass")
    print(f"  Image     : {img_w}x{img_h} px")
    print(f"  Target    : {cols} cols x {rows} rows  |  ov_lp {ov_lp_ratio*100:.0f}%")
    print(f"  Tile size : {t_size}px   Overlap: {ov_lp}px   Stride: {stde}px")
    print(f"  Actual    : {n_t_x} cols x {n_t_y} rows = {len(t)} t")
    print(f"{'='*70}\n")

    return t, t_size, ov_lp



def split_tile_2x2(
    tile: Dict,
    img_w: int,
    img_h: int,
    sub_ov_lp_ratio: float = 0.15,
) -> List[Dict]:
    
    x,  y  = tile["x"],   tile["y"]
    xe, ye = tile["x_end"], tile["y_end"]
    w,  h  = xe - x,  ye - y

    half_w = w / 2
    half_h = h / 2
    pad_x  = int(max(half_w, half_h) * sub_ov_lp_ratio)
    pad_y  = pad_x

    sub_t, seen = [], set()
    for row in range(2):
        for col in range(2):
            sx     = int(x + col * half_w)
            sy     = int(y + row * half_h)
            sx_end = min(int(x + (col + 1) * half_w) + pad_x, img_w)
            sy_end = min(int(y + (row + 1) * half_h) + pad_y, img_h)
            # extend start backwards on inner edges
            if col > 0:
                sx = max(0, sx - pad_x)
            if row > 0:
                sy = max(0, sy - pad_y)

            key = (sx, sy, sx_end, sy_end)
            if key not in seen and sx_end > sx and sy_end > sy:
                seen.add(key)
                sub_t.append({"x": sx, "y": sy, "x_end": sx_end, "y_end": sy_end})

    return sub_t


def compute_iom(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    min_area = min(mask1.sum(), mask2.sum())
    return float(inter / min_area) if min_area > 0 else 0.0


def nms_iom_simple(
    masks:  List[np.ndarray],
    scores: List[float],
    boxes:  List[np.ndarray],
    iom_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    if not masks:
        return [], [], []

    order  = np.argsort(scores)[::-1]
    keep  = []
    supp = set()

    for i in order:
        if i in supp:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in supp:
                continue
            if compute_iom(masks[i], masks[j]) >= iom_threshold:
                supp.add(j)

    return [masks[i] for i in keep], [scores[i] for i in keep], [boxes[i] for i in keep]


def nms_on_detections(
    detections:    List[Dict],
    iom_threshold: float = 0.5,
) -> List[Dict]:
    if not detections:
        return []

    threshold = STRICT_NMS_THRESHOLD if USE_MERGING else iom_threshold
    masks_k, scores_k, boxes_k = nms_iom_simple(
        [d["mask"]  for d in detections],
        [d["score"] for d in detections],
        [d["box"]   for d in detections],
        threshold,
    )
    return [
        {"mask": m, "box": b, "score": s, "area": int(m.sum())}
        for m, s, b in zip(masks_k, scores_k, boxes_k)
    ]


def run_tile_inference(
    model,
    device: torch.device,
    img: Image.Image,
    tile:  Dict,
    text_prompt: str,
    confidence_threshold: float,
) -> List[Dict]:
   
    x_start, y_start = tile["x"], tile["y"]
    x_end,   y_end   = tile["x_end"], tile["y_end"]
    img_w,   img_h   = img.size

    tile_img  = img.crop((x_start, y_start, x_end, y_end))
    processor = Sam3Processor(model, confidence_threshold=0.0, device=device)
    state  = processor.set_img(tile_img)
    state  = processor.set_text_prompt(state=state, prompt=text_prompt)

    if "masks" not in state or len(state["masks"]) == 0:
        return []

    detections = []
    for i in range(len(state["masks"])):
        score = float(state["scores"][i])
        if score < confidence_threshold:
            continue

        mask = state["masks"][i]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask.squeeze()

        mask_bool = (
            (mask_np > 0.5)
            if mask_np.dtype in [np.float32, np.float64]
            else mask_np.astype(bool)
        )

        # Embed mask into the full-img canvas
        f_mask = np.zeros((img_h, img_w), dtype=bool)
        f_mask[y_start:y_end, x_start:x_end] = mask_bool

        box = state["boxes"][i]
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()

        box_global = [
            float(box[0] + x_start),
            float(box[1] + y_start),
            float(box[2] + x_start),
            float(box[3] + y_start),
        ]

        detections.append({
            "mask":  f_mask,
            "box":   box_global,
            "score": score,
            "area":  int(mask_bool.sum()),
        })

    return detections

def process_tile_recursive(
    model,
    device:   torch.device,
    img: Image.Image,
    tile:  Dict,
    text_prompt:  str,
    confidence: float,
    img_area:  int,
    iom_threshold:  float = 0.5,
    depth:  int   = 0,
    max_depth: int   = 8,
) -> List[Dict]:
    img_w, img_h = img.size
    tile_area       = (tile["x_end"] - tile["x"]) * (tile["y_end"] - tile["y"])
    tile_area_ratio = tile_area / max(1, img_area)
    indent          = "  " * depth
    trigger_count = int(MODEL_CAP * CAP_TRIGGER_RATIO) 
    stop_count    = int(MODEL_CAP * CAP_STOP_RATIO)      

    print(
        f"{indent}[d={depth}] tile [{tile['x']},{tile['y']}→"
        f"{tile['x_end']},{tile['y_end']}]  "
        f"area={tile_area_ratio:.3f}  conf={confidence:.3f}"
    )

    # tile too small
    if tile_area_ratio < MIN_TILE_AREA_RATIO:
        print(f"{indent} tile area {tile_area_ratio:.3f} < {MIN_TILE_AREA_RATIO} → stop")
        raw  = run_tile_inference(model, device, img, tile, text_prompt, confidence)
        return nms_on_detections(raw, iom_threshold)

    # recursion ceiling 
    if depth >= max_depth:
        print(f"{indent}  max depth {max_depth} reached → stop")
        raw  = run_tile_inference(model, device, img, tile, text_prompt, confidence)
        return nms_on_detections(raw, iom_threshold)

    # Run inference + per-tile NMS 
    raw     = run_tile_inference(model, device, img, tile, text_prompt, confidence)
    cleaned = nms_on_detections(raw, iom_threshold)
    n       = len(cleaned)

    print(f"{indent}  detections after NMS: {n}  (trigger>={trigger_count}, stop<{stop_count})")

    # Decide: subdivide or accept 
    if n >= trigger_count:
        # Still saturating the model, split deeper
        new_conf = min(confidence * (1.0 + CONF_BUMP), 0.99)
        print(f"{indent}  🔀 saturated — splitting 2x2, new conf={new_conf:.3f}")

        sub_t = split_tile_2x2(tile, img_w, img_h, sub_ov_lp_ratio=0.15)
        all_sub   = []
        for st in sub_t:
            sub_dets = process_tile_recursive(
                model, device, img, st,
                text_prompt, new_conf, img_area,
                iom_threshold, depth + 1, max_depth,
            )
            all_sub.extend(sub_dets)
        return all_sub

    else:
        # Count is comfortably below the cap (or between stop and trigger)
        status = "✅ below stop" if n < stop_count else "〰 between stop/trigger"
        print(f"{indent}  {status} → accepting {n} detections")
        return cleaned


def build_sam3_img_predictor(device_str: str = "cuda", bpe_path: str = None):
    device = torch.device(
        device_str if (torch.cuda.is_available() or device_str != "cuda") else "cpu"
    )
    print(f"Loading SAM3 IMAGE model on {device}...")

    if bpe_path is None:
        bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    checkpoint_path = (
        "checkpoints/sam3.pt"
    )
    model = build_sam3_img_model(
        enable_inst_interactivity=False,
        enable_segmentation=True,
        bpe_path=bpe_path,
        eval_mode=True,
        load_from_HF=True,
        checkpoint_path="checkpoints/sam3.pt",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print("Ignored missing keys:", [k for k in missing if k.startswith("segmentation_head.")])

    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f" Model loaded on {device}")
    return model, device

class VisualizationHelper:

    def __init__(self, output_dir: str = "visualize_demo"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        

    def visualize_tile_grid(self, img_pil: Image.Image, t: List[Dict],
                            filename: str = "01_tile_grid.png"):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img_pil)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(t), 1)))
        for idx, t in enumerate(t):
            w = t["x_end"] - t["x"]
            h = t["y_end"] - t["y"]
            ax.add_patch(patches.Rectangle(
                (t["x"], t["y"]), w, h,
                linew=3, edgecolor=colors[idx % len(colors)],
                facecolor="none", linestyle="--", alpha=0.8,
            ))
            ax.text(
                t["x"] + w // 2, t["y"] + h // 2, f"T{idx+1}",
                color="white", fontsize=10, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor=colors[idx % len(colors)], alpha=0.8),
            )
        ax.set_title(f"Initial Tiling: {len(t)} t", fontsize=18, weight="bold")
        ax.axis("off")
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  ✓ Saved: {path}")

    def visualize_final_result(self, img_pil: Image.Image,
                               detections: List[Dict],
                               filename: str = "02_final_result.png"):
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(img_pil)
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linew=2, edgecolor="lime", facecolor="none", alpha=0.9,
            ))
        ax.set_title(
            f"Final count: {len(detections)} objects",
            fontsize=18, weight="bold", color="lime",
        )
        ax.axis("off")
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f" Saved: {path}  (total: {len(detections)} objects)")


def visualize_detections_on_img(
    frame_np:      np.ndarray,
    detections:    List[Dict[str, Any]],
    count:         int,
    save_path:     str,
    show_id:       bool = False,
    id_font_size:  int  = 18,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img     = frame_np.copy().astype(np.uint8)
    overlay = img.copy()

    rng    = np.random.default_rng(0)
    colors = [rng.integers(0, 256, size=3, dtype=np.uint8) for _ in range(len(detections))]

    for i, det in enumerate(detections):
        mask = det.get("mask")
        box  = det.get("box")
        if mask is None:
            continue
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * colors[i]).astype(np.uint8)
        if box is not None:
            x1, y1, x2, y2 = (
                max(int(box[0]), 0), max(int(box[1]), 0),
                min(int(box[2]), img.shape[1]-1), min(int(box[3]), img.shape[0]-1),
            )
            overlay[y1:y1+2,  x1:x2+1] = [255, 255, 255]
            overlay[y2-1:y2+1, x1:x2+1] = [255, 255, 255]
            overlay[y1:y2+1,  x1:x1+2] = [255, 255, 255]
            overlay[y1:y2+1, x2-1:x2+1] = [255, 255, 255]

    vis  = (0.5 * img + 0.5 * overlay).astype(np.uint8)
    pil_vis = Image.fromarray(vis)
    draw    = ImageDraw.Draw(pil_vis)

    if show_id:
        try:
            id_font = ImageFont.truetype("arial.ttf", id_font_size)
        except Exception:
            try:
                id_font = ImageFont.truetype("DejaVuSans-Bold.ttf", id_font_size)
            except Exception:
                id_font = ImageFont.load_default()
        for i, det in enumerate(detections):
            ys, xs = np.where(det["mask"])
            if len(xs) == 0:
                continue
            cx, cy  = int(np.mean(xs)), int(np.mean(ys))
            id_text = str(i + 1)
            bb      = draw.textbbox((0, 0), id_text, font=id_font)
            tw, th  = bb[2]-bb[0], bb[3]-bb[1]
            px, py  = cx - tw//2, cy - th//2
            draw.rectangle([px-8, py-8, px+tw+8, py+th+8], fill=(0,0,0,200))
            draw.text((px, py), id_text, fill=(255,255,255), font=id_font)

    text = f"Count: {count}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.rectangle([0, 0, tw+8, th+8], fill=(0,0,0,160))
    draw.text((4, 4), text, fill=(255,255,255), font=font)

    pil_vis.save(save_path)
    print(f" Saved visualization: {save_path}")

def update_counts_json(output_file: str, img_path: str, input_text: str, num_objects: int):
    if not output_file:
        return
    data = {}
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
    data.setdefault(img_path, {})[input_text] = int(num_objects)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated: {output_file}")

def count_objects_in_img(
    img_path:  str,
    text_prompt:  str,
    device_str:           str   = "cuda",
    confidence_threshold: float = 0.35,
    min_obj_area:         int   = 0,
    iom_threshold:        float = 0.5,
    apply_nms:            bool  = True,
    bpe_path:             str   = None,
    visualizer                  = None,
) -> Tuple[int, List[Dict[str, Any]], np.ndarray]:
    model, device = build_sam3_img_predictor(device_str, bpe_path)

    img  = Image.open(img_path).convert("RGB")
    frame_np = np.array(img)
    w, h = img.size
    img_area = w * h

    print(f"  Image : {w}x{h} px  ({img_area:,} px total)")

    t, t_size, ov_lp = gen_med_tile(w, h)

    if visualizer:
        visualizer.visualize_tile_grid(img, t)

    all_detections: List[Dict] = []
    for tile_idx, tile in enumerate(t):
        tile_area = (tile["x_end"] - tile["x"]) * (tile["y_end"] - tile["y"])
        print(
            f"\n── Tile {tile_idx+1}/{len(t)}  "
            f"[{tile['x']},{tile['y']}→{tile['x_end']},{tile['y_end']}]  "
            f"area={tile_area:,} px ({tile_area/img_area*100:.1f}%)"
        )
        tile_dets = process_tile_recursive(
            model=model, device=device, img=img,
            tile=tile, text_prompt=text_prompt,
            confidence=confidence_threshold,
            img_area=img_area,
            iom_threshold=iom_threshold,
            depth=0,
        )
        all_detections.extend(tile_dets)
        
    if apply_nms and all_detections:
        all_detections = nms_on_detections(all_detections, iom_threshold)
        
    if min_obj_area > 0:
        all_detections = [d for d in all_detections if d["area"] >= min_obj_area]
        
    num_objects = len(all_detections)

    if visualizer and all_detections:
        visualizer.visualize_final_result(img, all_detections)

    print(f" FINAL COUNT : {num_objects} objects")
  
    return num_objects, all_detections, frame_np

def get_args_parser():
    parser = argparse.ArgumentParser(
        "SAM3 Dense Scene Counting — Recursive Adaptive Tiling",
        add_help=True,
    )
    parser.add_argument("--img_path",           type=str,   required=True)
    parser.add_argument("--input_text",           type=str,   required=True,
                        help="Text prompt, e.g. 'person' or 'head'")
    parser.add_argument("--output_file",          type=str,   default="",
                        help="JSON file to accumulate counts")
    parser.add_argument("--device",               type=str,   default="cuda")
    parser.add_argument("--bpe_path",             type=str,   default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.45,
                        help="Initial confidence threshold (default: 0.35)")
    parser.add_argument("--min_obj_area",         type=int,   default=0,
                        help="Min pixel area to keep a detection (default: 0)")
    parser.add_argument("--iom_threshold",        type=float, default=0.5,
                        help="IoM threshold for NMS (default: 0.5)")
    parser.add_argument("--no_nms",               action="store_true",
                        help="Disable global NMS")
    parser.add_argument("--save_vis",             action="store_true",
                        help="Save an overlay visualization img")
    parser.add_argument("--vis_path",             type=str,   default="",
                        help="Path for the visualization img/directory")
    parser.add_argument("--show_id",              action="store_true",
                        help="Overlay detection IDs on visualization")
    parser.add_argument("--id_font_size",         type=int,   default=18)
    return parser


def main():
    parser = get_args_parser()
    args  = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    device_str = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device_str = "cpu"

    print(f"  Image      : {args.img_path}")
    print(f"  Prompt     : '{args.input_text}'")
    
    visualizer = None
    if ENABLE_VISUALIZATION:
        base = (
            args.vis_path if os.path.isdir(args.vis_path)
            else (os.path.dirname(args.vis_path) or ".") if args.vis_path
            else (os.path.dirname(args.img_path) or ".")
        )
        visualizer = VisualizationHelper(output_dir=os.path.join(base, "visualization_dense2"))

    num_objects, detections, frame_np = count_objects_in_img(
        img_path=args.img_path,
        text_prompt=args.input_text,
        device_str=device_str,
        confidence_threshold=args.confidence_threshold,
        min_obj_area=args.min_obj_area,
        iom_threshold=args.iom_threshold,
        apply_nms=not args.no_nms,
        bpe_path=args.bpe_path,
        visualizer=visualizer,
    )

    print(f"Final Count: {num_objects}")

    if detections:
        for i, det in enumerate(detections, 1):
            print(f"  #{i:>4}: score={det['score']:.3f}  area={det['area']} px")

    if args.output_file:
        update_counts_json(args.output_file, args.img_path, args.input_text, num_objects)

    if args.save_vis:
        vis_path = (
            args.vis_path if args.vis_path
            else os.path.splitext(args.img_path)[0] + "_dense_vis.png"
        )
        visualize_detections_on_img(
            frame_np, detections, num_objects, vis_path,
            args.show_id, args.id_font_size,
        )

    print("\nDone!\n")

if __name__ == "__main__":
    main()