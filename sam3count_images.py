#!/usr/bin/env python3

"""
SAM3 Object Counting with SMART DENSITY-AWARE ADAPTIVE TILING

IMPROVED MULTI-FACTOR DENSITY CALCULATION:
Distinguishes between:
- 6 large objects covering 60% → SPARSE (no tiling)  
- 100 small objects covering 60% → DENSE (tiling applied)

THREE FACTORS:
1. Coverage ratio: Total area occupied by bounding boxes
2. Object count: Number of detected objects (normalized)
3. Average object size: Size of each object relative to image

TILING DECISION RULES:
- >25 objects → Use tiling
- High coverage (>50%) + small objects (<5% avg) → Use tiling  
- Very small objects (<1% avg) → Use tiling
- Otherwise → No tiling

WORKFLOW:
1. Run SAM3 on full image (no tiling)
2. Calculate smart multi-factor density
3. If dense: Apply semantic-aware tiling and re-predict
4. Apply semantic-aware NMS
"""

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
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
from sam3.sam3.model.box_ops import box_xywh_to_cxcywh, box_cxcywh_to_xyxy
from sam3.sam3.visualization_utils import normalize_bbox

# ============================================================
# Configuration
# ============================================================
USE_MERGING = False
STRICT_NMS_THRESHOLD = 0.23
ENABLE_VISUALIZATION = True



# ====================================================================
# Adaptive Tiling Calculator
# ====================================================================
def calculate_adaptive_tile_params(
    image_width: int,
    image_height: int,
    tiling_rule: str = "MEDIUM",   # ← was: prompt_type: str = "INDIVIDUAL"
    min_tile_size: int = 97,
    max_tile_size: int = 1024,
) -> Tuple[int, int]:

    configs = {
        "LARGE":  {"cols": 2, "rows": 1, "overlap_ratio": 0.35,
                   "description": "Coarse grid for many larger objects"},
        "MEDIUM": {"cols": 4, "rows": 2, "overlap_ratio": 0.30,
                   "description": "Balanced grid for mid-size dense scenes"},
        "SMALL":  {"cols": 6, "rows": 4, "overlap_ratio": 0.25,
                   "description": "Fine grid for many small objects"},
    }
    config = configs.get(tiling_rule, configs["MEDIUM"])

    # Calculate tile size based on grid configuration
    tile_width = image_width / config['cols']
    tile_height = image_height / config['rows']

    # Use the larger dimension to ensure coverage
    tile_size = int(max(tile_width, tile_height))

    # Clamp to min/max bounds
    tile_size = max(min_tile_size, min(tile_size, max_tile_size))

    # Calculate overlap
    overlap = int(tile_size * config['overlap_ratio'])

    # Calculate actual grid that will be created
    stride = tile_size - overlap
    actual_cols = int(np.ceil(image_width / stride))
    actual_rows = int(np.ceil(image_height / stride))
    actual_tiles = actual_cols * actual_rows

    print(f"\n{'='*70}")
    print(f"🤖 ADAPTIVE TILING CONFIGURATION")
    print(f"{'='*70}")
    print(f"   Image size: {image_width}x{image_height} pixels")
    print(f"   Strategy: {config['description']}")
    print(f"   ")
    print(f"   Target grid: {config['cols']} cols x {config['rows']} rows")
    print(f"   Overlap ratio: {config['overlap_ratio']*100:.0f}%")
    print(f"   ")
    print(f"   📐 Calculated tile size: {tile_size}px")
    print(f"   🔗 Calculated overlap: {overlap}px")
    print(f"   📊 Actual grid: {actual_cols} cols x {actual_rows} rows → {actual_tiles} tiles")
    print(f"{'='*70}\n")

    return tile_size, overlap



# ============================================================
# SMART Multi-Factor Density Calculation (NEW & IMPROVED)
# ============================================================

def calculate_density_smart(
    boxes: List[np.ndarray], 
    image_width: int, 
    image_height: int
) -> Tuple[float, bool, Dict[str, Any]]:
    """
    IMPROVED density calculation with THREE factors

    Correctly distinguishes:
    - 6 large objects covering 60% → LOW density → No tiling
    - 100 small objects covering 60% → HIGH density → Use tiling

    Args:
        boxes: List of [x1, y1, x2, y2] bounding boxes
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        density_score: Combined metric (0.0 to 1.0+)
        use_tiling: Boolean decision
        metrics: Dict with detailed breakdown
    """
    if len(boxes) == 0:
        return 0.0, False, {
            'coverage': 0.0,
            'object_count': 0,
            'avg_object_size_ratio': 0.0,
            'density_score': 0.0,
            'object_count_score': 0.0,
            'size_score': 0.0,
            'decision': 'No objects detected',
            'tiling_rule': 'NONE',
            'use_tiling': False,
        }

    image_area = image_width * image_height

    # Factor 1: Coverage ratio (total box area / image area)
    total_box_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        box_area = width * height
        total_box_area += box_area

    coverage = total_box_area / image_area

    # Factor 2: Object count (normalized, maxes at 50 objects)
    object_count = len(boxes)
    object_count_score = min(object_count / 50.0, 1.0)

    # Factor 3: Average object size (smaller objects = higher density)
    avg_box_size = total_box_area / object_count
    avg_size_ratio = avg_box_size / image_area

    # Size score: 1.0 for small objects, 0.0 for large objects
    # Objects >10% of image get score ~0, objects <1% get score ~1
    size_score = 1.0 - min(avg_size_ratio / 0.1, 1.0)

    # Combined density with weights:
    # 30% coverage, 50% object count, 20% size
    density_score = 0.3 * coverage + 0.5 * object_count_score + 0.2 * size_score

    # ========================================
    # TILING DECISION LOGIC
    # ========================================
    use_tiling = False
    tiling_rule = "NONE"
    decision_reason = ""

    if object_count > 90:
        # Rule 1: Many objects always benefit from tiling
        use_tiling = True
        decision_reason = f"Many objects ({object_count} > 90)"
        tiling_rule = "LARGE"

    # if object_count <= 3:
    #         # Rule 1: Many objects always benefit from tiling
    #         use_tiling = True
    #         decision_reason = f"Less than normal for this benchmark `({object_count} < 3) may be uncertain"
    #         tiling_rule = "LARGE"

    elif coverage > 0.4 and avg_size_ratio < 0.02:
        # Rule 2: High coverage + small objects = dense scene
        use_tiling = True
        decision_reason = f"High coverage ({coverage:.1%}) with small objects (avg {avg_size_ratio:.1%})"
        tiling_rule = "MEDIUM"

    elif density_score > 0.7 and avg_size_ratio < 0.01:
        # Rule 3: Very small objects need tiling
        use_tiling = True
        decision_reason = f"Very small objects (avg {avg_size_ratio:.2%} of image)"
        tiling_rule = "SMALL"

    else:
        # No tiling needed
        use_tiling = False
        tiling_rule = "NONE"
        if object_count <= 90:
            decision_reason = f"Few large objects ({object_count}, avg {avg_size_ratio:.1%})"
        elif avg_size_ratio > 0.1:
            decision_reason = f"Large objects (avg {avg_size_ratio:.1%} each)"
        else:
            decision_reason = f"Moderate density ({object_count} objects)"

    metrics = {
        'coverage': float(coverage),
        'object_count': int(object_count),
        'avg_object_size_ratio': float(avg_size_ratio),
        'density_score': float(density_score),
        'object_count_score': float(object_count_score),
        'size_score': float(size_score),
        'decision': decision_reason,
        'use_tiling': use_tiling,
        "tiling_rule": tiling_rule,
    }

    return float(density_score), use_tiling, metrics


class VisualizationHelper:
    """Helper class for visualizing the tiling and detection process"""

    def __init__(self, output_dir="visualization_semantic_aware"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"VISUALIZATION MODE ENABLED")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

    def visualize_tile_grid(self, image_pil, tiles, filename="01_tile_grid.png"):
        """Visualize how the image is split into tiles"""
        print(f"\n[STEP 1] Visualizing tile grid ({len(tiles)} tiles)...")

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(image_pil)

        colors = plt.cm.tab10(np.linspace(0, 1, len(tiles)))

        for idx, tile in enumerate(tiles):
            rect = patches.Rectangle(
                (tile['x'], tile['y']),
                tile['width'], tile['height'],
                linewidth=3, edgecolor=colors[idx], facecolor='none', linestyle='--', alpha=0.8
            )
            ax.add_patch(rect)

            ax.text(
                tile['x'] + tile['width']//2,
                tile['y'] + tile['height']//2,
                f"Tile {idx+1}\n({tile['width']}x{tile['height']})",
                color='white', fontsize=12, weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.8, edgecolor='white', linewidth=2)
            )

        ax.set_title(f"Step 1: Image Split into {len(tiles)} Overlapping Tiles",
                    fontsize=18, weight='bold', pad=20)
        ax.axis('off')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Saved: {save_path}")

    def visualize_tile_detections(self, image_pil, tiles, all_tile_detections,
                                 filename_prefix="02_tile_detections"):
        """Visualize detections for each individual tile"""
        print(f"\n[STEP 2] Visualizing detections per tile...")

        num_tiles = len(tiles)
        cols = min(3, num_tiles)
        rows = (num_tiles + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if num_tiles == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        total_detections = 0
        for idx, (tile, detections) in enumerate(zip(tiles, all_tile_detections)):
            tile_img = image_pil.crop((
                tile['x'], tile['y'],
                tile['x'] + tile['width'],
                tile['y'] + tile['height']
            ))

            ax = axes[idx]
            ax.imshow(tile_img)

            for det in detections:
                x, y, w, h = det['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='lime', facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)

            total_detections += len(detections)
            ax.set_title(f"Tile {idx+1}: {len(detections)} objects detected",
                        fontsize=12, weight='bold', color='lime')
            ax.axis('off')

        for idx in range(num_tiles, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(f"Step 2: Detection Results Per Tile (Total: {total_detections} detections)",
                    fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Saved: {save_path}")
        print(f"   → Total detections across all tiles: {total_detections}")

    def visualize_global_detections_before_merge(self, image_pil, global_detections,
                                                filename="02b_all_detections_global.png"):
        """Visualize all detections in global coordinates before merging"""
        print(f"\n[STEP 2b] Visualizing all detections in global coordinates...")

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(image_pil)

        for det in global_detections:
            x, y, w, h = det['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.5
            )
            ax.add_patch(rect)

        ax.set_title(f"Step 2b: All Detections in Global Coordinates\n{len(global_detections)} detections (before merging)",
                    fontsize=16, weight='bold')
        ax.axis('off')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Saved: {save_path}")

    def visualize_merging_process(self, image_pil, detections_before_nms,
                                 detections_after_nms, filename="03_merging_process.png"):
        """Visualize before and after NMS merging"""
        print(f"\n[STEP 3] Visualizing semantic-aware NMS merging process...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        ax1.imshow(image_pil)
        for det in detections_before_nms:
            x, y, w, h = det['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.2
            )
            ax1.add_patch(rect)
        ax1.set_title(f"Before Semantic NMS\n{len(detections_before_nms)} detections",
                     fontsize=14, weight='bold', color='orange')
        ax1.axis('off')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        ax2.imshow(image_pil)
        for det in detections_after_nms:
            x, y, w, h = det['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=3, edgecolor='lime', facecolor='none', alpha=0.9
            )
            ax2.add_patch(rect)
        ax2.set_title(f"After Semantic NMS\n{len(detections_after_nms)} detections",
                     fontsize=14, weight='bold', color='lime')
        ax2.axis('off')

        fig.suptitle(f"Step 3: Intelligent Semantic NMS (Removed {len(detections_before_nms) - len(detections_after_nms)} detections)",
                    fontsize=16, weight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Saved: {save_path}")

    def visualize_final_result(self, image_pil, final_detections,
                              filename="04_final_result.png"):
        """Visualize the final merged detections"""
        print(f"\n[STEP 4] Visualizing final result...")

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.imshow(image_pil)

        for idx, det in enumerate(final_detections):
            x, y, w, h = det['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=3, edgecolor='lime', facecolor='none', alpha=0.9
            )
            ax.add_patch(rect)

            ax.text(x + w/2, y - 5, str(idx+1),
                   color='lime', fontsize=10, weight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))

        ax.set_title(f"Step 4: Final Result\nTotal Objects Detected: {len(final_detections)}",
                    fontsize=18, weight='bold', color='lime', pad=20)
        ax.axis('off')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Saved: {save_path}")

        print(f"\n{'='*60}")
        print(f"VISUALIZATION COMPLETE!")
        print(f"Final count: {len(final_detections)} objects")
        print(f"{'='*60}\n")

# ============================================================
# Density-Aware Tiled Detector
# ============================================================


class DensityAwareTiledDetector:
    """Handles uniform tiling + tile-level NMS (ROI filtering is applied outside this class)"""

    def __init__(self, tile_size=512, overlap=128, iou_threshold=0.5):
        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        print(f"ROI/Uniform Density-Aware Tiling: tile_size={tile_size}, overlap={overlap}")

    def generate_tiles(self, height, width, force_tiling=True):
        """
        Generate UNIFORM tiles only.
        ROI filtering is applied later in process_image_with_density_tiling(...).
        """
        return self._generate_uniform_tiles(height, width, force_tiling)

    def _deduplicate_tiles(self, tiles):
        """Remove duplicate tiles"""
        unique = []
        seen = set()
        for tile in tiles:
            key = (tile['x'], tile['y'], tile['x_end'], tile['y_end'])
            if key not in seen:
                seen.add(key)
                unique.append(tile)
        return unique

    def _generate_uniform_tiles(self, height, width, force_tiling):
        """Uniform tiling over the full image"""
        tiles = []
        stride = self.tile_size - self.overlap

        # Safety guard: avoid zero/negative stride
        if stride <= 0:
            stride = max(1, self.tile_size // 2)
            print(f"  ⚠️ Invalid stride from tile_size={self.tile_size}, overlap={self.overlap}; using stride={stride}")

        num_tiles_y = max(1, int(np.ceil(height / stride)))
        num_tiles_x = max(1, int(np.ceil(width / stride)))

        if force_tiling:
            if num_tiles_y == 1 and num_tiles_x == 1:
                if height >= self.tile_size // 2 and width >= self.tile_size // 2:
                    num_tiles_y = 2
                    num_tiles_x = 2

        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                y = min(ty * stride, max(0, height - self.tile_size))
                x = min(tx * stride, max(0, width - self.tile_size))
                y_end = min(y + self.tile_size, height)
                x_end = min(x + self.tile_size, width)

                tiles.append({
                    'x': x, 'y': y,
                    'x_end': x_end, 'y_end': y_end
                })

        return self._deduplicate_tiles(tiles)

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def nms_across_tiles(self, detections):
        """Suppress duplicate detections from overlapping tiles"""
        if len(detections) == 0:
            return []

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        suppressed = set()

        for i, det1 in enumerate(detections):
            if i in suppressed:
                continue

            keep.append(det1)

            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue

                det2 = detections[j]
                iou = self.compute_iou(det1['box'], det2['box'])
                if iou > self.iou_threshold:
                    suppressed.add(j)

        return keep


# ============================================================
# Semantic-Aware NMS Functions
# ============================================================

def compute_iom(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Minimum (IoM)"""
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()

    if area1 == 0 or area2 == 0:
        return 0.0

    min_area = min(area1, area2)
    iom = intersection / min_area
    return float(iom)


def nms_iom_simple(
    masks: List[np.ndarray],
    scores: List[float],
    boxes: List[np.ndarray],
    iom_threshold: float = 0.5
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Simple NMS using IoM (ORIGINAL LOGIC - for CONTAINER mode)
    Keeps highest scoring detections, suppresses overlaps
    """
    if len(masks) == 0:
        return [], [], []

    indices = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()

    for i in indices:
        if i in suppressed:
            continue

        keep.append(i)

        for j in indices:
            if j == i or j in suppressed:
                continue

            iom = compute_iom(masks[i], masks[j])
            if iom >= iom_threshold:
                suppressed.add(j)

    filtered_masks = [masks[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]
    filtered_boxes = [boxes[i] for i in keep]

    return filtered_masks, filtered_scores, filtered_boxes


def build_sam3_image_predictor(device_str: str = "cuda", bpe_path: str = None):
    """Build SAM3 image model"""
    device = torch.device(device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu")
    print(f"Loading SAM3 IMAGE model on {device}...")

    if bpe_path is None:
        bpe_path = "/project/advdls25/jowusu1/CountVid/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    checkpoint_path = "/project/advdls25/jowusu1/SAM33/sam3/sam3_fscd147_logs_long/checkpoints/checkpoint_29.pt"
    # checkpoint_path = "/project/advdls25/jowusu1/CountVid/fscd147/sam3.pt"
    model = build_sam3_image_model(
        enable_inst_interactivity=False,
        enable_segmentation=True,
        bpe_path=bpe_path,
        eval_mode=True,
        load_from_HF=True,
        checkpoint_path="/project/advdls25/jowusu1/CountVid/fscd147/sam3.pt",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Ignored missing keys:", [k for k in missing if k.startswith("segmentation_head.")])

    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"✅ Model loaded successfully on {device}")
    return model, device


# ============================================================
# Full Image Processing (NEW - No Tiling)
# ============================================================

def process_full_image(
    model,
    device: torch.device,
    image: Image.Image,
    text_prompt: str,
    confidence_threshold: float = 0.0,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Process entire image without tiling (first pass for density assessment)
    """
    width, height = image.size
    print(f"\n📸 Processing full image: {width}x{height} pixels (no tiling)")

    processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=text_prompt)

    if "masks" not in state or len(state["masks"]) == 0:
        print("   ⚠️  No detections found")
        return [], [], []

    num_detections = len(state["masks"])
    print(f"   ✅ Found {num_detections} initial detections")

    masks = []
    scores = []
    boxes = []

    for i in range(len(state["masks"])):
        mask = state["masks"][i]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask.squeeze()

        mask_bool = (mask_np > 0.5) if mask_np.dtype in [np.float32, np.float64] else mask_np.astype(bool)

        box = state["boxes"][i]
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()

        box_coords = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        score = float(state["scores"][i])

        masks.append(mask_bool)
        scores.append(score)
        boxes.append(box_coords)

    return masks, scores, boxes



def _clip_box_to_image(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width))
    y2 = max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _boxes_union_roi(boxes, width, height, pad_ratio=0.04, min_pad_px=16):
    """
    Build a single coverage ROI from cleaned detections.
    Returns ROI box [x1,y1,x2,y2] or None.
    """
    if not boxes:
        return None
    xs1 = [float(b[0]) for b in boxes]
    ys1 = [float(b[1]) for b in boxes]
    xs2 = [float(b[2]) for b in boxes]
    ys2 = [float(b[3]) for b in boxes]
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad_x = max(min_pad_px, pad_ratio * bw)
    pad_y = max(min_pad_px, pad_ratio * bh)
    return _clip_box_to_image([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def _box_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float(ix2 - ix1) * float(iy2 - iy1)


def _filter_tiles_by_roi(tiles, roi_box, min_intersection_ratio=0.05):
    """
    Keep tiles that overlap ROI by at least a small fraction of tile area.
    """
    if roi_box is None:
        return tiles
    kept = []
    for t in tiles:
        tb = [float(t['x']), float(t['y']), float(t['x_end']), float(t['y_end'])]
        inter = _box_intersection_area(tb, roi_box)
        ta = max(1.0, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        if inter / ta >= min_intersection_ratio or inter > 0:
            kept.append(t)
    return kept


def assess_stage1_detection_reliability(
    masks_initial,
    scores_initial,
    boxes_initial,
    detections_clean,
    image_width,
    image_height,count_after_nms: int = 0, 
):
    """
    Heuristic reliability score for deciding whether stage-1 cleaned detections
    are trustworthy enough to drive ROI-focused tiling.
    """
    raw_n = len(masks_initial)
    clean_n = len(detections_clean)
    raw_hit_cap = raw_n >= 150

    clean_scores = [float(d.get("score", 0.0)) for d in detections_clean]
    mean_clean_score = float(np.mean(clean_scores)) if clean_scores else 0.0

    areas = [float(d.get("area", 0.0)) for d in detections_clean]
    area_mean = float(np.mean(areas)) if areas else 0.0
    area_std = float(np.std(areas)) if len(areas) > 1 else 0.0
    area_cv = float(area_std / max(area_mean, 1e-6)) if areas else 0.0

    boxes_clean = [d["box"] for d in detections_clean if "box" in d]
    roi_box = _boxes_union_roi(boxes_clean, image_width, image_height)
    roi_area_ratio = 0.0
    if roi_box is not None:
        roi_area_ratio = ((roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1])) / max(1.0, image_width * image_height)

    clean_to_raw_ratio = float(clean_n / max(raw_n, 1)) if raw_n > 0 else 0.0

    score = 1.0
    reasons = []

    if raw_hit_cap:
        score -= 0.45
        reasons.append("raw detections hit/near model cap")
    if raw_n == 0 or clean_n == 0:
        score -= 0.65
        reasons.append("no usable stage-1 detections")
    if clean_n > 0 and mean_clean_score < 0.30:
        score -= 0.20
        reasons.append("low mean clean confidence")
    elif clean_n > 0 and mean_clean_score > 0.65:
        score += 0.05

    nms_n = max(count_after_nms, 1)   # avoid div-by-zero

    if clean_n <= 0.10 * nms_n and clean_to_raw_ratio < 0.10:
        score -= 0.15
        reasons.append(f"heavy pruning: only {clean_n}/{nms_n} ({clean_n/nms_n*100:.1f}%) survived confidence filter")
    if clean_n <= 0.08 * nms_n and area_cv > 2.5:
        score -= 0.10
        reasons.append(f"high size dispersion with low yield ({clean_n/nms_n*100:.1f}% of post-NMS)")
    if clean_n >= 0.05 * nms_n and roi_area_ratio > 0.90:
        score -= 0.10
        reasons.append("ROI spans almost whole image")
    if clean_n >= 0.05 * nms_n and roi_area_ratio < 0.01:
        score -= 0.10
        reasons.append("ROI extremely tiny")
    if clean_n <= 0.30 * nms_n and mean_clean_score < 0.25 and roi_area_ratio < 0.50:
        score -= 0.15
        reasons.append("low confidence + uncertain ROI")



    score = float(max(0.0, min(1.0, score)))
    use_roi_tiling = bool(score >= 0.55 and clean_n > 0 and roi_box is not None)

    metrics = {
        "reliability_score": score,
        "raw_count": int(raw_n),
        "clean_count": int(clean_n),
        "raw_hit_cap": bool(raw_hit_cap),
        "mean_clean_score": float(mean_clean_score),
        "clean_to_raw_ratio": float(clean_to_raw_ratio),
        "area_cv": float(area_cv),
        "roi_box": roi_box,
        "roi_area_ratio": float(roi_area_ratio),
        "use_roi_tiling": use_roi_tiling,
        "reasons": reasons if reasons else ["stage-1 looks usable"],
    }
    return metrics




def process_image_with_density_tiling(
    model,
    device: torch.device,
    image: Image.Image,
    text_prompt: str,
    tile_size: int = 512,
    overlap: int = 128,
    confidence_threshold: float = 0.5,
    visualizer=None,
    roi_box: Optional[List[float]] = None,
    roi_tile_intersection_ratio: float = 0.05,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Process image with ROI-only uniform tiling.
    If roi_box is provided: generate uniform full-image tiles, then keep only tiles overlapping ROI.
    If ROI is missing/weak or filters everything: fallback to full-image uniform tiling.
    """
    width, height = image.size

    tiled_detector = DensityAwareTiledDetector(tile_size=tile_size, overlap=overlap)

    # Always generate uniform tiles over the full image (NO attention-based tiles)
    base_tiles = tiled_detector.generate_tiles(height, width, force_tiling=True)
    total_tiles_before_roi = len(base_tiles)

    tiles = base_tiles
    if roi_box is not None:
        tiles = _filter_tiles_by_roi(
            base_tiles,
            roi_box,
            min_intersection_ratio=roi_tile_intersection_ratio
        )
        print(f"  🎯 ROI tile filtering: kept {len(tiles)}/{total_tiles_before_roi} tiles "
              f"(roi_intersection>={roi_tile_intersection_ratio:.2f})")

        if len(tiles) == 0:
            print("  ⚠️ ROI filtering removed all tiles; falling back to FULL-IMAGE uniform tiling")
            tiles = base_tiles
        else:
            print(f"  🎯 ROI-only tiling: {len(tiles)} tiles")
    else:
        print(f"  📐 No ROI provided; using FULL-IMAGE uniform tiling: {len(tiles)} tiles")

    # Add width/height for visualization
    if visualizer:
        for tile in tiles:
            tile['width'] = tile['x_end'] - tile['x']
            tile['height'] = tile['y_end'] - tile['y']
        visualizer.visualize_tile_grid(image, tiles)

    all_detections = []
    all_detections_by_tile = []
    total_raw_detections = 0

    for tile_idx, tile in enumerate(tiles):
        x_start, y_start = tile['x'], tile['y']
        x_end, y_end = tile['x_end'], tile['y_end']

        tile_img = image.crop((x_start, y_start, x_end, y_end))

        processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        state = processor.set_image(tile_img)
        state = processor.set_text_prompt(state=state, prompt=text_prompt)

        if "masks" not in state or len(state["masks"]) == 0:
            all_detections_by_tile.append([])
            continue

        num_tile_detections = len(state["masks"])
        total_raw_detections += num_tile_detections

        if num_tile_detections >= 190:
            print(f"   ⚠️ Tile {tile_idx+1}/{len(tiles)} hit limit: {num_tile_detections} detections")

        tile_detections = []
        for i in range(len(state["masks"])):
            mask = state["masks"][i]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = mask.squeeze()

            mask_bool = (mask_np > 0.5) if mask_np.dtype in [np.float32, np.float64] else mask_np.astype(bool)

            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y_start:y_end, x_start:x_end] = mask_bool

            box_tile = state["boxes"][i]
            if isinstance(box_tile, torch.Tensor):
                box_tile = box_tile.cpu().numpy()

            box_global = [
                float(box_tile[0] + x_start),
                float(box_tile[1] + y_start),
                float(box_tile[2] + x_start),
                float(box_tile[3] + y_start)
            ]

            score = float(state["scores"][i])

            detection = {
                'mask': full_mask,
                'box': box_global,
                'score': score,
                'bbox': [
                    float(box_tile[0]),
                    float(box_tile[1]),
                    float(box_tile[2] - box_tile[0]),
                    float(box_tile[3] - box_tile[1])
                ]
            }

            all_detections.append(detection)
            tile_detections.append(detection)

        all_detections_by_tile.append(tile_detections)

    print(f"   📊 Total raw detections: {total_raw_detections}")
    print(f"   📊 Before tile NMS: {len(all_detections)} detections")

    if visualizer and len(all_detections_by_tile) > 0:
        visualizer.visualize_tile_detections(image, tiles, all_detections_by_tile)

        global_dets = []
        for det in all_detections:
            box = det['box']
            global_dets.append({'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]]})
        visualizer.visualize_global_detections_before_merge(image, global_dets)

    if len(all_detections) > 0:
        all_detections = tiled_detector.nms_across_tiles(all_detections)
        print(f"   📊 After tile NMS: {len(all_detections)} detections")

    masks = [det['mask'] for det in all_detections]
    scores = [det['score'] for det in all_detections]
    boxes = [det['box'] for det in all_detections]

    return masks, scores, boxes


def apply_nms_and_filter(masks, scores, boxes, iom_threshold=0.5,
                         confidence_threshold=0.5, min_obj_area=0,
                         apply_nms=True, roi_conf_ratio=0.9):  # ← add param

    if apply_nms and len(masks) > 0:
        nms_threshold = STRICT_NMS_THRESHOLD if USE_MERGING else iom_threshold
        masks, scores, boxes = nms_iom_simple(masks, scores, boxes, nms_threshold)

    count_after_nms = len(masks)   # post-NMS, pre-confidence

    filtered_detections = []
    roi_candidates = []            # ← NEW: near-miss boxes for ROI

    reduced_threshold = confidence_threshold * roi_conf_ratio

    for i in range(len(masks)):
        mask  = masks[i]
        score = scores[i]
        box   = boxes[i]
        area  = int(mask.sum())

        if area < min_obj_area:
            continue

        if score >= reduced_threshold:     # ← qualifies for ROI
            roi_candidates.append({'mask': mask, 'box': box,
                                   'score': score, 'area': area})

        if score >= confidence_threshold:  # ← qualifies for final count
            filtered_detections.append({'mask': mask, 'box': box,
                                        'score': score, 'area': area})

    print(f"   Final count: {len(filtered_detections)}")
    print(f"   ROI candidates (≥{reduced_threshold:.3f}): {len(roi_candidates)}")
    return len(filtered_detections), filtered_detections, count_after_nms, roi_candidates

# ============================================================
# Main Counting Function with SMART Density Awareness (NEW)
# ============================================================

def count_objects_in_image(
    image_path: str,
    text_prompt: str,
    device_str: str = "cuda",
    tile_size: int = -1,  # -1 = adaptive
    tile_overlap: int = -1,  # -1 = adaptive
    confidence_threshold: float = 0.5,
    min_obj_area: int = 0,
    iom_threshold: float = 0.5,
    apply_nms: bool = True,
    bpe_path: str = None,
    visualizer=None,
) -> Tuple[int, List[Dict[str, Any]], np.ndarray]:
    """
    Count objects with SMART DENSITY-AWARE processing

    NEW WORKFLOW:
    1. Run SAM3 on full image (no tiling)
    2. Apply complete NMS & filtering on full-image results
    3. Calculate density on CLEAN detections
    4. If sparse → Return results (early exit)
    5. If dense → Re-run with adaptive tiling

    This ensures:
    - Tiling decision based on ACTUAL clean objects (not raw detections)
    - Books (6 large objects) won't trigger unnecessary tiling
    - Dense scenes (100 strawberries) will trigger tiling
    """

    # Build model
    model, device = build_sam3_image_predictor(device_str, bpe_path)

    # Initialize semantic analyzer
    print("\n" + "="*70)
    print("🧠 SMART DENSITY-AWARE OBJECT COUNTING")
    print("="*70)


    # Load image
    image = Image.open(image_path).convert("RGB")
    frame_np = np.array(image)
    width, height = image.size

    print(f"\n📐 Image: {width}x{height} pixels ({width*height:,} total)")

    # ========================================
    # STAGE 1: FULL IMAGE PREDICTION (NO TILING)
    # ========================================
    print("\n" + "="*70)
    print("📸 STAGE 1: Initial Full-Image Prediction")
    print("="*70)

    masks_initial, scores_initial, boxes_initial = process_full_image(
        model=model,
        device=device,
        image=image,
        text_prompt=text_prompt,
        confidence_threshold=0.0,
    )

    print(f"   Raw detections: {len(masks_initial)}")

    # ========================================
    # STAGE 2: COMPLETE PROCESSING (NMS + FILTERING)
    # ========================================
    print("\n" + "="*70)
    print("🎨 STAGE 2: Apply NMS & Filtering to Full-Image Results")
    print("="*70)

    num_objects_clean, detections_clean , count_after_nms , roi_candidates = apply_nms_and_filter(
        masks=masks_initial,
        scores=scores_initial,
        boxes=boxes_initial,
        iom_threshold=iom_threshold,
        confidence_threshold=confidence_threshold,
        min_obj_area=min_obj_area,
        apply_nms=apply_nms,
        roi_conf_ratio=0.9,
    )

    print(f"   Clean detections after NMS: {num_objects_clean}")

    # Extract clean boxes for density calculation
    boxes_clean = [det['box'] for det in detections_clean]

    # ========================================
    # STAGE 2.5: STAGE-1 QUALITY / RELIABILITY CHECK
    # ========================================

    # Use roi_candidates for ROI building (wider coverage than clean only)
    detections_for_roi = roi_candidates if len(roi_candidates) > len(detections_clean) else detections_clean
    reliability = assess_stage1_detection_reliability(
        masks_initial=masks_initial,
        scores_initial=scores_initial,
        boxes_initial=boxes_initial,
        detections_clean=detections_clean,
        image_width=width,
        image_height=height, count_after_nms=count_after_nms,
    )
    print("\n" + "="*70)
    print("🩺 STAGE 2.5: Stage-1 Reliability Check")
    print("="*70)
    print(f"   • Reliability score: {reliability['reliability_score']:.2f}/1.0")
    print(f"   • Raw/Clean counts: {reliability['raw_count']} / {reliability['clean_count']}")
    print(f"   • Mean clean score: {reliability['mean_clean_score']:.3f}")
    print(f"   • Clean/raw ratio: {reliability['clean_to_raw_ratio']:.3f}")
    print(f"   • ROI area ratio: {reliability['roi_area_ratio']:.2%}")
    print(f"   • ROI-tiling eligible: {'YES' if reliability['use_roi_tiling'] else 'NO'}")
    for rr in reliability['reasons']:
        print(f"     - {rr}")

    # ========================================
    # STAGE 3: DENSITY ANALYSIS ON CLEAN DETECTIONS
    # ========================================
    print("\n" + "="*70)
    print("🧮 STAGE 3: Smart Density Analysis on Clean Detections")
    print("="*70)

    density_score, use_tiling, metrics = calculate_density_smart(
        boxes_clean, width, height
    )
    tiling_rule = metrics["tiling_rule"]  
    print(f"\n📊 Density Metrics (based on clean detections):")
    print(f"   • Object count: {metrics['object_count']}")
    print(f"   • Coverage: {metrics['coverage']:.2%} of image")
    print(f"   • Avg size: {metrics['avg_object_size_ratio']:.2%} per object")
    print(f"   • Count score: {metrics['object_count_score']:.2f}/1.0")
    print(f"   • Size score: {metrics['size_score']:.2f}/1.0 (1.0=small)")
    print(f"   • Density: {density_score:.3f}")
    print(f"\n💡 Decision: {metrics['decision']}")

    # ========================================
    # DECISION POINT: USE FULL-IMAGE OR RE-RUN WITH TILING
    # ========================================
    print("\n" + "="*70)
    print("🎯 STAGE 3: Tiling Decision")
    print("="*70)
    # ADD right after: density_score, use_tiling, metrics = calculate_density_smart(...)
    
    if not use_tiling and count_after_nms > 10 and num_objects_clean == 0:
        print("⚠️  FORCE TILING: Many detections survived NMS but all failed confidence filter")
        print(f"    After NMS: {count_after_nms}  →  After confidence filter: 0")
        retention_rate = num_objects_clean / count_after_nms if count_after_nms > 0 else 0
        if 0.035 < retention_rate < 0.1:   # less than 3.5% survived confidence filter
            print(" low retention rate")
            use_tiling = True
            tiling_rule = "MEDIUM"
        elif retention_rate < 0.035:   # less than 5% survived confidence filter
            print(" Very low retention rate")
            use_tiling = True
            tiling_rule = "SMALL"
        metrics['decision'] = f"Forced tiling (0/{count_after_nms} survived confidence filter)"

    if not use_tiling:
        # EARLY EXIT - No tiling needed
        print(f"\n✅ SPARSE/MODERATE SCENE")
        print(f"   Using full-image results (tiling not needed)")
        print(f"   Reason: {metrics['decision']}")

        # Visualization for full-image results
        if visualizer and len(detections_clean) > 0:
            final_dets = []
            for det in detections_clean:
                box = det['box']
                final_dets.append({'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]]})
            visualizer.visualize_final_result(image, final_dets)

        print("\n" + "="*70)
        print(f"✅ FINAL COUNT: {num_objects_clean} objects")
        print(f"   Tiling used: NO")
        print(f"   Tiling rule: {tiling_rule}  ({metrics['decision']})")
        print("="*70 + "\n")

        return num_objects_clean, detections_clean, frame_np

    else:
        # DENSE SCENE - Re-run with tiling
        print(f"\n⚠️  DENSE SCENE DETECTED!")

        # ========================================
        # STAGE 4: RE-RUN WITH ADAPTIVE TILING
        # ========================================
        print("\n" + "="*70)
        print("🔄 STAGE 4: Re-running with Adaptive Tiling")
        print("="*70)

        if tile_size == -1 or tile_size is None:
            tile_size, tile_overlap = calculate_adaptive_tile_params(width, height, tiling_rule=tiling_rule)
        else:
            print(f"Manual tiling: {tile_size}px tiles, {tile_overlap}px overlap")

        # If tile_size >= image dimension, shrink for proper subdivision
        min_side = min(width, height)
        while tile_size > min_side // 2 and tile_size // 2 >= 96:
            tile_size = tile_size // 2
            tile_overlap = tile_size // 4
        print(f"  Final tile config after size-check: {tile_size}px tiles, {tile_overlap}px overlap")
            # Decide whether to tile full image attention regions or only stage-1 coverage ROI
        roi_box_for_tiling = reliability.get('roi_box') if reliability.get('use_roi_tiling', False) else None
        if roi_box_for_tiling is not None:
            print("\n🎯 Using ROI-only tiling on stage-1 cleaned coverage region (with padding)")
            print(f"   ROI box: {[round(v, 1) for v in roi_box_for_tiling]}")
        else:
            print("\n📐 Using FULL-IMAGE uniform tiling fallback (stage-1 considered weak/uncertain for ROI restriction)")

        masks_tiled, scores_tiled, boxes_tiled = process_image_with_density_tiling(
            model=model,
            device=device,
            image=image,
            text_prompt=text_prompt,
            tile_size=tile_size,
            overlap=tile_overlap,
            confidence_threshold=0.0,
            visualizer=visualizer,
            roi_box=roi_box_for_tiling,
            roi_tile_intersection_ratio=0.03,
        )

        print(f"\n   Tiled detections: {len(masks_tiled)}")

        # ========================================
        # STAGE 5: FINAL NMS ON TILED RESULTS
        # ========================================
        print("\n" + "="*70)
        print("🎨 STAGE 5: Apply NMS & Filtering to Tiled Results")
        print("="*70)
        # If tiling was forced because all detections failed confidence filter,
        # use 10% of original threshold to give low-confidence detections a chance
        if metrics['decision'].startswith("Forced tiling"):
            tiling_confidence = confidence_threshold * 0.85
            print(f"⚠️  Forced-tiling mode: reducing confidence threshold")
            print(f"    Original: {confidence_threshold:.3f}  →  Reduced: {tiling_confidence:.3f}")
        else:
            tiling_confidence = confidence_threshold

        num_objects_final, detections_final,_ , _ = apply_nms_and_filter(
            masks=masks_tiled,
            scores=scores_tiled,
            boxes=boxes_tiled,
            iom_threshold=iom_threshold,
            confidence_threshold=tiling_confidence,
            min_obj_area=min_obj_area,
            apply_nms=apply_nms,
        )

        # Visualization
        if visualizer and len(detections_final) > 0:
            final_dets = []
            for det in detections_final:
                box = det['box']
                final_dets.append({'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]]})
            visualizer.visualize_final_result(image, final_dets)

        print("\n" + "="*70)
        print(f"✅ FINAL COUNT: {num_objects_final} objects")
        print(f"   Tiling used: YES")

        print(f"   Improvement: {num_objects_final - num_objects_clean} more objects detected")
        print("="*70 + "\n")

        return num_objects_final, detections_final, frame_np


def visualize_detections_on_image(
    frame_np: np.ndarray,
    detections: List[Dict[str, Any]],
    count: int,
    save_path: str,
    show_id: bool = False,
    id_font_size: int = 18,
):
    """Overlay masks and boxes on image"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    img = frame_np.copy().astype(np.uint8)
    overlay = img.copy()

    rng = np.random.default_rng(0)
    colors = [rng.integers(0, 256, size=3, dtype=np.uint8) for _ in range(len(detections))]

    for i, det in enumerate(detections):
        mask = det["mask"]
        box = det.get("box", None)

        if mask is None:
            continue

        color = colors[i]
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)

        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, img.shape[1] - 1)
            y2 = min(y2, img.shape[0] - 1)

            overlay[y1:y1+2, x1:x2+1] = [255, 255, 255]
            overlay[y2-1:y2+1, x1:x2+1] = [255, 255, 255]
            overlay[y1:y2+1, x1:x1+2] = [255, 255, 255]
            overlay[y1:y2+1, x2-1:x2+1] = [255, 255, 255]

    vis = (0.5 * img + 0.5 * overlay).astype(np.uint8)
    pil_vis = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_vis)

    if show_id:
        try:
            id_font = ImageFont.truetype("arial.ttf", id_font_size)
        except Exception:
            try:
                id_font = ImageFont.truetype("DejaVuSans-Bold.ttf", id_font_size)
            except Exception:
                id_font = ImageFont.load_default()

        for i, det in enumerate(detections):
            mask = det["mask"]
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
            else:
                continue

            id_text = str(i + 1)
            bbox = draw.textbbox((0, 0), id_text, font=id_font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = cx - text_w // 2
            text_y = cy - text_h // 2
            padding = 8

            draw.rectangle(
                [text_x - padding, text_y - padding,
                 text_x + text_w + padding, text_y + text_h + padding],
                fill=(0, 0, 0, 200)
            )
            draw.text((text_x, text_y), id_text, fill=(255, 255, 255), font=id_font)

    text = f"Count: {count}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0, 160))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)

    pil_vis.save(save_path)
    print(f"🖼️ Saved visualization: {save_path}")


def update_counts_json(output_file: str, image_path: str, input_text: str, num_objects: int):
    """Update JSON file with counts"""
    if not output_file:
        return

    data = {}
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass

    if image_path not in data:
        data[image_path] = {}

    data[image_path][input_text] = int(num_objects)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"📁 Updated counts JSON: {output_file}")

# ============================================================
# CLI Interface
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser(
        "SAM3 Image Counting with ROI-ONLY / FULL-IMAGE UNIFORM DENSITY-AWARE TILING & SEMANTIC NMS",
        add_help=True,
    )

    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--input_text", type=str, required=True, help="Text prompt (e.g., 'bird', 'package of fruit')")
    parser.add_argument("--output_file", type=str, default="", help="JSON file to save counts")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--bpe_path", type=str, default=None, help="BPE vocab path")
    parser.add_argument("--tile_size", type=int, default=-1, help="Tile size in pixels. Use -1 for adaptive tiling (default: -1 = adaptive)")
    parser.add_argument("--tile_overlap", type=int, default=-1, help="Tile overlap in pixels. Use -1 for adaptive (default: -1 = adaptive)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Min confidence (default: 0.25)")
    parser.add_argument("--min_obj_area", type=int, default=0, help="Min pixel area (default: 0)")
    parser.add_argument("--iom_threshold", type=float, default=0.5, help="IoM threshold for NMS (default: 0.3)")

    parser.add_argument("--no_nms", action="store_true", help="Disable NMS")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization image")
    parser.add_argument("--vis_path", type=str, default="", help="Visualization output path")
    parser.add_argument("--show_id", action="store_true", help="Show ID numbers on visualization")
    parser.add_argument("--id_font_size", type=int, default=18, help="ID font size (default: 18)")

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    device_str = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available; falling back to CPU.")
        device_str = "cpu"

    print("\n" + "=" * 70)
    print("SAM3 with ROI-ONLY / FULL-IMAGE UNIFORM DENSITY-AWARE TILING & SEMANTIC NMS")
    print("=" * 70)
    print(f"📷 Image : {args.image_path}")
    print(f"📝 Prompt : '{args.input_text}'")
    print(f"💻 Device : {device_str}")
    print(f"🔲 Tile Size : {args.tile_size}")
    print(f"🔗 Tile Overlap: {args.tile_overlap}")
    print(f"⚙️ Confidence : {args.confidence_threshold}")
    print(f"🎯 Semantic NMS: {'Enabled (IoM=' + str(args.iom_threshold) + ')' if not args.no_nms else 'Disabled'}")
    print("=" * 70)

    visualizer = None
    if ENABLE_VISUALIZATION:
        if args.vis_path:
            vis_dir = os.path.join(args.vis_path if os.path.isdir(args.vis_path) else os.path.dirname(args.vis_path) or '.', 'visualization_semantic_aware')
        else:
            vis_dir = os.path.join(os.path.dirname(args.image_path) or '.', 'visualization_semantic_aware')
        visualizer = VisualizationHelper(output_dir=vis_dir)

    num_objects, detections, frame_np = count_objects_in_image(
        image_path=args.image_path,
        text_prompt=args.input_text,
        device_str=device_str,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        confidence_threshold=args.confidence_threshold,
        min_obj_area=args.min_obj_area,
        iom_threshold=args.iom_threshold,
        apply_nms=not args.no_nms,
        bpe_path=args.bpe_path,
        visualizer=visualizer,
    )

    print("\n" + "=" * 70)
    print(f"✅ Final Count: {num_objects}")
    print("=" * 70)

    if detections:
        print(f"\nDetection Details:")
        for i, det in enumerate(detections, 1):
            print(f"   #{i}: score={det['score']:.3f}, area={det['area']} pixels")

    if args.output_file:
        update_counts_json(args.output_file, args.image_path, args.input_text, num_objects)

    if args.save_vis:
        vis_path = args.vis_path if args.vis_path else os.path.splitext(args.image_path)[0] + "_hybrid_semantic_vis.png"
        visualize_detections_on_image(frame_np, detections, num_objects, vis_path, args.show_id, args.id_font_size)

    print("\n✅ Done!\n")



# ====================================================================
# Main Entry Point
# ====================================================================
if __name__ == "__main__":
    main()