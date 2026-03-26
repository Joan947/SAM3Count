import os
import re
import json
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from sam3.model_builder import build_sam3_video_predictor



class StaticTrackerPE:
    """
    Wraps SAM3 tracking with appearance-based re-identification
    using Perception Encoder features
    """
    def __init__(self, similarity_threshold=0.75, max_lost_frames=30, device='cuda'):
        self.similarity_threshold = similarity_threshold
        self.max_lost_frames = max_lost_frames
        self.device = device
        
        # Appearance memory
        self.appearance_bank = {}
        self.id_mapping = {}
        self.next_consistent_id = 1
        self.active_sam3_ids = set()
        self.lost_tracks = {}
        self.current_frame_idx = 0
        
    def extract_masked_pe_features(self, pe_features, mask):
        """Extract appearance features from Perception Encoder output"""
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.to(pe_features.device)
        
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        
        if mask.shape != pe_features.shape[1:]:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=pe_features.shape[1:],
                mode='nearest'
            ).squeeze().bool()
        
        masked_features = pe_features * mask.unsqueeze(0)
        pooled = masked_features.sum(dim=[1, 2]) / (mask.sum() + 1e-6)
        pooled = F.normalize(pooled, p=2, dim=0)
        
        return pooled
    
    def update_appearance_bank(self, sam3_id, features, confidence, keep_top_k=5):
        """Store high-quality appearance features"""
        if sam3_id not in self.appearance_bank:
            self.appearance_bank[sam3_id] = {'features': [], 'confidence': [], 'last_box': None}
        
        self.appearance_bank[sam3_id]['features'].append(features.cpu())
        self.appearance_bank[sam3_id]['confidence'].append(confidence)
        
        if len(self.appearance_bank[sam3_id]['features']) > keep_top_k:
            sorted_idx = np.argsort(self.appearance_bank[sam3_id]['confidence'])[-keep_top_k:]
            self.appearance_bank[sam3_id]['features'] = [
                self.appearance_bank[sam3_id]['features'][i] for i in sorted_idx
            ]
            self.appearance_bank[sam3_id]['confidence'] = [
                self.appearance_bank[sam3_id]['confidence'][i] for i in sorted_idx
            ]
    
    def compute_similarity(self, features_a, features_b):
        """Cosine similarity between feature vectors"""
        if isinstance(features_a, list):
            features_a = torch.stack(features_a)
        if isinstance(features_b, list):
            features_b = torch.stack(features_b)
        
        features_a = features_a.to(self.device)
        features_b = features_b.to(self.device)
        
        if features_a.dim() == 1:
            features_a = features_a.unsqueeze(0)
        if features_b.dim() == 1:
            features_b = features_b.unsqueeze(0)
        
        similarity = torch.mm(features_a, features_b.t())
        return similarity.max().item()
    
    def match_new_detection_to_lost_track(self, new_features, new_box):
        """Try to match a new detection to a previously lost track"""
        if not self.lost_tracks:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        for consistent_id, lost_info in list(self.lost_tracks.items()):
            frames_lost = self.current_frame_idx - lost_info['last_frame']
            if frames_lost > self.max_lost_frames:
                del self.lost_tracks[consistent_id]
                continue
            
            similarity = self.compute_similarity(new_features, lost_info['features'])
            
            if lost_info['last_box'] is not None and new_box is not None:
                last_center = np.array([(lost_info['last_box'][0] + lost_info['last_box'][2]) / 2,
                                       (lost_info['last_box'][1] + lost_info['last_box'][3]) / 2])
                new_center = np.array([(new_box[0] + new_box[2]) / 2,
                                      (new_box[1] + new_box[3]) / 2])
                distance = np.linalg.norm(last_center - new_center)
                
                if distance > 200:
                    similarity *= 0.5
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = consistent_id
        
        if best_match_id is not None:
            del self.lost_tracks[best_match_id]
            print(f"   Re-identified! Matched to lost track ID {best_match_id} (similarity: {best_similarity:.3f})")
        
        return best_match_id, best_similarity
    
    def process_frame(self, masks, scores, object_ids, boxes, pe_features, frame_idx):
        """Process a single frame and maintain consistent IDs"""
        self.current_frame_idx = frame_idx
        
        if object_ids is None or len(object_ids) == 0:
            return object_ids
        
        current_sam3_ids = set(object_ids)
        
        # Detect lost tracks
        lost_sam3_ids = self.active_sam3_ids - current_sam3_ids
        for lost_sam3_id in lost_sam3_ids:
            if lost_sam3_id in self.id_mapping:
                consistent_id = self.id_mapping[lost_sam3_id]
                
                if lost_sam3_id in self.appearance_bank:
                    features_list = self.appearance_bank[lost_sam3_id]['features']
                    avg_features = torch.stack(features_list).mean(dim=0)
                    last_box = self.appearance_bank[lost_sam3_id].get('last_box', None)
                    
                    self.lost_tracks[consistent_id] = {
                        'last_frame': frame_idx - 1,
                        'features': avg_features,
                        'last_box': last_box,
                        'sam3_id': lost_sam3_id
                    }
                    print(f"    Track {consistent_id} (SAM3 ID {lost_sam3_id}) lost at frame {frame_idx}")
        
        self.active_sam3_ids = current_sam3_ids
        
        # Process each detection
        new_consistent_ids = []
        
        for i, sam3_id in enumerate(object_ids):
            score = scores[i]
            mask = masks[i]
            box = boxes[i] if boxes is not None else None
            
            # Extract PE features
            features = self.extract_masked_pe_features(pe_features, mask)
            
            if sam3_id in self.id_mapping:
                consistent_id = self.id_mapping[sam3_id]
                if score > 0.7:
                    self.update_appearance_bank(sam3_id, features, score)
                    self.appearance_bank[sam3_id]['last_box'] = box
                
                new_consistent_ids.append(consistent_id)
            else:
                matched_id, similarity = self.match_new_detection_to_lost_track(features, box)
                
                if matched_id is not None:
                    consistent_id = matched_id
                    self.id_mapping[sam3_id] = consistent_id
                    self.appearance_bank[sam3_id] = {
                        'features': [features.cpu()],
                        'confidence': [score],
                        'last_box': box
                    }
                else:
                    consistent_id = self.next_consistent_id
                    self.next_consistent_id += 1
                    self.id_mapping[sam3_id] = consistent_id
                    self.update_appearance_bank(sam3_id, features, score)
                    self.appearance_bank[sam3_id]['last_box'] = box
                    print(f"   New object detected: ID {consistent_id} (SAM3 ID {sam3_id})")
                
                new_consistent_ids.append(consistent_id)
        
        return new_consistent_ids

# ------------------------------------------------

class DynamicTracker:
    """Robust multi-modal tracker with CONSERVATIVE defaults"""

    def __init__(self, similarity_threshold=0.75, max_lost_frames=50, device='cuda' , mode='balanced'):
        self.similarity_threshold = similarity_threshold
        self.max_lost_frames = max_lost_frames
        self.device = device
        self.mode = mode
        print(f"*** TRACKING MODE: {self.mode} ***")

        # Core tracking data
        self.appearance_bank = {}
        self.id_mapping = {}
        self.next_consistent_id = 1
        self.active_sam3_ids = set()
        self.lost_tracks = {}
        self.current_frame_idx = 0

        # Motion tracking
        self.velocity_history = {}

        # Temporal window for new IDs
        self.pending_new_ids = {}

        # Recent masks
        self.recent_masks = {}

        # Scene metrics (for diagnostics)
        self.scene_metrics = {
            'avg_motion_per_frame': [],
            'detections_per_frame': [],
            'total_frames_processed': 0
        }

    def get_adaptive_params(self, frame_idx=None):
        """
        Manual mode parameters (no automatic scene detection).

        Modes:
        - 'sequential' : cars / passing objects
        - 'crowd'      : penguins / crowded similar objects
        - 'static'     : clothes / shelf products
        - 'balanced'   : general default
        """
        mode = getattr(self, "mode", "balanced")

        if mode == "sequential":
            return {
                "reid_window": 5,
                "reid_appearance_thresh": 0.92,
                "reid_motion_required": True,
                "reid_motion_thresh": 0.7,
                "reid_combined_thresh": 0.85,
                "motion_max_error": 40,
                "duplicate_appearance_thresh": 0.85,
                "duplicate_spatial_thresh": 80,
                "iou_merge_thresh": 0.85,
            }
        elif mode == "crowd":
            return {
                "reid_window": 40,
                "reid_appearance_thresh": 0.72,
                "reid_motion_required": False,
                "reid_motion_thresh": 0.4,
                "reid_combined_thresh": 0.75,
                "motion_max_error": 100,
                "duplicate_appearance_thresh": 0.70,
                "duplicate_spatial_thresh": 150,
                "iou_merge_thresh": 0.75,
            }
        elif mode == "static":
            return {
                "reid_window": 15,
                "reid_appearance_thresh": 0.85,
                "reid_motion_required": False,
                "reid_motion_thresh": 0.3,
                "reid_combined_thresh": 0.80,
                "motion_max_error": 60,
                "duplicate_appearance_thresh": 0.82,
                "duplicate_spatial_thresh": 70,
                "iou_merge_thresh": 0.88,
            }
        else:  # balanced
            return {
                "reid_window": 20,
                "reid_appearance_thresh": 0.82,
                "reid_motion_required": True,
                "reid_motion_thresh": 0.5,
                "reid_combined_thresh": 0.78,
                "motion_max_error": 80,
                "duplicate_appearance_thresh": 0.78,
                "duplicate_spatial_thresh": 100,
                "iou_merge_thresh": 0.82,
            }





    def extract_masked_pe_features(self, pe_features, mask):
        """Extract appearance features"""
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        mask = mask.to(pe_features.device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5

        if mask.shape != pe_features.shape[1:]:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=pe_features.shape[1:],
                mode='nearest'
            ).squeeze().bool()

        masked_features = pe_features * mask.unsqueeze(0)
        pooled = masked_features.sum(dim=[1, 2]) / (mask.sum() + 1e-6)
        pooled = F.normalize(pooled, p=2, dim=0)
        return pooled

    def update_appearance_bank(self, sam3_id, features, confidence, keep_top_k=10):
        """Store appearance features"""
        if sam3_id not in self.appearance_bank:
            self.appearance_bank[sam3_id] = {'features': [], 'confidence': [], 'last_box': None}

        should_add = True
        if len(self.appearance_bank[sam3_id]['features']) > 0:
            existing_features = torch.stack(self.appearance_bank[sam3_id]['features']).to(self.device)
            similarity = torch.mm(features.unsqueeze(0), existing_features.t()).max().item()
            if similarity > 0.95:
                should_add = False

        if should_add:
            self.appearance_bank[sam3_id]['features'].append(features.cpu())
            self.appearance_bank[sam3_id]['confidence'].append(confidence)

        if len(self.appearance_bank[sam3_id]['features']) > keep_top_k:
            sorted_idx = np.argsort(self.appearance_bank[sam3_id]['confidence'])[-keep_top_k:]
            self.appearance_bank[sam3_id]['features'] = [
                self.appearance_bank[sam3_id]['features'][i] for i in sorted_idx
            ]
            self.appearance_bank[sam3_id]['confidence'] = [
                self.appearance_bank[sam3_id]['confidence'][i] for i in sorted_idx
            ]

    def update_velocity(self, sam3_id, frame_idx, box):
        """Track motion"""
        if box is None:
            return

        if sam3_id not in self.velocity_history:
            self.velocity_history[sam3_id] = []

        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        self.velocity_history[sam3_id].append((frame_idx, center))

        if len(self.velocity_history[sam3_id]) > 10:
            self.velocity_history[sam3_id].pop(0)

    def predict_position(self, sam3_id, target_frame):
        """Predict position using motion model"""
        if sam3_id not in self.velocity_history or len(self.velocity_history[sam3_id]) < 2:
            return None

        history = self.velocity_history[sam3_id]
        frames = np.array([f for f, _ in history])
        positions = np.array([p for _, p in history])

        dt = frames[-1] - frames[0]
        if dt == 0:
            return positions[-1]

        velocity = (positions[-1] - positions[0]) / dt
        time_delta = target_frame - frames[-1]
        predicted_pos = positions[-1] + velocity * time_delta

        return predicted_pos

    def compute_similarity(self, features_a, features_b):
        """Cosine similarity"""
        if isinstance(features_a, list):
            features_a = torch.stack(features_a)
        if isinstance(features_b, list):
            features_b = torch.stack(features_b)

        features_a = features_a.to(self.device)
        features_b = features_b.to(self.device)

        if features_a.dim() == 1:
            features_a = features_a.unsqueeze(0)
        if features_b.dim() == 1:
            features_b = features_b.unsqueeze(0)

        similarity_matrix = torch.mm(features_a, features_b.t())
        return similarity_matrix.max().item()

    def check_temporal_sync_with_pending(self, history_a, existing_sam3_id):
        """Check temporal sync - STRICT (prevents false merges)"""
        if existing_sam3_id not in self.velocity_history:
            return False

        history_b = self.velocity_history[existing_sam3_id]

        if len(history_a) < 2 or len(history_b) < 2:
            return False

        frames_a = set(f for f, _ in history_a)
        frames_b = set(f for f, _ in history_b)
        overlap = len(frames_a & frames_b)
        total = len(frames_a | frames_b)
        overlap_ratio = overlap / total if total > 0 else 0

        if overlap_ratio < 0.6:
            return False

        common_frames = sorted(frames_a & frames_b)

        if len(common_frames) < 2:
            return False

        positions_a = []
        positions_b = []
        for f in common_frames[-3:]:
            pos_a = next((pos for frame, pos in history_a if frame == f), None)
            pos_b = next((pos for frame, pos in history_b if frame == f), None)
            if pos_a is not None and pos_b is not None:
                positions_a.append(pos_a)
                positions_b.append(pos_b)

        if len(positions_a) < 2:
            return False

        distances = [np.linalg.norm(pa - pb) for pa, pb in zip(positions_a, positions_b)]
        avg_distance = np.mean(distances)

        if avg_distance > 10:
            return False

        vel_a = positions_a[-1] - positions_a[0]
        vel_b = positions_b[-1] - positions_b[0]
        velocity_diff = np.linalg.norm(vel_a - vel_b)

        return velocity_diff < 10

    def check_mask_complementarity(self, mask_a, mask_b):
        """Check if masks are adjacent parts of same object"""
        set_a = set(zip(*np.nonzero(mask_a)))
        set_b = set(zip(*np.nonzero(mask_b)))

        if not set_a or not set_b:
            return False

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        iou = intersection / union if union > 0 else 0

        if iou > 0.3:
            return False

        coords_a = np.array(list(set_a))
        coords_b = np.array(list(set_b))

        if len(coords_a) > 1000:
            coords_a = coords_a[np.random.choice(len(coords_a), 1000, replace=False)]
        if len(coords_b) > 1000:
            coords_b = coords_b[np.random.choice(len(coords_b), 1000, replace=False)]

        from scipy.spatial.distance import cdist
        min_dist = cdist(coords_a, coords_b).min()

        return min_dist < 30

    def check_multimodal_duplicate(self, sam3_id, features, box, mask, frame_idx, params):
        """Multi-modal duplicate detection with given params"""

        if sam3_id in self.pending_new_ids:
            pending_data = self.pending_new_ids[sam3_id]
            sam3_id_history = [
                (pending_data['first_frame'] + i,
                 np.array([(b[0] + b[2])/2, (b[1] + b[3])/2]))
                for i, b in enumerate(pending_data['boxes']) if b is not None
            ]
        elif sam3_id in self.velocity_history:
            sam3_id_history = self.velocity_history[sam3_id]
        else:
            sam3_id_history = []

        recent_cutoff = frame_idx - 10

        candidates = {}

        for existing_sam3_id, existing_consistent_id in self.id_mapping.items():
            if existing_sam3_id == sam3_id:
                continue
            candidates[existing_sam3_id] = existing_consistent_id

        for consistent_id, lost_info in self.lost_tracks.items():
            lost_sam3_id = lost_info['sam3_id']
            if lost_sam3_id == sam3_id:
                continue
            if lost_info['last_frame'] >= recent_cutoff:
                candidates[lost_sam3_id] = consistent_id

        for existing_sam3_id, existing_consistent_id in candidates.items():
            if existing_sam3_id not in self.velocity_history:
                continue
            if len(self.velocity_history[existing_sam3_id]) == 0:
                continue

            last_frame = self.velocity_history[existing_sam3_id][-1][0]
            if last_frame < recent_cutoff:
                continue

            if existing_sam3_id in self.appearance_bank:
                existing_features = self.appearance_bank[existing_sam3_id]['features']
                existing_box = self.appearance_bank[existing_sam3_id].get('last_box')
            else:
                lost_info = next((info for info in self.lost_tracks.values()
                                if info['sam3_id'] == existing_sam3_id), None)
                if lost_info is None:
                    continue
                existing_features = lost_info['features']
                existing_box = lost_info.get('last_box')

            appearance_sim = self.compute_similarity(features, existing_features)
            if appearance_sim < params['duplicate_appearance_thresh']:
                continue

            if box and existing_box:
                center_new = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
                center_exist = np.array([(existing_box[0] + existing_box[2])/2,
                                        (existing_box[1] + existing_box[3])/2])
                distance = np.linalg.norm(center_new - center_exist)

                if distance > params['duplicate_spatial_thresh']:
                    continue
            else:
                continue

            if not self.check_temporal_sync_with_pending(sam3_id_history, existing_sam3_id):
                continue

            existing_mask = self.recent_masks.get(existing_sam3_id)
            if existing_mask is not None:
                if not self.check_mask_complementarity(mask, existing_mask):
                    continue

            print(f"   DUPLICATE: SAM3 {sam3_id} = {existing_sam3_id} → ID {existing_consistent_id} (sim={appearance_sim:.2f}, dist={distance:.1f}px)")
            return existing_consistent_id

        return None

    def match_new_detection_with_motion(self, new_features, new_box, frame_idx, params):
        """Re-identification with STRICT motion requirement"""
        if not self.lost_tracks:
            return None, 0.0

        active_consistent_ids = set(self.id_mapping.values())

        if new_box is None:
            return None, 0.0

        new_center = np.array([(new_box[0] + new_box[2]) / 2,
                               (new_box[1] + new_box[3]) / 2])

        best_match_id = None
        best_score = 0.0
        best_reason = ""

        for consistent_id, lost_info in list(self.lost_tracks.items()):
            if consistent_id in active_consistent_ids:
                continue

            frames_lost = frame_idx - lost_info['last_frame']
            sam3_id = lost_info['sam3_id']

            if frames_lost > params['reid_window']:
                del self.lost_tracks[consistent_id]
                continue

            # STRICT: MUST have motion prediction
            predicted_pos = self.predict_position(sam3_id, frame_idx)
            if predicted_pos is None:
                if params['reid_motion_required']:
                    continue  # Skip if no motion history
                else:
                    # Fallback to last position
                    if sam3_id in self.velocity_history and self.velocity_history[sam3_id]:
                        last_pos = self.velocity_history[sam3_id][-1][1]
                        prediction_error = np.linalg.norm(new_center - last_pos)
                        motion_score = 0.3
                    else:
                        continue
            else:
                prediction_error = np.linalg.norm(new_center - predicted_pos)
                max_error = params['motion_max_error'] + frames_lost * 15

                if prediction_error > max_error:
                    continue

                motion_score = 1.0 - (prediction_error / max_error)

            appearance_sim = self.compute_similarity(new_features, lost_info['features'])

            # Spatial exclusion check
            spatial_conflict = False
            for active_sam3_id in self.id_mapping.keys():
                if active_sam3_id in self.appearance_bank:
                    active_box = self.appearance_bank[active_sam3_id].get('last_box')
                    if active_box:
                        active_center = np.array([(active_box[0] + active_box[2])/2,
                                                 (active_box[1] + active_box[3])/2])
                        if np.linalg.norm(new_center - active_center) < 30:
                            spatial_conflict = True
                            break

            if spatial_conflict:
                continue

            combined_score = 0.6 * appearance_sim + 0.4 * motion_score

            if (appearance_sim > params['reid_appearance_thresh'] and 
                motion_score > params['reid_motion_thresh'] and 
                combined_score > best_score):
                best_score = combined_score
                best_match_id = consistent_id
                best_reason = f"app={appearance_sim:.2f}, mot={motion_score:.2f}, gap={frames_lost}f"

        if best_match_id is not None and best_score < params['reid_combined_thresh']:
            return None, 0.0

        if best_match_id is not None:
            print(f"  Re-ID: ID {best_match_id} ({best_reason})")
            del self.lost_tracks[best_match_id]

        return best_match_id, best_score

    def handle_pending_id(self, sam3_id, features, box, mask, frame_idx, params):
        """Temporal window with multi-modal matching"""
        WAIT_FRAMES = 3

        if sam3_id not in self.pending_new_ids:
            self.pending_new_ids[sam3_id] = {
                'first_frame': frame_idx,
                'features': [features.cpu()],
                'boxes': [box],
                'masks': [mask]
            }
            return None

        pending_data = self.pending_new_ids[sam3_id]
        pending_data['features'].append(features.cpu())
        pending_data['boxes'].append(box)
        pending_data['masks'].append(mask)

        frames_waiting = frame_idx - pending_data['first_frame']
        if frames_waiting < WAIT_FRAMES:
            return None

        avg_features = torch.stack(pending_data['features']).mean(0)
        latest_box = pending_data['boxes'][-1]
        latest_mask = pending_data['masks'][-1]

        duplicate_id = self.check_multimodal_duplicate(sam3_id, avg_features, latest_box, latest_mask, frame_idx, params)
        del self.pending_new_ids[sam3_id]

        if duplicate_id is not None:
            return duplicate_id

        matched_id, score = self.match_new_detection_with_motion(avg_features, latest_box, frame_idx, params)
        if matched_id is not None:
            return matched_id

        return None

    def process_frame(self, masks, scores, object_ids, boxes, pe_features, frame_idx):
        """Process frame with conservative tracking"""
        self.current_frame_idx = frame_idx
        self.scene_metrics['total_frames_processed'] += 1
        self.scene_metrics['detections_per_frame'].append(len(object_ids) if object_ids else 0)

        # Get parameters (works from frame 1)
        params = self.get_adaptive_params(frame_idx)

        if object_ids is None or len(object_ids) == 0:
            return []

        current_sam3_ids = set(object_ids)

        lost_sam3_ids = self.active_sam3_ids - current_sam3_ids
        for lost_sam3_id in lost_sam3_ids:
            if lost_sam3_id in self.id_mapping:
                consistent_id = self.id_mapping[lost_sam3_id]
                if lost_sam3_id in self.appearance_bank:
                    features_list = self.appearance_bank[lost_sam3_id]['features']
                    self.lost_tracks[consistent_id] = {
                        'last_frame': frame_idx - 1,
                        'features': features_list,
                        'last_box': self.appearance_bank[lost_sam3_id].get('last_box', None),
                        'sam3_id': lost_sam3_id,
                    }

        self.active_sam3_ids = current_sam3_ids

        current_frame_masks = {}
        for i, sam3_id in enumerate(object_ids):
            mask = masks[i]
            rows, cols = np.nonzero(mask)
            current_frame_masks[sam3_id] = set(zip(rows, cols))

        new_consistent_ids = []

        for i, sam3_id in enumerate(object_ids):
            score = scores[i]
            mask = masks[i]
            box = boxes[i] if boxes is not None else None
            features = self.extract_masked_pe_features(pe_features, mask)

            self.recent_masks[sam3_id] = mask

            if sam3_id in self.id_mapping:
                consistent_id = self.id_mapping[sam3_id]
                self.update_appearance_bank(sam3_id, features, score)
                self.appearance_bank[sam3_id]['last_box'] = box
                self.update_velocity(sam3_id, frame_idx, box)
                new_consistent_ids.append(consistent_id)

            else:
                duplicate_found = False
                new_mask_set = current_frame_masks[sam3_id]

                for other_sam3_id in object_ids:
                    if other_sam3_id == sam3_id or other_sam3_id not in self.id_mapping:
                        continue

                    other_mask_set = current_frame_masks[other_sam3_id]
                    intersection = len(new_mask_set & other_mask_set)
                    union = len(new_mask_set | other_mask_set)

                    if union > 0:
                        iou = intersection / union

                        if iou > params['iou_merge_thresh']:
                            new_box = box
                            other_box = boxes[object_ids.index(other_sam3_id)] if boxes else None

                            if new_box and other_box:
                                center_new = np.array([(new_box[0] + new_box[2])/2, (new_box[1] + new_box[3])/2])
                                center_other = np.array([(other_box[0] + other_box[2])/2,
                                                        (other_box[1] + other_box[3])/2])
                                distance = np.linalg.norm(center_new - center_other)

                                if distance > params['duplicate_spatial_thresh']:
                                    continue

                            existing_consistent_id = self.id_mapping[other_sam3_id]
                            consistent_id = existing_consistent_id
                            self.id_mapping[sam3_id] = consistent_id
                            self.appearance_bank[sam3_id] = {
                                "features": [features.cpu()],
                                "confidence": [score],
                                "last_box": box
                            }
                            self.update_velocity(sam3_id, frame_idx, box)
                            duplicate_found = True
                            print(f"  IoU merge: SAM3 {sam3_id} → ID {consistent_id} (IoU={iou:.2f})")
                            break

                if duplicate_found:
                    new_consistent_ids.append(consistent_id)
                    continue

                matched_id = self.handle_pending_id(sam3_id, features, box, mask, frame_idx, params)

                if matched_id is None:
                    if sam3_id in self.pending_new_ids:
                        continue
                    else:
                        consistent_id = self.next_consistent_id
                        self.next_consistent_id += 1
                        self.id_mapping[sam3_id] = consistent_id
                        self.update_appearance_bank(sam3_id, features, score)
                        self.appearance_bank[sam3_id]["last_box"] = box
                        self.update_velocity(sam3_id, frame_idx, box)
                        new_consistent_ids.append(consistent_id)

                else:
                    consistent_id = matched_id
                    self.id_mapping[sam3_id] = consistent_id
                    self.appearance_bank[sam3_id] = {
                        "features": [features.cpu()],
                        "confidence": [score],
                        "last_box": box
                    }
                    self.update_velocity(sam3_id, frame_idx, box)
                    new_consistent_ids.append(consistent_id)

        for sid in list(self.recent_masks.keys()):
            if sid not in current_sam3_ids:
                del self.recent_masks[sid]

        for consistent_id in list(self.lost_tracks.keys()):
            if frame_idx - self.lost_tracks[consistent_id]['last_frame'] > params['reid_window']:
                del self.lost_tracks[consistent_id]

        return new_consistent_ids


def merge_duplicate_tracks(T, iou_threshold=0.5):
    """Post-processing merge"""
    print("\n" + "="*70)
    print("Post-processing: Merging duplicate tracks...")
    print("="*70)

    merged_count = 0
    ids_to_remove = set()
    track_ids = sorted(T.keys())

    for i in range(len(track_ids)):
        for j in range(i+1, len(track_ids)):
            id1, id2 = track_ids[i], track_ids[j]

            if id1 in ids_to_remove or id2 in ids_to_remove:
                continue

            frames1 = set(T[id1].keys())
            frames2 = set(T[id2].keys())
            common_frames = frames1 & frames2

            if len(common_frames) < 3:
                if frames1 and frames2:
                    last_frame1 = max(frames1)
                    first_frame2 = min(frames2)
                    last_frame2 = max(frames2)
                    first_frame1 = min(frames1)

                    gap = min(abs(last_frame1 - first_frame2), abs(last_frame2 - first_frame1))

                    if gap <= 5:
                        nearest_frame1 = last_frame1 if last_frame1 < first_frame2 else first_frame1
                        nearest_frame2 = first_frame2 if last_frame1 < first_frame2 else last_frame2

                        if nearest_frame1 in T[id1] and nearest_frame2 in T[id2]:
                            rows1, cols1 = T[id1][nearest_frame1]
                            rows2, cols2 = T[id2][nearest_frame2]

                            if rows1.size == 0 or cols1.size == 0 or rows2.size == 0 or cols2.size == 0:
                                continue

                            center1 = np.array([cols1.mean(), rows1.mean()])
                            center2 = np.array([cols2.mean(), rows2.mean()])
                            distance = np.linalg.norm(center1 - center2)

                            if distance < 50:
                                print(f"🔗 Temporal merge: ID {id2} → ID {id1} (gap={gap}f, dist={distance:.1f}px)")
                                for frame_idx, mask_data in T[id2].items():
                                    if frame_idx not in T[id1]:
                                        T[id1][frame_idx] = mask_data
                                ids_to_remove.add(id2)
                                merged_count += 1
                continue

            if len(common_frames) < 5:
                continue

            ious = []
            for frame_idx in list(common_frames)[:20]:
                rows1, cols1 = T[id1][frame_idx]
                rows2, cols2 = T[id2][frame_idx]
                set1 = set(zip(rows1, cols1))
                set2 = set(zip(rows2, cols2))
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                if union > 0:
                    ious.append(intersection / union)

            if not ious:
                continue

            avg_iou = sum(ious) / len(ious)
            if avg_iou > iou_threshold:
                print(f" IoU merge: ID {id2} → ID {id1} (IoU={avg_iou:.3f}, {len(common_frames)}f)")
                for frame_idx, mask_data in T[id2].items():
                    if frame_idx not in T[id1]:
                        T[id1][frame_idx] = mask_data
                ids_to_remove.add(id2)
                merged_count += 1

    for id_to_remove in ids_to_remove:
        del T[id_to_remove]

    print(f" Merged {merged_count} duplicate track(s)")
    print("="*70 + "\n")

    return T, {}


def sort_frame_names(frame_names):
    def numeric_key(p):
        m = re.search(r"\d+", p)
        if m is None:
            return p
        return int(os.path.splitext(p)[0][m.start():])
    return sorted(frame_names, key=numeric_key)


def load_video_frames(video_dir, sample_frames=0, downsample_factor=1.0):
    frame_names = [p for p in os.listdir(video_dir)
                   if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]]

    if len(frame_names) == 0:
        raise ValueError(f"No frames found in {video_dir}")

    frame_names = sort_frame_names(frame_names)
    print(f"Original frames: {len(frame_names)}")

    if sample_frames > 0 and len(frame_names) > sample_frames:
        idx = np.round(np.linspace(0, len(frame_names) - 1, sample_frames)).astype(int)
        frame_names = [frame_names[i] for i in idx]
    elif sample_frames == 0 and downsample_factor > 1.0:
        new_len = int(np.ceil(len(frame_names) / downsample_factor))
        idx = np.round(np.linspace(0, len(frame_names) - 1, new_len)).astype(int)
        frame_names = [frame_names[i] for i in idx]

    print(f"Using {len(frame_names)} frames")

    video_frames = []
    for fn in frame_names:
        img = Image.open(os.path.join(video_dir, fn)).convert("RGB")
        video_frames.append(np.array(img))

    return frame_names, video_frames


def run_sam3_video_tracking_with_reid(
    checkpoint_path, video_frames, video_dir, text_prompt, device,
    reid_similarity_threshold, max_lost_frames, conf_thresh, min_obj_area, mode 
):
    import datetime
    print(f"\n RUN TIMESTAMP: {datetime.datetime.now()}")
    print(f" Video: {video_dir}")
    print(f" Prompt: '{text_prompt}'")
    print(f" CONSERVATIVE Multi-Modal Tracking (robust defaults)")

    video_predictor = build_sam3_video_predictor()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model' in state_dict:
            video_predictor.model.load_state_dict(state_dict['model'])
        else:
            video_predictor.model.load_state_dict(state_dict)

    video_predictor.model.to(device)
    video_predictor.model.eval()

    # reid_tracker = OcclusionRobustTracker(
    #     similarity_threshold=reid_similarity_threshold,
    #     max_lost_frames=max_lost_frames,
    #     device=device,
    #     mode=mode
    # )


    if mode == "static":
        reid_tracker = StaticTrackerPE(
            similarity_threshold=reid_similarity_threshold,
            max_lost_frames=max_lost_frames,
            device=device,
        )
    else:
        reid_tracker = DynamicTracker(   # your newer OcclusionRobustTracker
            similarity_threshold=reid_similarity_threshold,
            max_lost_frames=max_lost_frames,
            device=device,
            mode=mode,              # 'sequential', 'crowd', 'balanced'
        )


    response = video_predictor.start_session(resource_path=video_dir, session_id=None)
    session_id = response["session_id"]

    response = video_predictor.add_prompt(
        session_id=session_id, frame_idx=0, text=text_prompt,
        points=None, point_labels=None, bounding_boxes=None,
        bounding_box_labels=None, obj_id=None
    )

    vl_backbone = video_predictor.model.detector.backbone
    T = {}
    use_pe = False

    print("\n Tracking...")
    for response in video_predictor.propagate_in_video(
        session_id=session_id, propagation_direction="forward",
        start_frame_idx=None, max_frame_num_to_track=None
    ):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        if not outputs:
            continue

        obj_ids = outputs.get("out_obj_ids", [])
        probs = outputs.get("out_probs", [])
        masks = outputs.get("out_binary_masks", [])
        boxes_xywh = outputs.get("out_boxes_xywh", [])

        if len(obj_ids) == 0:
            continue

        filtered_masks, filtered_scores, filtered_obj_ids, filtered_boxes = [], [], [], []

        for i in range(len(obj_ids)):
            obj_id = int(obj_ids[i])
            score = float(probs[i]) if len(probs) > i else 1.0
            mask = masks[i]

            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()

            mask_binary = mask > 0.5 if mask.dtype in [np.float32, np.float64] else mask.astype(bool)
            area = int(mask_binary.sum())

            if area < min_obj_area or score < conf_thresh:
                continue

            filtered_masks.append(mask_binary)
            filtered_scores.append(score)
            filtered_obj_ids.append(obj_id)

            if len(boxes_xywh) > i:
                x, y, w, h = boxes_xywh[i]
                filtered_boxes.append([x, y, x + w, y + h])
            else:
                rows, cols = np.nonzero(mask_binary)
                if len(rows) > 0:
                    filtered_boxes.append([cols.min(), rows.min(), cols.max(), rows.max()])
                else:
                    filtered_boxes.append(None)

        if len(filtered_masks) == 0:
            continue


        # Map SAM3's frame_idx to downsampled video_frames index
        frame_list_idx = min(frame_idx, len(video_frames) - 1)
        frame_rgb = torch.from_numpy(video_frames[frame_list_idx]).to(device).float() / 255.0
    
        # frame_rgb = torch.from_numpy(video_frames[frame_idx]).to(device).float() / 255.0
        frame_rgb_chw = frame_rgb.permute(2, 0, 1).unsqueeze(0)
        frame_resized = torch.nn.functional.interpolate(
            frame_rgb_chw, size=(1008, 1008), mode='bilinear', align_corners=False
        )
        frame_normalized = (frame_resized - 0.5) / 0.5

        if use_pe:
            try:
                with torch.no_grad():
                    vision_backbone = vl_backbone.vision_backbone
                    vision_output = vision_backbone(frame_normalized)
                    sam3_out, sam3_pos, sam2_out, sam2_pos = vision_output
                    pe_features = sam3_out[0]
                    if pe_features.dim() == 4:
                        pe_features = pe_features.squeeze(0)
            except:
                pe_features = frame_rgb.permute(2, 0, 1)
                use_pe = False
        else:
            pe_features = frame_rgb.permute(2, 0, 1)

        consistent_ids = reid_tracker.process_frame(
            masks=filtered_masks, scores=filtered_scores,
            object_ids=filtered_obj_ids, boxes=filtered_boxes,
            pe_features=pe_features, frame_idx=frame_idx
        )

        for i, consistent_id in enumerate(consistent_ids):
            mask = filtered_masks[i]
            rows, cols = np.nonzero(mask)

            if consistent_id not in T:
                T[consistent_id] = {}
            T[consistent_id][frame_idx] = (rows, cols)

    print(f"\n Processed {len(video_frames)} frames")
    # print(f" SAM3 detections per frame: avg={np.mean(reid_tracker.scene_metrics['detections_per_frame']):.1f}")
    if hasattr(reid_tracker, "scene_metrics"):
        if reid_tracker.scene_metrics.get("detections_per_frame"):
            avg_det = np.mean(reid_tracker.scene_metrics["detections_per_frame"])
            print(f" SAM3 detections per frame: avg={avg_det:.1f}")
    print(f" Before merge: {len(T)} tracks")

    video_predictor.close_session(session_id)
    return T


def get_curr_count(frame_idx, T):
    return sum(1 for obj_id in T if frame_idx in T[obj_id])


def render_final_video(video_dir, frame_names, T, output_path, fps, font_size=8):
    if len(frame_names) == 0:
        return

    first = cv2.imread(os.path.join(video_dir, frame_names[0]))
    if first is None:
        print(f" Could not read first frame in {video_dir}")
        return

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rng = np.random.default_rng(0)
    colors = {obj_id: tuple(map(int, rng.integers(0, 256, size=3))) for obj_id in T.keys()}
    font_scale = max(font_size / 24.0, 0.4)

    for frame_idx, fname in enumerate(frame_names):
        img = cv2.imread(os.path.join(video_dir, fname))
        if img is None:
            print(f" Could not read first frame in {video_dir}")
        
            continue

        overlay = img.copy()
        id_positions = {}

        for obj_id, frames in T.items():
            if frame_idx in frames:
                rows, cols = frames[frame_idx]
                overlay[rows, cols] = colors[obj_id]
                if rows.size > 0:
                    id_positions[obj_id] = (int(cols.mean()), int(rows.mean()))

        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        cv2.putText(blended, f"Count: {get_curr_count(frame_idx, T)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        for obj_id, (cx, cy) in id_positions.items():
            text = f"ID {obj_id}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            x, y = max(cx - 10, 0), max(cy - 10, 0)
            cv2.rectangle(blended, (x, y - th - bl), (x + tw, y + bl), (0, 0, 0), -1)
            cv2.putText(blended, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(blended)

    writer.release()
    print(f" Saved: {output_path}")


def save_T_npz(T, path):
    arrays = {}
    for obj_id, frame_dict in T.items():
        for frame_idx, (rows, cols) in frame_dict.items():
            arrays[f"obj{obj_id}_frame{frame_idx}_rows"] = rows
            arrays[f"obj{obj_id}_frame{frame_idx}_cols"] = cols
    np.savez_compressed(path, **arrays)


def update_counts_json(output_file, orig_vid_dir, input_text, num_objects):
    if not output_file:
        return

    data = {}
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
            except:
                pass

    if orig_vid_dir not in data:
        data[orig_vid_dir] = {}
    data[orig_vid_dir][input_text] = int(num_objects)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)


def get_args_parser():
    parser = argparse.ArgumentParser("SAM3 Conservative Multi-Modal Tracking")
    parser.add_argument("--sam3_checkpoint", type=str, 
                       default="/cluster/medbow/project/advdls25/jowusu1/CountVid/countvid/lib/python3.10/site-packages/assets/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--font_size", type=int, default=8)
    parser.add_argument("--min_obj_area", type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--reid_similarity_threshold", type=float, default=0.05)
    parser.add_argument("--max_lost_frames", type=int, default=5)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument("--sample_frames", type=int, default=0)
    parser.add_argument("--downsample_factor", type=float, default=1.0)
    parser.add_argument(
    "--mode",
    type=str,
    default="balanced",
    choices=["sequential", "crowd", "static", "balanced"],
    help="Tracking mode: sequential(cars), crowd(penguins), static(clothes), balanced(general)",
    )
    parser.add_argument("--save_T", action="store_true")
    parser.add_argument("--save_final_video", action="store_true")
    parser.add_argument("--output_fps", type=float, default=30)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    device_str = "cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device

    frame_names, video_frames = load_video_frames(args.video_dir, args.sample_frames, args.downsample_factor)

    T = run_sam3_video_tracking_with_reid(
        checkpoint_path=args.sam3_checkpoint,
        video_frames=video_frames,
        video_dir=args.video_dir,
        text_prompt=args.input_text,
        device=device_str,
        reid_similarity_threshold=args.reid_similarity_threshold,
        max_lost_frames=args.max_lost_frames,
        conf_thresh=args.confidence_threshold,
        min_obj_area=args.min_obj_area,
        mode=args.mode,
    )

    T, _ = merge_duplicate_tracks(T, iou_threshold=0.5)

    print("\n" + "="*70)
    print("ID Timeline:")
    print("="*70)
    for consistent_id in sorted(T.keys()):
        frames = sorted(T[consistent_id].keys())
        if frames:
            gaps = []
            for i in range(len(frames)-1):
                gap = frames[i+1] - frames[i]
                if gap > 1:
                    gaps.append(f"gap {gap} at f{frames[i]}")
            gap_str = f" ({', '.join(gaps)})" if gaps else ""
            print(f"ID {consistent_id}: frames {frames[0]}-{frames[-1]} ({len(frames)} total){gap_str}")
    print("="*70 + "\n")

    num_objects = len(T)

    print("\n" + "="*50)
    print(f"Prompt: '{args.input_text}'")
    print(f"Video: {args.video_dir}")
    print(f"Final Count: {num_objects}")
    print("="*50)

    if args.save_T and args.output_dir:
        save_T_npz(T, os.path.join(args.output_dir, "tracks_T.npz"))

    # if args.save_final_video and args.output_dir:
    #     render_final_video(args.video_dir, frame_names, T,
    #                       os.path.join(args.output_dir, "final-video-reid.mp4"),
    #                       args.output_fps, args.font_size)
        
    if args.save_final_video and args.output_dir:
        out_name = os.path.basename(os.path.normpath(args.output_dir))
        video_out_path = os.path.join(args.output_dir, f"{out_name}.mp4")
        render_final_video(
            video_dir=args.video_dir,
            frame_names=frame_names,
            T=T,
            output_path=video_out_path,
            fps=args.output_fps,
            font_size=args.font_size,
    )    

    if args.output_file:
        update_counts_json(args.output_file, args.video_dir, args.input_text, num_objects)


if __name__ == "__main__":
    main()