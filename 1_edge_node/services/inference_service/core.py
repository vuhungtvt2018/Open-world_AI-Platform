import os
import time
import cv2
import sys
import numpy as np
from pathlib import Path
from packages.workflow.pipeline import Pipeline
from packages.utils.utils import (ts, union_box, pad_and_clip_box, 
                     arrow_angle, rotate_image, transform_points, need_clean, clean_data,
                     transform_bbox_to_original_coords)
from packages.utils.visualize import concat_anomaly_crops
from packages.workflow.events import default_event_bus, EventBus
from packages.core.config import AppConfig

# ROOT = 1_edge_node/  (2 levels up from services/inference_service/core.py)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))


class WebInference:
    def __init__(self, config):
        self.cfg = config
        self.pipeline = Pipeline(config)

    def run_inspect(self, cam_id: str = "default"):
        from services.camera_service.core import camera_manager
        frame = camera_manager.get_frame(cam_id)
        if frame is None:
            return {"status": "error", "message": "No input frame available"}

        if self.cfg.FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)

        H, W = frame.shape[:2]
        name = ts()

        # Lưu ảnh gốc
        orig_path = os.path.join(self.pipeline.dir_original, f"IMG_{name}.jpg")
        cv2.imwrite(orig_path, frame)

        start_time = time.time()

        # 1. Keypoint Detection
        keypoint_objects = []
        result_keypoints = self.pipeline.keypoint_detection.predict(frame, self.cfg.KEYPOINT_DETECTION)
        for result in result_keypoints:
            raw_boxes = result.boxes.xywh.cpu().numpy().astype(int)
            raw_keypoints = result.keypoints.data.cpu().numpy()
            filtered_keypoints, filtered_box = [], []
            for box, group in zip(raw_boxes, raw_keypoints):
                pt1, pt2 = group[0], group[1]
                if (len(pt1) < 3 or len(pt2) < 3
                        or pt1[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD
                        or pt2[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD):
                    continue
                filtered_keypoints.append([[int(pt1[0]), int(pt1[1]), pt1[-1]],
                                           [int(pt2[0]), int(pt2[1]), pt2[-1]]])
                x, y, w, h = box
                filtered_box.append([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)])
            keypoint_objects.append({
                "name": result.names.get('bolt', 'unknown'),
                "keypoints": filtered_keypoints,
                "boxes": filtered_box
            })

        if need_clean(keypoint_objects):
            keypoint_objects = clean_data(keypoint_objects)

        # 2. Segmentation
        r = self.pipeline.detector.predict(frame)
        segment_objects = []
        seg_boxes = []
        if r is not None and hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            seg_boxes = xyxy.tolist()
            masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
            if masks.shape[1] != H or masks.shape[2] != W:
                up = np.zeros((masks.shape[0], H, W), dtype=np.uint8)
                for i in range(masks.shape[0]):
                    up[i] = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
                masks = up
            for box, mask in zip(seg_boxes, masks):
                segment_objects.append({'box': box, 'mask': mask})
        else:
            masks = np.zeros((0, H, W), dtype=np.uint8)

        # 3. Mapping & Alignment
        mapping_object = self.pipeline.match_objects(keypoint_objects, segment_objects)
        if mapping_object:
            for m in mapping_object:
                m["union_box"] = temp = union_box(m["kp_box"], m["seg_box"])
                x1p, y1p, x2p, y2p = pad_and_clip_box(temp[0], temp[1], temp[2], temp[3],
                                                       self.cfg.PAD_RATIO, W, H, square=True)
                m["union_box_pad"] = [x1p, y1p, x2p, y2p]

        # 4. Rotation & Anomaly Detection
        ng_any = False
        per_objects = []
        if mapping_object:
            for i, obj in enumerate(mapping_object):
                landmark_1 = obj['keypoints'][1][:2]
                landmark_2 = obj['keypoints'][0][:2]
                x1, y1, x2, y2 = obj['union_box_pad']
                mask = obj['mask']

                angle = arrow_angle(landmark_1, landmark_2)
                rotated_frame, M = rotate_image(frame, 90 - angle)
                rotated_mask, _ = rotate_image(mask, 90 - angle)

                box_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                rotated_box = transform_points(box_pts, M).astype(int)
                x_min, y_min = rotated_box[:, 0].min(), rotated_box[:, 1].min()
                x_max, y_max = rotated_box[:, 0].max(), rotated_box[:, 1].max()

                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                max_val = max(abs(x_max - x_min), abs(y_max - y_min))

                crop_x1 = max(0, center_x - max_val // 2)
                crop_y1 = max(0, center_y - max_val // 2)
                crop_x2 = center_x + max_val // 2
                crop_y2 = center_y + max_val // 2
                crop = rotated_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop.size == 0 or crop_mask.size == 0:
                    continue

                crop_name = f"CROP_{name}_{i}.jpg"
                crop_path = os.path.join(self.pipeline.dir_crops, crop_name)
                cv2.imwrite(crop_path, crop)

                """
                09082026 - KHAI - Add codes to save mask
                """
                # Lưu mask
                mask_name = f"MASK_{name}_{i}.jpg"
                mask_path = os.path.join(self.pipeline.dir_masks, mask_name)
                cv2.imwrite(mask_path, crop_mask)

                """
                23082026 - KHAI - Make crops and corresponding masks more concentrated 
                """
                # Process crops
                y_indices, x_indices = np.where(crop_mask > 0)
                tight_y1, tight_y2 = int(y_indices.min()), int(y_indices.max() + 1)
                tight_x1, tight_x2 = int(x_indices.min()), int(x_indices.max() + 1)
                crop_pre = crop[tight_y1:tight_y2, tight_x1:tight_x2]
                crop_mask_pre = crop_mask[tight_y1:tight_y2, tight_x1:tight_x2]

                # Anomaly Detection
                out = self.pipeline.anomaly.run(crop_pre, crop_mask_pre)
                is_ng = bool(getattr(out, "is_ng", False))
                
                # Visualization tiles
                hm_tile = None
                if getattr(out, "heatmap_display") is not None:
                    hm_tile = out.heatmap_display.copy()
                    """
                    19082026 - KHAI - Hide objects not in the main focus in heatmap tile for visualization
                    """
                    hm_h, hm_w = hm_tile.shape[:2]        
                    mask_resized = cv2.resize((crop_mask_pre > 0).astype(np.uint8), (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)
                    mask_3ch = mask_resized[:, :, None]
                    hm_tile = hm_tile * mask_3ch
                    label_small = f"obj{i} {'NG' if is_ng else 'OK'}"
                    color = (0, 0, 255) if is_ng else (0, 200, 0)
                    cv2.putText(hm_tile, label_small, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                anomalies_full = []                
                if is_ng:
                    ng_any = True
                    # Lưu heatmap cho NG
                    cv2.imwrite(os.path.join(self.pipeline.dir_ng, f"NG_HEAT_{name}_{i}.jpg"), out.heatmap_display)

                    for ak, a in enumerate(getattr(out, "boxes", [])):
                        """
                        08082026 - KHAI - Refactor code to adhere to modified Pipeline
                        """
                        cx1, cy1, cx2, cy2 = int(a.xmin), int(a.ymin), int(a.xmax), int(a.ymax)
                        """
                        16082026 - KHAI - Convert bbox in crop to bbox in original
                        """
                        bbox_in_crop = [cx1 + tight_x1, cy1 + tight_y1, cx2 + tight_x1, cy2 + tight_y1]
                        crop_origin = (crop_x1, crop_y1)
                        bbox_in_original = transform_bbox_to_original_coords(
                            bbox_in_crop,
                            M,
                            crop_origin,
                            frame.shape,
                        )
                        anomalies_full.append({
                            "k": ak,
                            "bbox_full": bbox_in_original,
                            "bbox_in_object_crop": bbox_in_crop,
                            "bbox_in_original": bbox_in_original,
                        })
                        anom_crop = crop_pre[cy1:cy2, cx1:cx2]
                        if anom_crop.size != 0:
                            anom_name = f"ANOM_{name}_obj{i}_{ak}.jpg"
                            anom_path = os.path.join(self.pipeline.dir_anom_crops, anom_name)
                            cv2.imwrite(anom_path, anom_crop)
                            
                            # Gọi API phân loại của pipeline
                            cls_res = self.pipeline.classify_defect(anom_path)
                            if cls_res is not None:
                                label = cls_res.get("label", "NG")
                                sim = float(cls_res.get("similarity", 0.0))
                                anomalies_full[-1]["cls_label"] = label
                                anomalies_full[-1]["cls_similarity"] = sim
                            else:
                                anomalies_full[-1]["cls_label"] = "NG"
                                anomalies_full[-1]["cls_similarity"] = None

                # Draw on crop for visualization
                crop_labeled = crop.copy()
                """
                19082026 - KHAI - Add Vignette mask to crop to focus on main object
                """
                # Blur surrounding of main object
                h, w = crop.shape[:2]
                cx, cy = (tight_x1 + tight_x2) // 2, (tight_y1 + tight_y2) // 2
                X, Y = np.meshgrid(np.arange(w), np.arange(h))
                max_radius = np.sqrt(w**2 + h**2) / 2.0
                dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

                # Vignette mask
                vignette_mask = np.clip(1.0 - (dist_from_center / max_radius) * 0.90, 0.10, 1.0)
                vignette_mask_3ch = np.dstack([vignette_mask] * 3)

                # Dim background area
                vignetted_crop = (crop.astype(np.float32) * vignette_mask_3ch * 0.5).astype(np.uint8)

                # Segmentation mask blending
                binary_mask = (crop_mask > 0).astype(np.float32)
                feathered_mask = cv2.GaussianBlur(binary_mask, (15, 15), 0)[:, :, None]
                crop_labeled = (crop.astype(np.float32) * feathered_mask + 
                                vignetted_crop.astype(np.float32) * (1.0 - feathered_mask)).astype(np.uint8)
                
                if is_ng and len(anomalies_full) > 0:
                    for idx_disp, a in enumerate(anomalies_full, start=1):
                        """
                        19082026 - KHAI - Fix bbox thickness and label size for crop images
                        """
                        cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                        cv2.rectangle(crop_labeled, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)
                        
                        label_text = f"NG{idx_disp}"
                        sim_val = a.get("cls_similarity", None)
                        if self.cfg.DEFECT_CLS_ENABLE and a.get("cls_label") and a["cls_label"] != "NG":
                            if sim_val is not None:
                                if float(sim_val) >= float(self.cfg.DEFECT_CLS_SIM_THRESHOLD):
                                    label_text = f"{a['cls_label']} ({float(sim_val):.2f})"
                                else:
                                    label_text = f"{a['cls_label']}? ({float(sim_val):.2f})"
                            else:
                                label_text = a['cls_label']
                                
                        """
                        19082026 - KHAI - Increase font size for label in crop images
                        """
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 2)
                        tx1, ty1 = cx1, max(0, cy1 - th - 6)
                        tx2, ty2 = cx1 + tw + 8, cy1
                        cv2.rectangle(crop_labeled, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
                        cv2.putText(crop_labeled, label_text, (cx1 + 3, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                per_objects.append({
                    "product_id": 1,
                    "index": i,
                    "bbox": [0, 0, crop.shape[1], crop.shape[0]],
                    "score": round(float(getattr(out, "classification_score", 0.0)), 4),
                    "is_ng": is_ng,
                    "overlap_ratio": round(float(getattr(out, "overlap_ratio", 0.0)), 4) if crop.size != 0 else 0.0,
                    "crop": crop_name,
                    "crop_url": f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}/crops/{crop_name}",
                    "hm_tile": hm_tile,
                    "crop_labeled": crop_labeled,
                    "anomalies": anomalies_full
                })

        # 1. Heatmap panel
        hm_tiles = [obj["hm_tile"] for obj in per_objects if obj["hm_tile"] is not None]
        hm_panel = concat_anomaly_crops(hm_tiles, target_h=400) if hm_tiles else np.zeros((400, 400, 3), dtype=np.uint8)
        hm_name = f"HM_{name}.jpg"
        cv2.imwrite(os.path.join(self.pipeline.session_root, hm_name), hm_panel)
        
        # 2. Labeled crops panel
        labeled_crops = [obj["crop_labeled"] for obj in per_objects]
        crops_panel = concat_anomaly_crops(labeled_crops, target_h=400) if labeled_crops else np.zeros((400, 400, 3), dtype=np.uint8)
        crops_name = f"CROPS_VIS_{name}.jpg"
        cv2.imwrite(os.path.join(self.pipeline.session_root, crops_name), crops_panel)

        # 3. Overall output
        overall_vis = frame.copy()
        for i, obj in enumerate(per_objects):
            """
            10082026 - KHANH - Visualize OK and NG objects in overall output
            """
            object_index = obj["index"]

            if object_index >= len(mapping_object):
                continue

            """
            16082026 - KHAI - Convert bbox in crop to bbox in original
            """
            is_ng = obj["is_ng"]
            if is_ng and obj.get("anomalies"):
                for ak, anomaly in enumerate(obj["anomalies"], start=1):
                    x1, y1, x2, y2 = anomaly.get("bbox_in_original", [0, 0, 0, 0])
                    cv2.rectangle(
                        overall_vis,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3,
                    )

                    """
                    19082026 - KHAI - Increase font size for overall image
                    """
                    label = anomaly.get("cls_label") or f"NG{ak}"
                    if anomaly.get("cls_similarity") is not None:
                        label = f"{label} ({float(anomaly['cls_similarity']):.2f})"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        2,
                    )
                    label_top = max(0, y1 - text_height - baseline - 8)
                    cv2.rectangle(
                        overall_vis,
                        (x1, label_top),
                        (x1 + text_width + 10, y1),
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        overall_vis,
                        label,
                        (x1 + 5, y1 - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
        
        overall_name = f"OVERALL_{name}.jpg"
        cv2.imwrite(os.path.join(self.pipeline.session_root, overall_name), overall_vis)

        session_url = f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}"
        vis_urls = {
            "heatmap": f"{session_url}/{hm_name}",
            "crops": f"{session_url}/{crops_name}",
            "overall": f"{session_url}/{overall_name}"
        }

        clean_objects = []
        max_score = 0.0
        ng_count = 0
        for obj in per_objects:
            clean_obj = {k: v for k, v in obj.items() if k not in ["hm_tile", "crop_labeled"]}
            clean_objects.append(clean_obj)
            max_score = max(max_score, obj["score"])
            if obj["is_ng"]:
                ng_count += 1

        latency_ms = (time.time() - start_time) * 1000

        """
        23082026 - KHAI - Add task type and camera ID fields to record to feed to event bus
        """
        record = {
            "timestamp": name,
            "image": os.path.basename(orig_path),
            "task_type": "inspection",
            "camera_id": cam_id,
            "ng_detected": ng_any,
            "latency_ms": round(latency_ms, 2),
            "total_objects": len(per_objects),
            "ng_count": ng_count,
            "max_score": round(max_score, 4),
            "objects": clean_objects
        }

        # Publish event thay vì gọi DB trực tiếp (Event-driven Architecture)
        default_event_bus.publish(EventBus.EVENT_INFERENCE_DONE, record=record)

        """
        23082026 - KHAI - Add task type and camera ID fields to final output
        """
        return {
            "status": "success",
            "timestamp": name,
            "task_type": "inspection",
            "camera_id": cam_id,
            "ng_detected": ng_any,
            "metrics": {
                "latency_ms": round(latency_ms, 2),
                "total_objects": len(per_objects),
                "ng_count": ng_count,
                "max_score": round(max_score, 4)
            },
            "vis_urls": vis_urls,
            "original_image_url": f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}/original/IMG_{name}.jpg",
            "objects": clean_objects
        }


# Singleton instance – shared across all routers
inference_engine = WebInference(cfg)
