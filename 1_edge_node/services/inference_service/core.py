import os
import time
import cv2
import sys
import numpy as np
from pathlib import Path
from packages.workflow.pipeline import Pipeline
from packages.utils.utils import (ts, union_box, pad_and_clip_box,
                     arrow_angle, rotate_image, transform_points, need_clean, clean_data)
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

                cx1 = center_x - max_val // 2
                cy1 = center_y - max_val // 2
                cx2 = center_x + max_val // 2
                cy2 = center_y + max_val // 2
                crop = rotated_frame[max(0, cy1):cy2, max(0, cx1):cx2]
                crop_mask = rotated_mask[max(0, cy1):cy2, max(0, cx1):cx2]

                if crop.size == 0 or crop_mask.size == 0:
                    continue

                crop_name = f"CROP_{name}_{i}.jpg"
                crop_path = os.path.join(self.pipeline.dir_crops, crop_name)
                cv2.imwrite(crop_path, crop)

                out = self.pipeline.anomaly.run_on_crop(crop, crop_mask)
                is_ng = bool(out.get("is_ng", False))

                hm_tile = None
                if out.get("hm_disp") is not None:
                    hm_tile = out["hm_disp"].copy()
                    label_small = f"obj{i} {'NG' if is_ng else 'OK'}"
                    color = (0, 0, 255) if is_ng else (0, 200, 0)
                    cv2.putText(hm_tile, label_small, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                anomalies_full = []
                cls_labels_i = []

                if is_ng:
                    ng_any = True
                    cv2.imwrite(os.path.join(self.pipeline.dir_ng, f"NG_HEAT_{name}_{i}.jpg"), out["hm_disp"])
                    for ak, a in enumerate(out.get("anomalies", [])):
                        bx1, by1, bx2, by2 = a["bbox_in_object_crop"]
                        anomalies_full.append({
                            "k": ak,
                            "bbox_full": [int(bx1), int(by1), int(bx2), int(by2)],
                            "bbox_in_object_crop": [int(bx1), int(by1), int(bx2), int(by2)],
                        })
                        anom_crop = crop[by1:by2, bx1:bx2]
                        if anom_crop.size != 0:
                            anom_name = f"ANOM_{name}_obj{i}_{ak}.jpg"
                            anom_path = os.path.join(self.pipeline.dir_anom_crops, anom_name)
                            cv2.imwrite(anom_path, anom_crop)
                            cls_res = self.pipeline._classify_defect(anom_path)
                            if cls_res is not None:
                                label = cls_res.get("label", "NG")
                                sim = float(cls_res.get("similarity", 0.0))
                                anomalies_full[-1]["cls_label"] = label
                                anomalies_full[-1]["cls_similarity"] = sim
                                cls_labels_i.append(label)
                            else:
                                anomalies_full[-1]["cls_label"] = "NG"
                                anomalies_full[-1]["cls_similarity"] = None

                # Draw on crop for visualization
                crop_labeled = crop.copy()
                if is_ng and len(anomalies_full) > 0:
                    for idx_disp, a in enumerate(anomalies_full, start=1):
                        bx1, by1, bx2, by2 = a["bbox_in_object_crop"]
                        cv2.rectangle(crop_labeled, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
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
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        tx1, ty1 = bx1, max(0, by1 - th - 6)
                        tx2, ty2 = bx1 + tw + 8, by1
                        cv2.rectangle(crop_labeled, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
                        cv2.putText(crop_labeled, label_text, (bx1 + 3, by1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                per_objects.append({
                    "index": i,
                    "bbox": [0, 0, crop.shape[1], crop.shape[0]],
                    "score": round(float(out.get("score", 0.0)), 4),
                    "is_ng": is_ng,
                    "overlap_ratio": round(float(out.get("overlap_ratio", 0.0)), 4) if crop.size != 0 else 0.0,
                    "crop": crop_name,
                    "crop_url": f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}/crops/{crop_name}",
                    "hm_tile": hm_tile,
                    "crop_labeled": crop_labeled,
                    "anomalies": anomalies_full
                })

        # Visualization panels
        hm_tiles = [obj["hm_tile"] for obj in per_objects if obj["hm_tile"] is not None]
        hm_panel = concat_anomaly_crops(hm_tiles, target_h=400) if hm_tiles else np.zeros((400, 400, 3), dtype=np.uint8)
        hm_name = f"HM_{name}.jpg"
        cv2.imwrite(os.path.join(self.pipeline.session_root, hm_name), hm_panel)

        labeled_crops = [obj["crop_labeled"] for obj in per_objects]
        crops_panel = concat_anomaly_crops(labeled_crops, target_h=400) if labeled_crops else np.zeros((400, 400, 3), dtype=np.uint8)
        crops_name = f"CROPS_VIS_{name}.jpg"
        cv2.imwrite(os.path.join(self.pipeline.session_root, crops_name), crops_panel)

        overall_vis = frame.copy()
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

        record = {
            "timestamp": name,
            "image": os.path.basename(orig_path),
            "ng_detected": ng_any,
            "latency_ms": round(latency_ms, 2),
            "total_objects": len(per_objects),
            "ng_count": ng_count,
            "max_score": round(max_score, 4),
            "objects": clean_objects
        }

        # Publish event thay vì gọi DB trực tiếp (Event-driven Architecture)
        default_event_bus.publish(EventBus.EVENT_INFERENCE_DONE, record=record)

        return {
            "status": "success",
            "timestamp": name,
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
