import os
import time

import cv2
import numpy as np

from packages.camera import RTSP_Threaded_Camera
from packages.utils.utils import (
    ts,
    union_box,
    pad_and_clip_box,
    arrow_angle,
    rotate_image,
    transform_points,
    need_clean,
    clean_data,
)
from packages.utils.visualize import concat_anomaly_crops
from packages.workflow.pipeline import Pipeline

from .app_config import cfg


"""
07082026 - KHAI - Refactor code to adhere to modified Pipeline
"""
class WebInference:
    def __init__(self, config):
        self.cfg = config
        self.pipeline = Pipeline(config)
        
        # Khởi tạo Multi-Camera Stream
        self.cameras = {} # id -> Camera_Instance
        self.input_mode = {} # id -> "stream", "basler", "folder"
        self.current_frame = None
        self.current_image_name = "in_memory_image"
        
    def set_camera_mode(self, cam_id: str, mode: str):
        if self.input_mode.get(cam_id) == mode:
            return
            
        if cam_id in self.cameras and self.cameras[cam_id] is not None:
            self.cameras[cam_id].stop()
            self.cameras.pop(cam_id)
            
        self.input_mode[cam_id] = mode
        
        if mode == "stream":
            rtsp_val = self.cfg.RTSP_URL
            if str(rtsp_val).isdigit():
                rtsp_val = int(rtsp_val)
            self.cameras[cam_id] = RTSP_Threaded_Camera(rtsp_val, width=self.cfg.CAM_WIDTH, height=self.cfg.CAM_HEIGHT)
        elif mode == "basler":
            from packages.camera.basler import Basler_Threaded_Camera
            self.cameras[cam_id] = Basler_Threaded_Camera(width=self.cfg.CAM_WIDTH, height=self.cfg.CAM_HEIGHT)

    def get_frame(self, cam_id: str = "default"):
        if self.input_mode.get(cam_id) in ["stream", "basler"] and cam_id in self.cameras:
            frame = self.cameras[cam_id].read()
            return frame
        else:
            # Lấy ảnh trực tiếp từ RAM (không đọc ổ cứng)
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def run_inspect(self, cam_id: str = "default"):
        frame = self.get_frame(cam_id)
        if frame is None:
            return {"status": "error", "message": "No input frame available"}
            
        if self.cfg.FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)
            
        H, W = frame.shape[:2]
        name = ts()
        insp_time_str = self.pipeline.format_inspection_time(name)
        
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
                if (len(pt1) < 3 or len(pt2) < 3 or pt1[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD or pt2[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD):
                    continue
                filtered_keypoints.append([[int(pt1[0]), int(pt1[1]), pt1[-1]], [int(pt2[0]), int(pt2[1]), pt2[-1]]])
                x, y, w, h = box
                filtered_box.append([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)])
            keypoint_objects.append({"name": result.names.get('bolt', 'unknown'), "keypoints": filtered_keypoints, "boxes": filtered_box})
        
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
                x1p, y1p, x2p, y2p = pad_and_clip_box(temp[0], temp[1], temp[2], temp[3], self.cfg.PAD_RATIO, W, H, square=True)
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
                
                # Tính góc và xoay
                angle = arrow_angle(landmark_1, landmark_2)
                rotated_frame, M = rotate_image(frame, 90-angle)
                rotated_mask, _ = rotate_image(mask, 90-angle)

                # Xoay box và crop
                box_pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                rotated_box = transform_points(box_pts, M).astype(int)
                x_min, y_min = rotated_box[:,0].min(), rotated_box[:,1].min()
                x_max, y_max = rotated_box[:,0].max(), rotated_box[:,1].max()
                
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                max_val = max(abs(x_max - x_min), abs(y_max - y_min))
                
                cx1, cy1, cx2, cy2 = center_x - max_val // 2, center_y - max_val // 2, center_x + max_val // 2, center_y + max_val // 2
                crop = rotated_frame[max(0,cy1):cy2, max(0,cx1):cx2]
                crop_mask = rotated_mask[max(0,cy1):cy2, max(0,cx1):cx2]

                if crop.size == 0 or crop_mask.size == 0:
                    continue

                # Lưu crop
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

                # Anomaly Detection
                out = self.pipeline.anomaly.run(crop, crop_mask)
                is_ng = bool(getattr(out, "is_ng", False))
                
                # Visualization tiles
                hm_tile = None
                if getattr(out, "heatmap_display") is not None:
                    hm_tile = out.heatmap_display.copy()
                    label_small = f"obj{i} {'NG' if is_ng else 'OK'}"
                    color = (0, 0, 255) if is_ng else (0, 200, 0)
                    cv2.putText(hm_tile, label_small, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                anomalies_full = []
                cls_labels_i = []
                
                if is_ng:
                    ng_any = True
                    # Lưu heatmap cho NG
                    cv2.imwrite(os.path.join(self.pipeline.dir_ng, f"NG_HEAT_{name}_{i}.jpg"), out.heatmap_display)

                    for ak, a in enumerate(getattr(out, "boxes", [])):
                        """
                        08082026 - KHAI - Refactor code to adhere to modified Pipeline
                        """
                        cx1, cy1, cx2, cy2 = int(a.xmin), int(a.ymin), int(a.xmax), int(a.ymax)
                        anomalies_full.append({
                            "k": ak,
                            "bbox_full": [int(cx1), int(cy1), int(cx2), int(cy2)],
                            "bbox_in_object_crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
                        })
                        anom_crop = crop[cy1:cy2, cx1:cx2]
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
                                cls_labels_i.append(label)
                            else:
                                anomalies_full[-1]["cls_label"] = "NG"
                                anomalies_full[-1]["cls_similarity"] = None


                # Draw on crop for visualization
                crop_labeled = crop.copy()
                if is_ng and len(anomalies_full) > 0:
                    for idx_disp, a in enumerate(anomalies_full, start=1):
                        cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                        cv2.rectangle(crop_labeled, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                        
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
                        tx1, ty1 = cx1, max(0, cy1 - th - 6)
                        tx2, ty2 = cx1 + tw + 8, cy1
                        cv2.rectangle(crop_labeled, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
                        cv2.putText(crop_labeled, label_text, (cx1 + 3, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                per_objects.append({
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
        hm_path = os.path.join(self.pipeline.session_root, hm_name)
        cv2.imwrite(hm_path, hm_panel)
        
        # 2. Labeled crops panel
        labeled_crops = [obj["crop_labeled"] for obj in per_objects]
        crops_panel = concat_anomaly_crops(labeled_crops, target_h=400) if labeled_crops else np.zeros((400, 400, 3), dtype=np.uint8)
        crops_name = f"CROPS_VIS_{name}.jpg"
        crops_vis_path = os.path.join(self.pipeline.session_root, crops_name)
        cv2.imwrite(crops_vis_path, crops_panel)

        # 3. Overall output
        overall_vis = frame.copy()
        for i, obj in enumerate(per_objects):
            """
            10082026 - KHANH - Visualize OK and NG objects in overall output
            """
            object_index = obj["index"]

            if object_index >= len(mapping_object):
                continue

            x1, y1, x2, y2 = map(
                int,
                mapping_object[object_index]["seg_box"],
            )

            is_ng = obj["is_ng"]
            color = (0, 0, 255) if is_ng else (0, 255, 0)
            status = "NG" if is_ng else "OK"
            label = f"Object {object_index}: {status}"

            cv2.rectangle(
                overall_vis,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )

            label_top = max(0, y1 - text_height - baseline - 8)

            cv2.rectangle(
                overall_vis,
                (x1, label_top),
                (x1 + text_width + 10, y1),
                color,
                -1,
            )

            cv2.putText(
                overall_vis,
                label,
                (x1 + 5, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        
        overall_name = f"OVERALL_{name}.jpg"
        overall_path = os.path.join(self.pipeline.session_root, overall_name)
        cv2.imwrite(overall_path, overall_vis)
        
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

        # 5. Summary Record
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
        from packages.workflow.events import default_event_bus, EventBus
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

inference_engine = WebInference(cfg)
