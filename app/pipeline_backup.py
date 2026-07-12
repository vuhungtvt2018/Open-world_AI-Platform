# app/pipeline.py
import os
import json
import cv2
import numpy as np
import requests
from typing import List, Dict, Optional
from datetime import datetime

from .config import AppConfig
from .camera import RTSP_Threaded_Camera
from .session import make_session_dir
from .utils import (ts, ensure_dirs, pad_and_clip_box, need_clean, clean_data, 
                    arrow_angle, rotate_image, transform_points, union_box, compute_iou)
from .visualize import draw_preview, concat_anomaly_crops
from .detectors.yolo_detector import YOLODetector
from .detectors.anomaly import AnomalyInferencer, AnomalyConfig


class Pipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        prod_dir = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME)
        ensure_dirs(prod_dir)

        (self.session_root, self.dir_original, self.dir_crops, self.dir_masks,
         self.dir_bboxes, self.dir_ng, self.dir_anom_bboxes, self.dir_anom_crops, self.dir_anom_crop_vis, 
         self.keypoint_crops, self.masks_keypoint_crops) = make_session_dir(prod_dir)

        self.results_json_path = os.path.join(self.session_root, "inspection_results.json")
        if not os.path.exists(self.results_json_path):
            with open(self.results_json_path, "w", encoding="utf-8") as f:
                json.dump({"results": []}, f, ensure_ascii=False, indent=2)

        self.detector = YOLODetector(cfg.MODEL_PATH)
        self.keypoint_detection = YOLODetector(cfg.KEYPOINTS_MODEL_PATH)
        
        anom_cfg = AnomalyConfig(
            input_size=cfg.ANOMALY_INPUT_SIZE,
            score_thres=cfg.ANOMALY_SCORE_THRESHOLD,
            inside_overlap_min=cfg.ANOMALY_INSIDE_OVERLAP_MIN,
            min_area_ratio=cfg.ANOMALY_MIN_AREA_RATIO,
            bbox_pad_ratio=cfg.ANOMALY_BBOX_PAD_RATIO,
            save_all=cfg.SAVE_ALL_ANOMALIES,
            show_all_boxes=cfg.SHOW_ALL_ANOMALY_BOXES,
            amap_threshold=cfg.ANOMALY_AMAP_THRESHOLD,
            redo_center_crop=cfg.REDO_CENTER_CROP,
            center_crop=cfg.CENTER_CROP,
        )
        self.anomaly = AnomalyInferencer(cfg.ANOMALY_MODEL_PATH, cfg.ANOMALY_DEVICE, anom_cfg)

        self.searcher = None  # không dùng VecDB

    # ================== Helpers ==================
    def _format_inspection_time(self, ts_name: str) -> str:
        try:
            dt = datetime.strptime(ts_name, "%Y%m%d_%H%M%S_%f")
        except Exception:
            dt = datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _post_qc(self, file_path: str, description: str, inspection_time_str: str, actual_result: str):
        if not self.cfg.API_SYNC_ENABLE:
            return
        url = self.cfg.API_SYNC_URL
        data = {
            "ProductionInstruction": self.cfg.PRODUCTION_INSTRUCTION or "",
            "ItemCode": self.cfg.ITEM_CODE or "",
            "InspectionTime": inspection_time_str,
            "Description": description or "",
            "OtherInfo": self.cfg.OTHER_INFO_DEFAULT or "",
            "ActualResult": actual_result,
        }
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
                r = requests.post(url, data=data, files=files, timeout=float(self.cfg.API_TIMEOUT))
            if 200 <= r.status_code < 300:
                print(f"[SYNC] OK -> {os.path.basename(file_path)} | result={actual_result} | desc='{description}'")
            else:
                print(f"[SYNC] FAIL {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[SYNC] ERROR: {e}")

    def _unique_preserve(self, seq: List[str]) -> List[str]:
        seen, out = set(), []
        for s in seq:
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # === NEW: classify defect via external API ===
    def _classify_defect(self, img_path: str) -> Optional[Dict]:
        """Gọi API phân loại lỗi. Trả về {'label': str, 'similarity': float, 'raw': dict} hoặc None nếu lỗi/không hợp lệ."""
        if not self.cfg.DEFECT_CLS_ENABLE:
            return None
        try:
            with open(img_path, "rb") as f:
                files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
                data = {
                    "top_k": str(self.cfg.DEFECT_CLS_TOPK),
                    "metric": self.cfg.DEFECT_CLS_METRIC,
                    "ItemCode": self.cfg.ITEM_CODE or "",
                }
                r = requests.post(self.cfg.DEFECT_CLS_URL, data=data, files=files, timeout=float(self.cfg.API_TIMEOUT))
            if not (200 <= r.status_code < 300):
                print(f"[CLS] FAIL {r.status_code}: {r.text[:200]}")
                return None
            arr = r.json()
            if not isinstance(arr, list) or len(arr) == 0:
                return None
            rec = arr[0]
            dist = float(rec.get("distance", 1.0))
            metric = str(self.cfg.DEFECT_CLS_METRIC).lower()
            # Map distance -> similarity
            if "cosine" in metric:
                similarity = 1.0 - dist
            else:
                # fallback an toàn
                similarity = 1.0 - dist
            label = rec.get("ErrorDetail") or "NG"
            return {"label": str(label), "similarity": float(similarity), "raw": rec}
        except Exception as e:
            print(f"[CLS] ERROR: {e}")
            return None

    def match_objects(self, keypoint_objects, segment_objects, iou_thresh=0.5):
        kp_boxes = keypoint_objects[0]["boxes"]
        kp_keypoints = keypoint_objects[0]["keypoints"]

        matched = []
        used_segments = set()

        for idx, kp_box in enumerate(kp_boxes):
            best_match = None
            best_iou = 0
            for j, seg_obj in enumerate(segment_objects):
                if j in used_segments:
                    continue
                iou = compute_iou(kp_box, seg_obj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = (j, seg_obj)
            if best_match and best_iou >= iou_thresh:
                j, seg_obj = best_match
                used_segments.add(j)
                matched.append({
                    "keypoints": kp_keypoints[idx],
                    "kp_box": kp_box,
                    "seg_box": seg_obj["box"],
                    "mask": seg_obj["mask"],
                    "iou": best_iou
                })
        
        
        
        return matched

    # ================== MAIN LOOP ==================
    def run(self):
        print(f"[INFO] Session dir: {self.session_root}")
        print("[INFO] SPACE=Capture & Inspect | ENTER=Confirm/Reset | q=Quit")

        cam = RTSP_Threaded_Camera(self.cfg.RTSP_URL, width=self.cfg.CAM_WIDTH, height=self.cfg.CAM_HEIGHT)
        saved_count = 0
        await_confirm = False
        captured_frame = None

        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
        cv2.namedWindow("anomaly_crops_view", cv2.WINDOW_NORMAL)
        cv2.namedWindow("keypoint_crop", cv2.WINDOW_NORMAL)
        cv2.namedWindow('preview_overlay', cv2.WINDOW_NORMAL)

        try:
            while True:
                live = cam.read()

                if live is None:
                    cv2.waitKey(1)
                    continue

                if self.cfg.FLIP_VERTICAL:
                    live = cv2.flip(live, 0)
                H, W = live.shape[:2]

                frame_show = captured_frame if await_confirm and captured_frame is not None else live
                disp = draw_preview(
                    frame_show,
                    self.cfg.DISPLAY_SCALE,
                    "WAIT ENTER" if await_confirm else "READY (SPACE)",
                    saved_count
                )
                cv2.imshow("preview", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                if await_confirm and (key in (13, 10)):  # ENTER
                    await_confirm = False
                    captured_frame = None
                    empty_h = max(1, int(H * self.cfg.DISPLAY_SCALE))
                    empty_w = max(1, int(W * self.cfg.DISPLAY_SCALE))
                    empty = np.zeros((empty_h, empty_w, 3), dtype=np.uint8)
                    cv2.imshow("heatmap", empty)
                    cv2.imshow("anomaly_crops_view", empty)
                    cv2.imshow("keypoint_crop", empty)
                    cv2.imshow('preview_overlay', empty)
                    continue

                if await_confirm:
                    continue

                if key == 32:  # SPACE
                    frame = live.copy()
                    frame_overlay = frame.copy()
                    captured_frame = frame.copy()
                    H, W = frame.shape[:2]
                    
                    mapping_object = []
                    anomalies_crops = []

                    name = ts()
                    insp_time_str = self._format_inspection_time(name)
                    orig_path = os.path.join(self.dir_original, f"IMG_{name}.jpg")
                    cv2.imwrite(orig_path, frame)
                    
                    # ====== Keypoint detection ======
                    keypoint_objects = []
                    result_kepoints = self.keypoint_detection.predict(frame, self.cfg.KEYPOINT_DETECTION)
                    
                    for result in result_kepoints:
                        raw_boxes = result.boxes.xywh.cpu().numpy().astype(int)
                        raw_keypoints = result.keypoints.data.cpu().numpy()
                                
                        filtered_keypoints, filtered_box = [], []
                                
                        for box, group in zip (raw_boxes, raw_keypoints):
                            pt1, pt2 = group[0], group[1]

                            if (len(pt1) < 3 or len(pt2) < 3 or pt1[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD or pt2[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD):
                                continue
                                                    
                            filtered_keypoints.append([[int(pt1[0]), int(pt1[1]), pt1[-1]],
                                                    [int(pt2[0]), int(pt2[1]), pt2[-1]]])
                            
                            x, y, w, h = box
                            x1, y1 = int(x - w/2), int(y - h/2)
                            x2, y2 = int(x + w/2), int(y + h/2)
                            filtered_box.append([int(x1), int(y1), int(x2), int(y2)])
                                    
                        keypoint_objects.append(
                            {
                                "name": result.names.get('bolt', 'unknown'),
                                "keypoints": filtered_keypoints,
                                "boxes": filtered_box
                            }
                        )

                    # Chuẩn hóa dữ liệu
                    if need_clean(keypoint_objects):
                        keypoint_objects = clean_data(keypoint_objects)
                    
                    
                    # ========== Segmentation ==========
                    r = self.detector.predict(frame)
                    
                    segment_objects= []

                    seg_boxes = []
                    if (r is not None) and hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if getattr(r.boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=int)
                        seg_boxes = xyxy.tolist()
                        
                    else:
                        xyxy = np.zeros((0, 4), dtype=int)
                        cls_ids = np.zeros((0,), dtype=int)

                    bbox_json = {"image": os.path.basename(orig_path), "bboxes": seg_boxes, "cls": cls_ids.tolist()}
                    with open(os.path.join(self.dir_bboxes, f"IMG_{name}.json"), "w", encoding="utf-8") as f:
                        json.dump(bbox_json, f, ensure_ascii=False, indent=2)

                    Hm, Wm = H, W
                    if (r is not None) and hasattr(r, "masks") and r.masks is not None and r.masks.data is not None:
                        masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
                        if masks.ndim == 3 and (masks.shape[1] != Hm or masks.shape[2] != Wm):
                            up = np.zeros((masks.shape[0], Hm, Wm), dtype=np.uint8)
                            for i in range(masks.shape[0]):
                                up[i] = cv2.resize(masks[i], (Wm, Hm), interpolation=cv2.INTER_NEAREST)

                            masks = up
                    else:
                        masks = np.zeros((0, Hm, Wm), dtype=np.uint8)
                        
                    # Tạo list segment object
                    for box, mask in zip(seg_boxes, masks):
                        segment_objects.append({'box':box, 'mask': mask})
                                     
                                     
                    # ====== Mapping keypoint và segment ======
                    
                    # Add union_box và union_box_pad vào mapping_object
                    mapping_object = self.match_objects(keypoint_objects, segment_objects)
                    
                    if mapping_object is not None and len(mapping_object) > 0:
                        for m in mapping_object:
                            m["union_box"] = temp = union_box(m["kp_box"], m["seg_box"])
                            x1p, y1p, x2p, y2p = pad_and_clip_box(temp[0], temp[1], temp[2], temp[3], self.cfg.PAD_RATIO, W, H, square=True)
                            m["union_box_pad"] = [x1p, y1p, x2p, y2p]
                        
                    # Xoay theo góc và crop
                    if mapping_object is not None and len(mapping_object) > 0:
                        for index, obj in enumerate(mapping_object):
                            
                            landmark_1 = obj['keypoints'][1][:2]
                            landmark_2 = obj['keypoints'][0][:2]
                            x1, y1, x2, y2 = obj['union_box_pad']
                            mask = obj['mask']
                            
                            # Tính góc
                            angle = arrow_angle(landmark_1, landmark_2)
                            
                            # Xoay ảnh và mask
                            rotated_frame, M = rotate_image(frame, 90-angle)
                            rotated_mask, _ = rotate_image(mask, 90-angle)

                            # Xoay box
                            box_pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                            rotated_box = transform_points(box_pts, M).astype(int)
                            x_min, y_min = rotated_box[:,0].min(), rotated_box[:,1].min()
                            x_max, y_max = rotated_box[:,0].max(), rotated_box[:,1].max()

                            center_x = (x_min + x_max) // 2
                            center_y = (y_min + y_max) // 2  
                            width  = abs(x_max - x_min)
                            height = abs(y_max - y_min)
                            
                            max_index = np.argmax([height, width])   # max_index = 0 → y lớn hơn, max_index = 1 → x lớn hơn
                            max_value = [height, width][max_index]

                            # Crop ảnh và mask theo box vuông
                            if max_index == 0:  # Chiều cao lớn hơn
                                    x1, y1, x2, y2 = center_x - max_value // 2, center_y - height // 2, center_x + max_value // 2, center_y + height // 2                              
                                    
                            else:  # Chiều rộng lớn hơn                                
                                    x1, y1, x2, y2 = center_x - width // 2, center_y - max_value // 2, center_x + width // 2, center_y + max_value // 2
                            
                            crop = rotated_frame[y1 : y2, x1 : x2]
                            crop_mask = rotated_mask[y1 : y2, x1 : x2]

                            # Lưu crop và crop_mask nếu hợp lệ
                            if crop.shape[0] > 0 and crop.shape[1] > 0 and crop_mask.shape[0] > 0 and crop_mask.shape[1] > 0:
                                    anomalies_crops.append([crop, crop_mask])
                                    obj['union_box_pad_rotated'] = [int(x1), int(y1), int(x2), int(y2)]
                                    obj['rotated_crop'] = crop
                                    obj['rotated_crop_mask'] = crop_mask
                                    
                                    crop_filename = os.path.join(self.keypoint_crops, f"KEYPOINT_{name}_{index}.jpg")
                                    mask_filename = os.path.join(self.masks_keypoint_crops, f"KEYPOINT_MASK_{name}_{index}.jpg")
                                    cv2.imwrite(crop_filename, crop)
                                    cv2.imwrite(mask_filename, ((crop_mask > 0) * 255).astype(np.uint8))
    
                            # Vẽ các thông tin lên frame_overlay
                            # Vẽ landmark
                            cv2.circle(frame_overlay, landmark_1, 6, (0,0,255), -1)
                            cv2.circle(frame_overlay, landmark_2, 6, (255,255,0), -1)
                            
                            # # Vẽ keypoint bbox
                            # cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

                            # Vẽ mũi tên đỏ→vàng
                            cv2.arrowedLine(frame_overlay, landmark_1, landmark_2, (0,255,255), 2, tipLength=0.3)

                            # Vẽ trục Ox tham chiếu (màu xanh)
                            axis_end = (landmark_1[0]+80, landmark_1[1])  # chiều ngang phải
                            cv2.arrowedLine(frame_overlay, landmark_1, axis_end, (255,0,0), 2, tipLength=0.3)
                            cv2.putText(frame_overlay, "Ox", (axis_end[0]+5, axis_end[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                            # # --- Chú thích ở trên cùng bbox ---
                            # cv2.putText(frame_overlay, f'{angle:.1f}', (x1, y1),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2) 
                            
                            # # Vẽ bbox keypoint
                            # cv2.rectangle(frame_overlay, 
                            #               (obj['kp_box'][0], obj['kp_box'][1]), (obj['kp_box'][2], obj['kp_box'][3]), 
                            #               (255, 255, 255), 2)
                            
                            # Vẽ bbox segment và polygon
                            # color_mask = np.zeros_like(frame_overlay)
                            # color_mask[obj['mask'] > 0] = [255, 255, 0]
                            # frame_overlay = cv2.addWeighted(frame_overlay, 1, color_mask, 0.3, 0)
                            
                            contour, _ = cv2.findContours(obj['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contour:
                                cv2.drawContours(frame_overlay, contour, -1, (0, 255, 0), 2)
                            # cv2.rectangle(frame_overlay, 
                            #               (obj['seg_box'][0], obj['seg_box'][1]), (obj['seg_box'][2], obj['seg_box'][3]), 
                            #               (255, 0, 255), 2)
                            
                            # Vẽ mapping box
                            cv2.rectangle(frame_overlay, 
                                        (obj['union_box_pad'][0], obj['union_box_pad'][1]), (obj['union_box_pad'][2], obj['union_box_pad'][3]), 
                                        (255, 255, 0), 2)
                    
                    # Hiển thị overlay
                    if len(anomalies_crops) == 0:
                        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
                        canvas_mask = np.zeros((200, 200, 3), dtype=np.uint8)
                    else:
                        canvas = concat_anomaly_crops([im['rotated_crop'] for im in mapping_object], target_h=300)
                        canvas_mask = concat_anomaly_crops([((im['rotated_crop_mask'] > 0) * 255).astype(np.uint8) for im in mapping_object], target_h=300) 
                        canvas_mask = cv2.cvtColor(canvas_mask, cv2.COLOR_GRAY2BGR) if len(canvas_mask.shape) == 2 else canvas_mask
                    
                    cv2.imshow("keypoint_crop", np.vstack((canvas, canvas_mask)))

                    # =========== Anomaly Detection ===========
                    
                    ng_any = False
                    per_objects: List[dict] = []
                    anomaly_crop_views: List[tuple] = []
                    heatmap_tiles: List[np.ndarray] = []

                    if mapping_object is not None and len(mapping_object) > 0:
                        for i in range(len(mapping_object)):
                            obj = mapping_object[i]
                            if 'rotated_crop' not in obj or 'rotated_crop_mask' not in obj:
                                continue
                            crop = obj['rotated_crop']
                            obj_mask_crop = obj['rotated_crop_mask']
                            x1, y1, x2, y2 = 0, 0, crop.shape[1], crop.shape[0]
                            x1p, y1p, x2p, y2p = 0, 0, obj_mask_crop.shape[1], obj_mask_crop.shape[0]
                            
                            crop_path = os.path.join(self.dir_crops, f"CROP_{name}_{i}.jpg")
                            if crop.size != 0:
                                cv2.imwrite(crop_path, crop)

                            out = self.anomaly.run_on_crop(crop, obj_mask_crop)

                            anomalies_full = []
                            cls_labels_i: List[str] = []  # Nhãn phân loại của object i (nếu đủ similarity)
                            
                            # ========== Anomaly Classification ==========
                            if out["is_ng"]:
                                for ak, a in enumerate(out["anomalies"]):
                                    cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                                    gx1 = x1p + cx1
                                    gy1 = y1p + cy1
                                    gx2 = x1p + cx2
                                    gy2 = y1p + cy2
                                    anomalies_full.append({
                                        "k": ak,
                                        "bbox_full": [int(gx1), int(gy1), int(gx2), int(gy2)],
                                        "bbox_in_object_crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
                                    })
                                    anom_crop = crop[gy1:gy2, gx1:gx2]
                                    if anom_crop.size != 0:
                                        anom_name = f"ANOM_{name}_obj{i}_{ak}.jpg"
                                        anom_path = os.path.join(self.dir_anom_crops, anom_name)
                                        cv2.imwrite(anom_path, anom_crop)

                                        # === Gọi API phân loại ===
                                        cls_res = self._classify_defect(anom_path)
                                        if cls_res is not None and cls_res.get("similarity", 0.0) >= float(self.cfg.DEFECT_CLS_SIM_THRESHOLD):
                                            label = cls_res.get("label", "NG")
                                            sim = float(cls_res.get("similarity", 0.0))
                                            anomalies_full[-1]["cls_label"] = label
                                            anomalies_full[-1]["cls_similarity"] = sim
                                            cls_labels_i.append(label)
                                        else:
                                            anomalies_full[-1]["cls_label"] = "NG"
                                            anomalies_full[-1]["cls_similarity"] = None

                            if (out["is_ng"] or self.cfg.SAVE_ALL_ANOMALIES) and len(anomalies_full) > 0:
                                obj_json = {
                                    "timestamp": name,
                                    "image": os.path.basename(orig_path),
                                    "object_index": i,
                                    "object_bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "score": round(float(out["score"]), 6),
                                    "overlap_ratio": round(float(out["overlap_ratio"]), 6),
                                    "anomalies": anomalies_full,
                                }
                                with open(os.path.join(self.dir_anom_bboxes, f"ANOM_{name}_obj{i}.json"), "w", encoding="utf-8") as f:
                                    json.dump(obj_json, f, ensure_ascii=False, indent=2)

                            mask_crop_vis = (obj_mask_crop * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(self.dir_masks, f"MASK_{name}_{i}.png"), mask_crop_vis)

                            if crop.size != 0 and out.get("hm_disp") is not None:
                                hm_tile = out["hm_disp"].copy()
                                label_small = f"obj{i} {'NG' if out.get('is_ng') else 'OK'}"
                                color = (0, 0, 255) if out.get("is_ng") else (0, 200, 0)
                                (tw, th), _ = cv2.getTextSize(label_small, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(hm_tile, (5, 5), (10 + tw + 5, 10 + th + 5), color, -1)
                                cv2.putText(hm_tile, label_small, (10, 10 + th),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                heatmap_tiles.append(hm_tile)

                            if crop.size != 0 and out.get("is_ng"):
                                ng_any = True

                            per_objects.append({
                                "index": i,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "score": round(float(out.get("score", 0.0)), 6) if crop.size != 0 else 0.0,
                                "is_ng": bool(out.get("is_ng")),
                                "overlap_ratio": round(float(out.get("overlap_ratio", 0.0)), 6) if crop.size != 0 else 0.0,
                                "crop": os.path.basename(crop_path) if crop.size != 0 else "",
                                "anomaly_count": len(anomalies_full),
                                "anomaly_bboxes_sample": [a["bbox_full"] for a in anomalies_full[:3]],
                            })

                            # ==== hiển thị crop với nhãn ====
                            crop_labeled = crop.copy()
                            if out.get("is_ng") and len(anomalies_full) > 0:
                                for idx_disp, a in enumerate(anomalies_full, start=1):
                                    cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                                    cv2.rectangle(crop_labeled, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

                                    # NEW: nếu có nhãn phân loại & similarity vượt ngưỡng thì in nhãn đó, ngược lại NG+số
                                    label_text = f"NG{idx_disp}"
                                    sim_val = a.get("cls_similarity", None)
                                    if self.cfg.DEFECT_CLS_ENABLE and a.get("cls_label") and sim_val is not None:
                                        if float(sim_val) >= float(self.cfg.DEFECT_CLS_SIM_THRESHOLD):
                                            label_text = f"{a['cls_label']} ({float(sim_val):.2f})"

                                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    tx1, ty1 = cx1, max(0, cy1 - th - 6)
                                    tx2, ty2 = cx1 + tw + 8, cy1
                                    cv2.rectangle(crop_labeled, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
                                    cv2.putText(crop_labeled, label_text, (cx1 + 3, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                (255, 255, 255), 2, cv2.LINE_AA)
                            anomaly_crop_views.append((crop_labeled, i))

                            # LƯU & SYNC per-object
                            if out.get("is_ng"):
                                # heatmap & amap (per-object)
                                if crop.size != 0 and out.get("hm_disp") is not None and out.get("am_raw") is not None:
                                    cv2.imwrite(os.path.join(self.dir_ng, f"NG_HEAT_{name}_{i}.jpg"), out["hm_disp"])
                                    np.save(os.path.join(self.dir_ng, f"NG_AMAP_{name}_{i}.npy"), out["am_raw"])

                                vis_name = f"ANOM_CROP_VIS_{name}_obj{i}.jpg"
                                vis_path = os.path.join(self.dir_anom_crop_vis, vis_name)
                                try:
                                    cv2.imwrite(vis_path, crop_labeled)
                                except Exception as e:
                                    print(f"[WARN] save vis failed: {e}")
                                    vis_path = None

                                # Mô tả gửi API Sync:
                                labels_unique = self._unique_preserve(cls_labels_i)
                                desc = ", ".join(labels_unique) if labels_unique else "NG"

                                if vis_path is not None:
                                    self._post_qc(file_path=vis_path,
                                                description=desc,
                                                inspection_time_str=insp_time_str,
                                                actual_result="2")
                            else:
                                if crop.size != 0 and os.path.isfile(crop_path):
                                    self._post_qc(file_path=crop_path,
                                                description="",
                                                inspection_time_str=insp_time_str,
                                                actual_result="1")
                                    
                            cv2.imshow('preview_overlay', frame_overlay) 
                    
                    # ==== HIỂN THỊ PANEL HEATMAP (CONCAT TẤT CẢ) ====
                    if len(heatmap_tiles) > 0:
                        hm_panel = concat_anomaly_crops(heatmap_tiles, target_h=300)
                    else:
                        hm_panel = np.zeros((max(1, int(H*self.cfg.DISPLAY_SCALE)),
                                             max(1, int(W*self.cfg.DISPLAY_SCALE)), 3), dtype=np.uint8)
                    cv2.imshow("heatmap", hm_panel)

                    # panel anomaly crops
                    if len(anomaly_crop_views) == 0 or not ng_any:
                        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
                    else:
                        canvas = concat_anomaly_crops([im for (im, _i) in anomaly_crop_views], target_h=300)
                    cv2.imshow("anomaly_crops_view", canvas)

                    # summary record
                    record = {
                        "timestamp": name,
                        "image": os.path.basename(orig_path),
                        "ng_detected": bool(ng_any),
                        "objects": per_objects
                    }
                    with open(self.results_json_path, "r+", encoding="utf-8") as f:
                        data = json.load(f)
                        data["results"].append(record)
                        f.seek(0)
                        json.dump(data, f, ensure_ascii=False, indent=2)
                        f.truncate()

                    saved_count += 1
                    await_confirm = True
        finally:
            cv2.destroyAllWindows()
            cam.stop()
            print("[INFO] Done.")
