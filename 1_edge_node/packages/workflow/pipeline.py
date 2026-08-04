# packages/workflow/pipeline.py
from pathlib import Path
import sys
import os

# Resolve ROOT directory (2 levels up from edge-agent: edge)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import requests
from typing import List, Dict, Optional
from datetime import datetime

from packages.core.config import AppConfig
from services.session import make_session_dir
from packages.utils.utils import (ensure_dirs, compute_iou)
from packages.ai.tasks.detection import YOLODetector
from packages.ai.tasks.anomaly import AnomalyInferencer, AnomalyConfig

class Pipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        print("[PIPELINE] Khởi tạo các thư mục...")
        self.prod_dir = os.path.join(self.cfg.CAPTURE_DIR, self.cfg.PRODUCT_NAME)
        ensure_dirs(self.prod_dir)

        s_root, orig, crops, masks, bboxes, ng, anom_box, anom_crop, anom_vis, kp_crop, mk_crop = make_session_dir(self.prod_dir)
        self.session_root = s_root
        self.dir_original = orig
        self.dir_crops = crops
        self.dir_masks = masks
        self.dir_bboxes = bboxes
        self.dir_ng = ng
        self.dir_anom_bboxes = anom_box
        self.dir_anom_crops = anom_crop
        self.dir_anom_vis = anom_vis
        self.dir_keypoint_crops = kp_crop
        self.dir_masks_keypoint_crops = mk_crop

        print("[PIPELINE] Khởi tạo mô hình YOLODetector Segmentation...")
        self.detector = YOLODetector(self.cfg.MODEL_PATH)
        
        self.keypoint_detection = None
        if self.cfg.KEYPOINT_DETECTION:
            print("[PIPELINE] Khởi tạo mô hình YOLODetector Keypoints...")
            self.keypoint_detection = YOLODetector(self.cfg.KEYPOINTS_MODEL_PATH)

        print("[PIPELINE] Khởi tạo mô hình Anomaly (Anomalib)...")
        anom_cfg = AnomalyConfig(
            input_size=self.cfg.ANOMALY_INPUT_SIZE,
            score_thres=self.cfg.ANOMALY_SCORE_THRESHOLD,
            inside_overlap_min=self.cfg.ANOMALY_INSIDE_OVERLAP_MIN,
            min_area_ratio=self.cfg.ANOMALY_MIN_AREA_RATIO,
            bbox_pad_ratio=self.cfg.ANOMALY_BBOX_PAD_RATIO,
            save_all=self.cfg.SAVE_ALL_ANOMALIES,
            show_all_boxes=self.cfg.SHOW_ALL_ANOMALY_BOXES,
            amap_threshold=self.cfg.ANOMALY_AMAP_THRESHOLD,
            redo_center_crop=self.cfg.REDO_CENTER_CROP,
            center_crop=self.cfg.CENTER_CROP
        )
        self.anomaly = AnomalyInferencer(
            model_path=self.cfg.ANOMALY_MODEL_PATH,
            device=self.cfg.ANOMALY_DEVICE,
            cfg=anom_cfg
        )
        
        # Thử tải mô hình vào bộ nhớ (warmup)
        print("[PIPELINE] Khởi động Anomalib (warmup)...")
        dummy_img = np.zeros((self.cfg.ANOMALY_INPUT_SIZE, self.cfg.ANOMALY_INPUT_SIZE, 3), dtype=np.uint8)
        dummy_mask = np.ones((self.cfg.ANOMALY_INPUT_SIZE, self.cfg.ANOMALY_INPUT_SIZE), dtype=np.uint8)
        self.anomaly.run_on_crop(dummy_img, dummy_mask)
        print("[PIPELINE] Sẵn sàng!")

    def match_objects(self, kp_objs: List[Dict], seg_objs: List[Dict], iou_thresh: float = 0.5):
        """
        Nối (match) các kết quả keypoint và segmentation dựa trên IOU của bounding box.
        """
        matched = []
        seg_used = set()
        
        for kp_obj in kp_objs:
            kp_boxes = kp_obj['boxes']
            kp_keypoints = kp_obj['keypoints']
            for i, k_box in enumerate(kp_boxes):
                best_iou = 0
                best_seg_idx = -1
                
                for j, seg_obj in enumerate(seg_objs):
                    if j in seg_used: continue
                    iou = compute_iou(k_box, seg_obj['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_seg_idx = j
                        
                if best_iou > iou_thresh:
                    matched.append({
                        'kp_box': k_box,
                        'keypoints': kp_keypoints[i],
                        'seg_box': seg_objs[best_seg_idx]['box'],
                        'mask': seg_objs[best_seg_idx]['mask']
                    })
                    seg_used.add(best_seg_idx)
        return matched

    def _format_inspection_time(self, timestamp_str: str) -> str:
        """Chuyển đổi chuỗi YYYYMMDD_HHMMSS_ffffff thành YYYY-MM-DD HH:MM:SS"""
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp_str

    def _classify_defect(self, image_path: str) -> Optional[Dict]:
        """Gọi API phân loại lỗi ngoài và trả về nhãn lỗi + similarity."""
        if not self.cfg.DEFECT_CLS_ENABLE:
            return None
        url = self.cfg.DEFECT_CLS_URL
        if not url:
            return None

        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                data = {
                    "top_k": str(self.cfg.DEFECT_CLS_TOPK),
                    "metric": self.cfg.DEFECT_CLS_METRIC,
                    "ItemCode": self.cfg.ITEM_CODE or "",
                }
                resp = requests.post(url, data=data, files=files, timeout=float(self.cfg.API_TIMEOUT), proxies={"http": None, "https": None})
            if not (200 <= resp.status_code < 300):
                print(f"[CLS] FAIL {resp.status_code}: {resp.text[:200]}")
                return None
            res_json = resp.json()
            if not isinstance(res_json, list) or len(res_json) == 0:
                    return None
            rec = res_json[0]
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
            print(f"[DEFECT_CLS ERROR] Failed to classify {image_path}: {e}")
            return None
