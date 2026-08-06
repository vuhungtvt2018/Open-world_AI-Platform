# packages/ai/tasks/anomaly/__init__.py
import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import asdict

from anomalib.deploy.inferencers import OpenVINOInferencer
from anomalib.utils.post_processing import superimpose_anomaly_map

from packages.core.config import AnomalyConfig
from packages.ai.tasks.base_task import BaseVisionTask, InferenceResult, BoundingBox


"""
06082026 - KHAI - Make AnomalyInferencer a subclass of BaseVisionTask, refactor codes
"""
class AnomalyInferencer(BaseVisionTask):
    def __init__(self, model_path: str, device: str, config: AnomalyConfig | Dict[str, Any] | None):
        super().__init__(model_path, device, config)
        if isinstance(self.config, AnomalyConfig):
            self.input_size = getattr(self.config, "input_size", 256)
            self.save_all = getattr(self.config, "save_all", False)
            self.redo_center_crop = getattr(self.config, "redo_center_crop", False)
            self.center_crop = getattr(self.config, "center_crop", 448)
            self.amap_threshold = getattr(self.config, "amap_threshold", 0.7)
            self.score_thres = getattr(self.config, "score_thres", 0.5)
            self.min_area_ratio = getattr(self.config, "min_area_ratio", 1e-3)
            self.bbox_pad_ratio = getattr(self.config, "bbox_pad_ratio", 0.02)
        elif isinstance(self.config, dict):
            self.input_size = int(self.config.get("input_size", 256))
            self.save_all = bool(self.config.get("save_all", False))
            self.redo_center_crop = bool(self.config.get("redo_center_crop", False))
            self.center_crop = int(self.config.get("center_crop", 448))
            self.amap_threshold = float(self.config.get("amap_threshold", 0.7))
            self.score_thres = float(self.config.get("score_thres", 0.5))
            self.min_area_ratio = float(self.config.get("min_area_ratio", 1e-3))
            self.bbox_pad_ratio = float(self.config.get("bbox_pad_ratio", 0.02))
        else:
            self.input_size = 256
            self.save_all = False
            self.redo_center_crop = False
            self.center_crop = 448
            self.amap_threshold = 0.7
            self.score_thres = 0.5
            self.min_area_ratio = 1e-3
            self.bbox_pad_ratio = 0.02

    def load_model(self):
        self.model = OpenVINOInferencer(
            path=self.model_path,
            device=self.device,
            config=asdict(self.config) or None,
        )
        self.model.predict(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
    
    def _init_result(self, crop_bgr: Optional[np.ndarray]):
        return {
            "score": 0.0,
            "is_ng": False,
            "hm_disp": None,
            "am_raw": None,
            "overlap_ratio": 0.0,
            "anomalies": [],
            "crop_vis": crop_bgr.copy() if crop_bgr is not None else None,
        }

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tiền xử lý ảnh: resize, normalize, chuyển thành tensor..."""
        if image is None or image.size == 0:
            return None, None, None
        
        crop_rs = cv2.resize(image, (self.input_size, self.input_size))
        crop_rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB)
        return crop_rs, crop_rgb, image

    def predict(
        self,
        preprocessed_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Đưa dữ liệu qua mạng neural để lấy raw output"""
        crop_rs, crop_rgb, crop_bgr = preprocessed_data
        if crop_rgb is None or crop_rgb.size == 0:
            return 0.0, crop_rs, crop_rgb, None

        preds = self.model.predict(image=crop_rgb)

        score = float(getattr(preds, "pred_score", 0.0))
        am_raw = preds.anomaly_map
        if am_raw.ndim == 4:
            am_raw = am_raw[0, 0]
        elif am_raw.ndim == 3:
            am_raw = am_raw[0]

        return score, crop_rs, crop_rgb, crop_bgr, am_raw

    def postprocess(
        self,
        raw_output: Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        obj_mask_crop: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> InferenceResult:
        """Hậu xử lý raw output (NMS, scale bboxes...) và trả về InferenceResult chuẩn"""
        score, crop_rs, _, crop_bgr, am_raw = raw_output

        # Trường hợp ảnh rỗng / lỗi
        if crop_bgr is None or crop_bgr.size == 0 or am_raw is None:
            init_res = self._init_result(crop_bgr)
            return InferenceResult(
                classification_score=init_res["score"],
                is_ng=init_res["is_ng"],
                heatmap_display=init_res["hm_disp"],
                anomaly_map_raw=init_res["am_raw"],
                overlap_ratio=init_res["overlap_ratio"],
                visualized_image=init_res["crop_vis"],
            )

        # 1. Process Anomaly Map & Mask
        am_raw = self._process_anomaly_map(am_raw, obj_mask_crop)
        obj_mask_resz = self._resize_mask(obj_mask_crop, am_raw)

        # 2. Thresholding
        am_norm, is_ng_base = self._threshold(am_raw, score)

        # 3. Compute Overlap
        high_bin, overlap_ratio = self._compute_overlap(am_norm, obj_mask_resz)

        # 4. Generate Heatmap
        am_masked, hm_disp = self._generate_heatmap(am_norm, obj_mask_resz, crop_rs)

        # 5. Extract Anomalies (Raw dicts)
        anomalies_raw = self._extract_anomalies(high_bin, obj_mask_resz, crop_bgr)
        is_ng = is_ng_base and len(anomalies_raw) > 0

        # 6. Draw Boxes Visual
        crop_vis = crop_bgr.copy()
        if len(anomalies_raw) > 0 and (self.save_all or is_ng):
            crop_vis = self._draw_boxes(anomalies_raw, crop_vis)

        # 7. Convert anomalies dicts -> Pydantic BoundingBox instances
        boxes: List[BoundingBox] = []
        for anomaly in anomalies_raw:
            bbox = anomaly["bbox_in_object_crop"]
            boxes.append(
                BoundingBox(
                    xmin=float(bbox[0]),
                    ymin=float(bbox[1]),
                    xmax=float(bbox[2]),
                    ymax=float(bbox[3]),
                    confidence=float(score),
                    class_id=int(anomaly["k"]),
                    class_name=f"a{anomaly['k']}",
                )
            )

        return InferenceResult(
            classification_score=score,
            is_ng=is_ng,
            boxes=boxes,
            heatmap_display=hm_disp,
            anomaly_map_raw=am_raw,
            anomaly_map_masked=am_masked,
            overlap_ratio=overlap_ratio,
            visualized_image=crop_vis,
            raw_output=raw_output,
        )

    def run(self, image: np.ndarray, obj_mask_crop: Optional[np.ndarray], *args, **kwargs) -> InferenceResult:
        """
        Hàm chính Pipeline cho 1 lượt inference hoàn chỉnh.
        Workflow sẽ chỉ gọi hàm này.
        """
        preprocessed = self.preprocess(image)
        raw_output = self.predict(preprocessed)
        result = self.postprocess(raw_output, obj_mask_crop, *args, **kwargs)
        return result

    def _process_anomaly_map(self, am_raw, obj_mask_crop):
        # Nếu cấu hình yêu cầu crop center lại
        if self.redo_center_crop:
            OFF = (self.input_size - self.center_crop) // 2
            if am_raw.shape[:2] != (self.center_crop, self.center_crop):
                am_raw = cv2.resize(
                    am_raw, (self.center_crop, self.center_crop),
                    interpolation=cv2.INTER_LINEAR
                )
            am_full = np.zeros((self.input_size, self.input_size), dtype=am_raw.dtype)
            am_full[OFF:OFF + self.center_crop, OFF:OFF + self.center_crop] = am_raw
            am_raw = am_full  # từ đây trở đi am_raw là input_size x input_size
        
        return am_raw

    def _resize_mask(self, obj_mask_crop, am_raw):
        # mask -> am size
        obj_mask_resz = cv2.resize(
            (obj_mask_crop * 255).astype(np.uint8),
            (am_raw.shape[1], am_raw.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        obj_mask_resz = (obj_mask_resz > 0).astype(np.uint8)
        
        return obj_mask_resz

    def _threshold(self, am_raw: np.ndarray, score: float):
        # Chuẩn hóa & threshold sơ bộ
        am_norm = am_raw.copy()
        thr = float(self.amap_threshold)
        am_norm[am_norm < thr] = 0.0

        # Quyết định NG/OK chỉ theo score (in_object=True), rồi nếu OK thì am_norm=0 luôn
        is_ng_base = score >= self.score_thres
        in_object = True
        is_ng = bool((is_ng_base and in_object) or (self.save_all and in_object))
        if not is_ng:
            # nếu OK thì am_norm = 0 hoàn toàn
            am_norm[...] = 0.0
        
        return am_norm, is_ng

    def _compute_overlap(self, am_norm, obj_mask_resz):
        # Từ am_norm → nhị phân (để tính overlap & tìm vùng lỗi)
        am_u8 = (am_norm * 255).astype(np.uint8)
        _, high_bin = cv2.threshold(am_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_bin = (high_bin > 0).astype(np.uint8)

        total_high = int(high_bin.sum())
        inside_high = int((high_bin * obj_mask_resz).sum())
        overlap_ratio = (inside_high / total_high) if total_high > 0 else 0.0

        return high_bin, overlap_ratio

    def _generate_heatmap(self, am_norm, obj_mask_resz, crop_rs):
        # Heatmap hiển thị: giữ nguyên cách tính cũ (không set đen nếu OK)
        am_masked = am_norm * obj_mask_resz
        hm = superimpose_anomaly_map(anomaly_map=am_masked, image=crop_rs, normalize=True)
        hm_disp = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

        return am_masked, hm_disp

    def _extract_anomalies(self, high_bin, obj_mask_resz, crop_bgr):
        # Trích vùng lỗi theo high_bin đã (có thể là 0 nếu OK)
        Hc, Wc = crop_bgr.shape[:2]
        hb_inside = (high_bin * obj_mask_resz).astype(np.uint8) * 255
        hb_crop = cv2.resize(hb_inside, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

        min_area_px = int(round(self.min_area_ratio * (Wc * Hc)))
        contours, _ = cv2.findContours(hb_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anomalies = []
        ak = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(1, min_area_px):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            from packages.utils.utils import pad_and_clip_box
            cx1, cy1, cx2, cy2 = pad_and_clip_box(x, y, x + w, y + h, self.bbox_pad_ratio, Wc, Hc)
            anomalies.append({
                "k": ak,
                "bbox_in_object_crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "area_px": int(area),
            })
            ak += 1
            
        return anomalies
        
    def _draw_boxes(self, anomalies, crop_vis):
        for anomaly in anomalies:
            x1, y1, x2, y2 = anomaly['bbox_in_object_crop']
            cv2.rectangle(crop_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"a{anomaly['k']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(crop_vis, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), (0, 0, 255), -1)
            cv2.putText(crop_vis, label, (x1 + 2, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return crop_vis