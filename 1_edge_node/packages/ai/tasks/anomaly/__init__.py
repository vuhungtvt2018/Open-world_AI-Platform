# packages/ai/tasks/anomaly/__init__.py
import cv2
import numpy as np
from dataclasses import dataclass
from anomalib.deploy.inferencers import OpenVINOInferencer
from anomalib.utils.post_processing import superimpose_anomaly_map
from packages.utils.utils import pad_and_clip_box

@dataclass
class AnomalyConfig:
    input_size: int
    score_thres: float
    inside_overlap_min: float
    min_area_ratio: float
    bbox_pad_ratio: float
    save_all: bool
    show_all_boxes: bool
    amap_threshold: float = 0.7
    redo_center_crop: bool = True
    center_crop: int = 448

# class AnomalyInferencer:
#     def __init__(self, model_path: str, device: str, cfg: AnomalyConfig):
#         self.cfg = cfg
#         self.infer = OpenVINOInferencer(path=model_path, device=device)

#     def run_on_crop(self, crop_bgr, obj_mask_crop):
#         # Trả về: dict với score, is_ng, hm_disp, am_raw, overlap_ratio, anomalies(list), crop_vis
#         result = {
#             "score": 0.0,
#             "is_ng": False,
#             "hm_disp": None,
#             "am_raw": None,
#             "overlap_ratio": 0.0,
#             "anomalies": [],
#             "crop_vis": crop_bgr.copy() if crop_bgr is not None else None,
#         }
#         if crop_bgr is None or crop_bgr.size == 0:
#             return result

#         Hc, Wc = crop_bgr.shape[:2]
#         crop_rs = cv2.resize(crop_bgr, (self.cfg.input_size, self.cfg.input_size))
#         crop_rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB)
#         preds = self.infer.predict(image=crop_rgb)

#         score = float(getattr(preds, "pred_score", 0.0))
#         am_raw = preds.anomaly_map
#         if am_raw.ndim == 4:
#             am_raw = am_raw[0, 0]
#         elif am_raw.ndim == 3:
#             am_raw = am_raw[0]

#         # Nếu cấu hình yêu cầu crop center lại
#         if self.cfg.redo_center_crop:
#             OFF = (self.cfg.input_size - self.cfg.center_crop) // 2
#             if am_raw.shape[:2] != (self.cfg.center_crop, self.cfg.center_crop):
#                 am_raw = cv2.resize(
#                     am_raw, (self.cfg.center_crop, self.cfg.center_crop),
#                     interpolation=cv2.INTER_LINEAR
#                 )
#             am_full = np.zeros((self.cfg.input_size, self.cfg.input_size), dtype=am_raw.dtype)
#             am_full[OFF:OFF + self.cfg.center_crop, OFF:OFF + self.cfg.center_crop] = am_raw
#             am_raw = am_full  # từ đây trở đi am_raw là input_size x input_size

#         # mask -> am size
#         obj_mask_resz = cv2.resize(
#             (obj_mask_crop * 255).astype(np.uint8),
#             (am_raw.shape[1], am_raw.shape[0]),
#             interpolation=cv2.INTER_NEAREST
#         )
#         obj_mask_resz = (obj_mask_resz > 0).astype(np.uint8)

#         # Chuẩn hóa & threshold sơ bộ
#         am_norm = am_raw.copy()
#         thr = float(self.cfg.amap_threshold)
#         am_norm[am_norm < thr] = 0.0

#         # Quyết định NG/OK chỉ theo score (in_object=True), rồi nếu OK thì am_norm=0 luôn
#         is_ng_base = score >= self.cfg.score_thres
#         in_object = True
#         is_ng = bool((is_ng_base and in_object) or (self.cfg.save_all and in_object))
#         if not is_ng:
#             # nếu OK thì am_norm = 0 hoàn toàn
#             am_norm[...] = 0.0

#         # Từ am_norm → nhị phân (để tính overlap & tìm vùng lỗi)
#         am_u8 = (am_norm * 255).astype(np.uint8)
#         _, high_bin = cv2.threshold(am_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         high_bin = (high_bin > 0).astype(np.uint8)

#         total_high = int(high_bin.sum())
#         inside_high = int((high_bin * obj_mask_resz).sum())
#         overlap_ratio = (inside_high / total_high) if total_high > 0 else 0.0

#         # Heatmap hiển thị: giữ nguyên cách tính cũ (không set đen nếu OK)
#         am_masked = am_norm * obj_mask_resz
#         hm = superimpose_anomaly_map(anomaly_map=am_masked, image=crop_rs, normalize=True)
#         hm_disp = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

#         # Trích vùng lỗi theo high_bin đã (có thể là 0 nếu OK)
#         hb_inside = (high_bin * obj_mask_resz).astype(np.uint8) * 255
#         hb_crop = cv2.resize(hb_inside, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

#         min_area_px = int(round(self.cfg.min_area_ratio * (Wc * Hc)))
#         contours, _ = cv2.findContours(hb_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         anomalies = []
#         crop_vis = crop_bgr.copy()
#         ak = 0
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < max(1, min_area_px):
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             from packages.utils.utils import pad_and_clip_box
#             cx1, cy1, cx2, cy2 = pad_and_clip_box(x, y, x + w, y + h, self.cfg.bbox_pad_ratio, Wc, Hc)
#             anomalies.append({
#                 "k": ak,
#                 "bbox_in_object_crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
#                 "area_px": int(area),
#             })
#             if len(anomalies) > 0 and (self.cfg.show_all_boxes or is_ng):
#                 cv2.rectangle(crop_vis, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
#                 label = f"a{ak}"

#                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                 cv2.rectangle(crop_vis, (cx1, max(0, cy1 - th - 4)), (cx1 + tw + 4, cy1), (0, 0, 255), -1)
#                 cv2.putText(crop_vis, label, (cx1 + 2, max(th, cy1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5, (255, 255, 255), 1, cv2.LINE_AA)
#             ak += 1
            
#         is_ng = is_ng and len(anomalies) > 0

#         result.update({
#             "score": score,
#             "is_ng": is_ng,
#             "hm_disp": hm_disp,
#             "am_raw": am_raw,
#             "am_masked": am_masked,
#             "overlap_ratio": overlap_ratio,
#             "anomalies": anomalies,
#             "crop_vis": crop_vis,
#         })
#         return result









class AnomalyInferencer:
    def __init__(self, model_path: str, device: str, cfg: AnomalyConfig):
        self.cfg = cfg
        self.infer = OpenVINOInferencer(path=model_path, device=device)
    
    def _init_result(self, crop_bgr):
        return {
            "score": 0.0,
            "is_ng": False,
            "hm_disp": None,
            "am_raw": None,
            "overlap_ratio": 0.0,
            "anomalies": [],
            "crop_vis": crop_bgr.copy() if crop_bgr is not None else None,
        }
        
    def _preprocess(self, crop_bgr):
        crop_rs = cv2.resize(crop_bgr, (self.cfg.input_size, self.cfg.input_size))
        crop_rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB)
        return crop_rs, crop_rgb

    def _predict(self, crop_rgb):
        preds = self.infer.predict(image=crop_rgb)

        score = float(getattr(preds, "pred_score", 0.0))
        am_raw = preds.anomaly_map
        if am_raw.ndim == 4:
            am_raw = am_raw[0, 0]
        elif am_raw.ndim == 3:
            am_raw = am_raw[0]

        return score, am_raw

    def _process_anomaly_map(self, am_raw, obj_mask_crop):
        # Nếu cấu hình yêu cầu crop center lại
        if self.cfg.redo_center_crop:
            OFF = (self.cfg.input_size - self.cfg.center_crop) // 2
            if am_raw.shape[:2] != (self.cfg.center_crop, self.cfg.center_crop):
                am_raw = cv2.resize(
                    am_raw, (self.cfg.center_crop, self.cfg.center_crop),
                    interpolation=cv2.INTER_LINEAR
                )
            am_full = np.zeros((self.cfg.input_size, self.cfg.input_size), dtype=am_raw.dtype)
            am_full[OFF:OFF + self.cfg.center_crop, OFF:OFF + self.cfg.center_crop] = am_raw
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

    def _threshold(self, am_raw, score):
        # Chuẩn hóa & threshold sơ bộ
        am_norm = am_raw.copy()
        thr = float(self.cfg.amap_threshold)
        am_norm[am_norm < thr] = 0.0

        # Quyết định NG/OK chỉ theo score (in_object=True), rồi nếu OK thì am_norm=0 luôn
        is_ng_base = score >= self.cfg.score_thres
        in_object = True
        is_ng = bool((is_ng_base and in_object) or (self.cfg.save_all and in_object))
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

        min_area_px = int(round(self.cfg.min_area_ratio * (Wc * Hc)))
        contours, _ = cv2.findContours(hb_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anomalies = []
        ak = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(1, min_area_px):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            from packages.utils.utils import pad_and_clip_box
            cx1, cy1, cx2, cy2 = pad_and_clip_box(x, y, x + w, y + h, self.cfg.bbox_pad_ratio, Wc, Hc)
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

    def run_on_crop(self, crop_bgr, obj_mask_crop):
        """
        1. init result
        2. return if crop is None
        3. preprocess
        4. predict
        5. process anomaly map
        6. resize mask
        7. threshold
        8. compute overlap
        9. generate heatmap
        10. extract anomalies
        11. update is_ng
        12. draw boxes
        13. update result
        14. return result
        """
        result = self._init_result(crop_bgr)
        if crop_bgr is None or crop_bgr.size == 0:
            return result

        # 1. PREPROCESS
        crop_rs, crop_rgb = self._preprocess(crop_bgr)
        # 2. PREDICT
        score, am_raw = self._predict(crop_rgb)
        # 3. PROCESS ANOMALY MAP
        am_raw = self._process_anomaly_map(am_raw, obj_mask_crop)
        # 4. RESIZE MASK
        obj_mask_resz = self._resize_mask(obj_mask_crop, am_raw)
        # 5. THRESHOLD
        am_norm, is_ng_base = self._threshold(am_raw, score)
        # 6. COMPUTE OVERLAP
        high_bin, overlap_ratio = self._compute_overlap(am_norm, obj_mask_resz)
        # 7. GENERATE HEATMAP
        am_masked, hm_disp = self._generate_heatmap(am_norm, obj_mask_resz, crop_rs)
        # 8. EXTRACT ANOMALIES
        anomalies = self._extract_anomalies(high_bin, obj_mask_resz, crop_bgr)
        is_ng = is_ng_base and len(anomalies) > 0
        
        crop_vis = crop_bgr.copy()
        if len(anomalies) > 0 and (self.cfg.save_all or is_ng):
            crop_vis = self._draw_boxes(anomalies, crop_vis)

        result.update({
            "score": score,
            "is_ng": is_ng,
            "hm_disp": hm_disp,
            "am_raw": am_raw,
            "am_masked": am_masked,
            "overlap_ratio": overlap_ratio,
            "anomalies": anomalies,
            "crop_vis": crop_vis,
        })

        return result