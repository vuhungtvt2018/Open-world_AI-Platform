# packages/ai/tasks/anomaly/yolo_inferencer.py
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from packages.core.config import AnomalyConfig
from packages.ai.tasks.base_task import BaseVisionTask, InferenceResult, BoundingBox
from packages.utils.utils import pad_and_clip_box


class YOLOInferencer(BaseVisionTask):
    """
    Inferencer phát hiện vùng bất thường bằng Ultralytics YOLO.

    Class này có cùng hàm run(image, obj_mask_crop) và cùng kiểu trả về
    InferenceResult với AnomalibInferencer, vì vậy Pipeline không cần thay đổi
    luồng xử lý phía sau.

    Hỗ trợ:
    - YOLO detection: bbox được dùng làm vùng anomaly.
    - YOLO segmentation: mask của model được dùng làm vùng anomaly.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        config: AnomalyConfig | Dict[str, Any] | None,
    ):
        self.input_size = int(self._get_config(config, "input_size", 640))
        self.score_thres = float(self._get_config(config, "score_thres", 0.5))
        self.iou_thres = float(self._get_config(config, "iou_thres", 0.45))
        self.max_det = int(self._get_config(config, "max_det", 10))
        self.agnostic_nms = bool(
            self._get_config(config, "agnostic_nms", True)
        )
        self.task = str(self._get_config(config, "task", "segment"))
        self.inside_overlap_min = float(
            self._get_config(config, "inside_overlap_min", 0.0)
        )
        self.min_area_ratio = float(
            self._get_config(config, "min_area_ratio", 1e-3)
        )
        self.bbox_pad_ratio = float(
            self._get_config(config, "bbox_pad_ratio", 0.02)
        )
        self.save_all = bool(self._get_config(config, "save_all", False))
        self.show_all_boxes = bool(
            self._get_config(config, "show_all_boxes", True)
        )

        super().__init__(model_path, device, config)

    @staticmethod
    def _get_config(
        config: AnomalyConfig | Dict[str, Any] | None,
        key: str,
        default: Any,
    ) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        if config is not None:
            return getattr(config, key, default)
        return default

    def _get_yolo_device(self) -> str | int:
        """Chuyển tên device của Anomalib sang định dạng Ultralytics."""
        if isinstance(self.device, int):
            return self.device

        device = str(self.device).strip().lower()

        if device == "cpu":
            return "cpu"
        if device in {"gpu", "cuda"}:
            return 0
        if device.startswith("cuda:"):
            try:
                return int(device.split(":", maxsplit=1)[1])
            except ValueError:
                return 0

        return device

    def load_model(self) -> None:
        self.model = YOLO(
            self.model_path,
            task=self.task,
        )

    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Ultralytics tự resize và normalize ảnh trong model.predict()."""
        if image is None or image.size == 0:
            return None
        return image

    def predict(
        self,
        preprocessed_data: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Any]:
        if preprocessed_data is None:
            return None, None

        predictions = self.model.predict(
            source=preprocessed_data,
            imgsz=self.input_size,
            conf=self.score_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            device=self._get_yolo_device(),
            verbose=False,
        )

        prediction = predictions[0] if predictions else None
        return preprocessed_data, prediction

    def postprocess(
        self,
        raw_output: Tuple[Optional[np.ndarray], Any],
        obj_mask_crop: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> InferenceResult:
        crop_bgr, prediction = raw_output

        if crop_bgr is None or crop_bgr.size == 0 or prediction is None:
            return self._empty_result(crop_bgr, raw_output)

        height, width = crop_bgr.shape[:2]
        object_mask = self._prepare_object_mask(
            obj_mask_crop,
            width=width,
            height=height,
        )

        anomaly_map = np.zeros((height, width), dtype=np.float32)
        boxes = []
        draw_items = []

        yolo_boxes = prediction.boxes
        yolo_masks = prediction.masks

        if yolo_boxes is not None:
            for index in range(len(yolo_boxes)):
                yolo_box = yolo_boxes[index]
                confidence = float(yolo_box.conf.item())
                class_id = int(yolo_box.cls.item())

                xyxy = (
                    yolo_box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                x1, y1, x2, y2 = self._clip_box(
                    xyxy,
                    width=width,
                    height=height,
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                region_mask = self._get_region_mask(
                    yolo_masks=yolo_masks,
                    index=index,
                    bbox=(x1, y1, x2, y2),
                    width=width,
                    height=height,
                )

                area_px = int(region_mask.sum())
                min_area_px = max(
                    1,
                    int(round(self.min_area_ratio * width * height)),
                )
                if area_px < min_area_px:
                    continue

                inside_px = int((region_mask * object_mask).sum())
                inside_ratio = inside_px / area_px if area_px > 0 else 0.0
                if inside_ratio < self.inside_overlap_min:
                    continue

                padded_box = pad_and_clip_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    self.bbox_pad_ratio,
                    width,
                    height,
                )
                px1, py1, px2, py2 = map(int, padded_box)

                class_name = self._get_class_name(
                    prediction.names,
                    class_id,
                )
                
                print(
                    "[YOLO ANOMALY]",
                    f"names={prediction.names}",
                    f"class_id={class_id}",
                    f"class_name={class_name}",
                    f"confidence={confidence:.4f}",
                )

                boxes.append(
                    BoundingBox(
                        xmin=float(px1),
                        ymin=float(py1),
                        xmax=float(px2),
                        ymax=float(py2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
                draw_items.append(
                    {
                        "bbox": (px1, py1, px2, py2),
                        "confidence": confidence,
                        "class_name": class_name,
                    }
                )

                anomaly_map = np.maximum(
                    anomaly_map,
                    region_mask.astype(np.float32) * confidence,
                )

        anomaly_map_masked = anomaly_map * object_mask
        score = max((box.confidence for box in boxes), default=0.0)
        is_ng = len(boxes) > 0

        total_anomaly = int((anomaly_map > 0).sum())
        inside_anomaly = int(
            ((anomaly_map > 0).astype(np.uint8) * object_mask).sum()
        )
        overlap_ratio = (
            inside_anomaly / total_anomaly
            if total_anomaly > 0
            else 0.0
        )

        crop_vis = crop_bgr.copy()
        if draw_items and (self.show_all_boxes or self.save_all or is_ng):
            crop_vis = self._draw_boxes(crop_vis, draw_items)

        heatmap_display = self._generate_heatmap(
            crop_bgr,
            anomaly_map_masked,
        )

        return InferenceResult(
            classification_score=float(score),
            is_ng=is_ng,
            boxes=boxes,
            heatmap_display=heatmap_display,
            anomaly_map_raw=anomaly_map,
            anomaly_map_masked=anomaly_map_masked,
            overlap_ratio=float(overlap_ratio),
            visualized_image=crop_vis,
            raw_output=raw_output,
        )

    def run(
        self,
        image: np.ndarray,
        obj_mask_crop: Optional[np.ndarray],
        *args,
        **kwargs,
    ) -> InferenceResult:
        preprocessed = self.preprocess(image)
        raw_output = self.predict(preprocessed)
        return self.postprocess(
            raw_output,
            obj_mask_crop,
            *args,
            **kwargs,
        )

    @staticmethod
    def _prepare_object_mask(
        obj_mask_crop: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> np.ndarray:
        if obj_mask_crop is None or obj_mask_crop.size == 0:
            return np.ones((height, width), dtype=np.uint8)

        mask = cv2.resize(
            obj_mask_crop.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def _clip_box(
        xyxy: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        x1 = int(np.clip(np.floor(xyxy[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(xyxy[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(xyxy[2]), 0, width))
        y2 = int(np.clip(np.ceil(xyxy[3]), 0, height))
        return x1, y1, x2, y2

    @staticmethod
    def _get_region_mask(
        yolo_masks: Any,
        index: int,
        bbox: Tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> np.ndarray:
        if yolo_masks is not None and index < len(yolo_masks.data):
            mask = (
                yolo_masks.data[index]
                .detach()
                .cpu()
                .numpy()
            )
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            return (mask > 0.5).astype(np.uint8)

        x1, y1, x2, y2 = bbox
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    @staticmethod
    def _get_class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, f"class_{class_id}"))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return f"class_{class_id}"

    @staticmethod
    def _generate_heatmap(
        crop_bgr: np.ndarray,
        anomaly_map: np.ndarray,
    ) -> np.ndarray:
        heatmap_u8 = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(
            crop_bgr,
            0.6,
            heatmap_bgr,
            0.4,
            0.0,
        )

        # Giữ cùng convention với AnomalibInferencer hiện tại.
        return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _draw_boxes(
        image: np.ndarray,
        draw_items: list[dict],
    ) -> np.ndarray:
        for item in draw_items:
            x1, y1, x2, y2 = item["bbox"]
            label = (
                f"{item['class_name']} "
                f"{item['confidence']:.2f}"
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            label_y1 = max(0, y1 - text_h - 6)
            cv2.rectangle(
                image,
                (x1, label_y1),
                (min(image.shape[1], x1 + text_w + 6), y1),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 3, max(text_h, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return image

    @staticmethod
    def _empty_result(
        crop_bgr: Optional[np.ndarray],
        raw_output: Any,
    ) -> InferenceResult:
        return InferenceResult(
            classification_score=0.0,
            is_ng=False,
            boxes=[],
            heatmap_display=None,
            anomaly_map_raw=None,
            anomaly_map_masked=None,
            overlap_ratio=0.0,
            visualized_image=(
                crop_bgr.copy()
                if crop_bgr is not None
                else None
            ),
            raw_output=raw_output,
        )