# app/tools/data_collector.py
import os, json, argparse, glob
from datetime import datetime

import cv2
import numpy as np

from app.detectors.yolo_detector import YOLODetector

# project modules
from app.config import AppConfig
from app.camera import RTSP_Threaded_Camera
from app.utils import ensure_dirs, ts, pad_and_clip_box
from app.session import make_session_dir

def resolve_collect_root(cfg: AppConfig, out_dir_cli: str | None) -> str:
    if out_dir_cli:
        return out_dir_cli
    return os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, cfg.DATA_COLLECTOR_FOLDER)

def main():
    parser = argparse.ArgumentParser(description="Data Collector Tool")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--product", type=str, default=None, help="Override PRODUCT_NAME")
    parser.add_argument("--rtsp", type=str, default=None, help="Override RTSP_URL")
    parser.add_argument("--model", type=str, default=None, help="Override MODEL_PATH")
    parser.add_argument("--out_dir", type=str, default=None, help="Absolute output dir for collector (overrides config)")
    parser.add_argument("--display-scale", type=float, default=None, help="Override DISPLAY_SCALE")
    parser.add_argument("--pad", type=float, default=None, help="Override PAD_RATIO")
    parser.add_argument("--no-flip-vertical", action="store_true", help="Disable vertical flip even if config says true")
    parser.add_argument("--mode", type=str, choices=["origin", "model"], default="model",
                        help="origin: chỉ lưu ảnh gốc; model: dùng YOLO segmentation để cắt lưu đối tượng")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)

    # overrides
    if args.product: cfg.PRODUCT_NAME = args.product
    if args.rtsp:    cfg.RTSP_URL = args.rtsp
    if args.model:   cfg.MODEL_PATH = args.model
    if args.display_scale is not None: cfg.DISPLAY_SCALE = float(args.display_scale)
    if args.pad is not None:           cfg.PAD_RATIO = float(args.pad)
    if args.no_flip_vertical:          cfg.FLIP_VERTICAL = False

    collect_root = resolve_collect_root(cfg, args.out_dir)
    ensure_dirs(collect_root)

    prod_dir = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME)
    ensure_dirs(prod_dir)

    # session folders (KHÔNG còn none_backgrounds)
    session_root, dir_original, dir_crops, dir_masks, dir_bboxes, *_ = make_session_dir(collect_root)
    print(f"[INFO] Session dir: {session_root}")

    detector = None
    if args.mode == "model":
        print("[INFO] Loading YOLO segmentation model...")
        detector = YOLODetector(cfg.MODEL_PATH)

    cam = RTSP_Threaded_Camera(cfg.RTSP_URL, width=cfg.CAM_WIDTH, height=cfg.CAM_HEIGHT)

    capturing = False
    saved_count = 0
    print("[INFO] SPACE=start/stop capture, q=quit]")
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                cv2.waitKey(1)
                continue

            if cfg.FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)

            H, W = frame.shape[:2]

            # HUD
            disp = cv2.resize(frame, None, fx=cfg.DISPLAY_SCALE, fy=cfg.DISPLAY_SCALE)
            status = "CAPTURING" if capturing else "IDLE"
            cv2.putText(disp, f"Status: {status} | Saved: {saved_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(disp, f"Status: {status} | Saved: {saved_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            chx, chy = disp.shape[1]//2, disp.shape[0]//2
            cv2.line(disp, (chx-20, chy), (chx+20, chy), (255,255,255), 1)
            cv2.line(disp, (chx, chy-20), (chx, chy+20), (255,255,255), 1)
            cv2.imshow("preview", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:
                capturing = not capturing
                print(f"[INFO] Capturing: {capturing}")

            if not capturing:
                continue

            name = ts()

            # save original
            orig_path = os.path.join(dir_original, f"IMG_{name}.jpg")
            cv2.imwrite(orig_path, frame)

            if args.mode == "origin":
                saved_count += 1
                continue

            # YOLO seg
            r = detector.predict(frame)
            if r is None:
                continue

            # boxes & classes
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                cls_ids = r.boxes.cls.cpu().numpy().astype(int) if getattr(r.boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=int)
                boxes = xyxy.tolist()
            else:
                xyxy = np.zeros((0,4), dtype=int)
                cls_ids = np.zeros((0,), dtype=int)
                boxes = []

            # save bboxes json
            with open(os.path.join(dir_bboxes, f"IMG_{name}.json"), "w", encoding="utf-8") as f:
                json.dump({"image": os.path.basename(orig_path), "bboxes": boxes, "cls": cls_ids.tolist()},
                          f, ensure_ascii=False, indent=2)

            # masks (resize -> origin size HxW)
            if hasattr(r, "masks") and r.masks is not None and r.masks.data is not None:
                masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)  # [N,hm,wm]
                if masks.ndim == 3 and (masks.shape[1] != H or masks.shape[2] != W):
                    up = np.zeros((masks.shape[0], H, W), dtype=np.uint8)
                    for i in range(masks.shape[0]):
                        up[i] = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
                    masks = up
            else:
                masks = np.zeros((0, H, W), dtype=np.uint8)

            # per object: SQUARE CROP + MASK THEO CROP
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                x1p, y1p, x2p, y2p = pad_and_clip_box(
                    x1, y1, x2, y2, cfg.PAD_RATIO, W, H, square=True
                )

                crop = frame[y1p:y2p, x1p:x2p]
                cv2.imwrite(os.path.join(dir_crops, f"CROP_{name}_{i}.jpg"), crop)

                if masks.shape[0] > i:
                    full_mask = (masks[i] * 255).astype(np.uint8)
                    mask_crop = full_mask[y1p:y2p, x1p:x2p]
                else:
                    mask_crop = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)

                cv2.imwrite(os.path.join(dir_masks, f"MASK_{name}_{i}.png"), mask_crop)

            saved_count += 1

    finally:
        cv2.destroyAllWindows()
        cam.stop()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
