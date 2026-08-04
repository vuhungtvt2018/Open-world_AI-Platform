import cv2
import numpy as np
from typing import List

def draw_preview(frame, display_scale: float, status: str, saved_count: int):
    disp = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
    cv2.putText(disp, f"Status: {status} | Saved: {saved_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(disp, f"Status: {status} | Saved: {saved_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    chx, chy = disp.shape[1]//2, disp.shape[0]//2
    cv2.line(disp, (chx-20, chy), (chx+20, chy), (255,255,255), 1)
    cv2.line(disp, (chx, chy-20), (chx, chy+20), (255,255,255), 1)
    return disp

# ghép các crop đã vẽ bbox

def concat_anomaly_crops(anomaly_crop_views: List[np.ndarray], target_h: int = 300):
    if len(anomaly_crop_views) == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    resized = []
    for imgc in anomaly_crop_views:
        if imgc is None or imgc.size == 0:
            continue
        scale = target_h / max(1, imgc.shape[0])
        new_w = max(1, int(imgc.shape[1] * scale))
        img_resz = cv2.resize(imgc, (new_w, target_h))
        resized.append(img_resz)
    if len(resized) == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    return cv2.hconcat(resized)
