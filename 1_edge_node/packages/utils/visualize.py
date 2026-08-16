import cv2
import math
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
"""
16082026 - KHAI - Arrange cropped views as grids
"""
def concat_anomaly_crops(anomaly_crop_views: List[np.ndarray], target_h: int = 300) -> np.ndarray:
    if len(anomaly_crop_views) == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)

    valid_views = [img for img in anomaly_crop_views if img is not None and img.size > 0]
    num_views = len(valid_views)
    if num_views == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)

    if num_views <= 3:
        cols = num_views
        rows = 1
    else:
        cols = math.ceil(math.sqrt(num_views))
        rows = math.ceil(num_views / cols)

    avg_aspect_ratio = sum(img.shape[1] / max(1, img.shape[0]) for img in valid_views) / num_views
    target_w = max(1, int(target_h * avg_aspect_ratio))

    resized = [cv2.resize(img, (target_w, target_h)) for img in valid_views]

    total_slots = rows * cols
    if len(resized) < total_slots:
        pad_count = total_slots - len(resized)
        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized.extend([blank] * pad_count)

    grid_rows = []
    for r in range(rows):
        row_imgs = resized[r * cols : (r + 1) * cols]
        grid_rows.append(cv2.hconcat(row_imgs))

    return cv2.vconcat(grid_rows)
