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

    resized = []
    for img in valid_views:
        src_h, src_w = img.shape[:2]
        scale = target_h / max(1, src_h)
        new_w = max(1, int(round(src_w * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized.append(cv2.resize(
            img, (new_w, target_h), interpolation=interpolation
        ))

    grid_rows = []
    for r in range(rows):
        row_imgs = resized[r * cols : (r + 1) * cols]
        if row_imgs:
            grid_rows.append(cv2.hconcat(row_imgs))

    max_row_w = max(row.shape[1] for row in grid_rows)
    for index, row in enumerate(grid_rows):
        pad_right = max_row_w - row.shape[1]
        if pad_right > 0:
            grid_rows[index] = cv2.copyMakeBorder(
                row, 0, 0, 0, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

    return cv2.vconcat(grid_rows)
