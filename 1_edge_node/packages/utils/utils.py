import os
from datetime import datetime
import cv2
import numpy as np
import torch
import math

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def pad_and_clip_box(
    x1: int, y1: int, x2: int, y2: int,
    pad_ratio: float, W: int, H: int,
    square: bool = False
):
    """
    Pad bbox theo tỷ lệ rồi clip trong ảnh. Nếu square=True, tạo bbox VUÔNG
    với tâm là tâm bbox đã pad, cố gắng giữ nguyên kích thước (cạnh = max(w,h) sau pad),
    và dịch để nằm gọn trong khung ảnh nếu bị vướng biên.

    Ghi chú:
      - Toạ độ trả về theo convention cũ (x2,y2 có thể == W-1/H-1). Khi dùng Numpy
        slice frame[y1:y2, x1:x2] thì biên phải/dưới là exclusive (tiêu chuẩn trước đây của bạn).
    """
    # bước 1: pad như cũ
    w = x2 - x1
    h = y2 - y1
    px = int(round(w * pad_ratio))
    py = int(round(h * pad_ratio))
    nx1 = clamp(x1 - px, 0, W - 1)
    ny1 = clamp(y1 - py, 0, H - 1)
    nx2 = clamp(x2 + px, 0, W - 1)
    ny2 = clamp(y2 + py, 0, H - 1)

    if not square:
        return nx1, ny1, nx2, ny2

    # bước 2: lấy tâm sau pad
    pw = max(1, nx2 - nx1)
    ph = max(1, ny2 - ny1)
    cx = (nx1 + nx2) / 2.0
    cy = (ny1 + ny2) / 2.0

    # bước 3: kích thước vuông mong muốn
    side = int(round(max(pw, ph)))
    side = max(1, min(side, W, H))  # không vượt ảnh
    half = side / 2.0

    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = sx1 + side
    sy2 = sy1 + side

    # bước 4: nếu vượt biên, dịch lại để nằm gọn trong ảnh (giữ side)
    shift_x = 0
    shift_y = 0
    if sx1 < 0:
        shift_x = -sx1
    elif sx2 > W:
        shift_x = W - sx2
    if sy1 < 0:
        shift_y = -sy1
    elif sy2 > H:
        shift_y = H - sy2
    sx1 += shift_x
    sx2 += shift_x
    sy1 += shift_y
    sy2 += shift_y

    # bước 5: clip lần cuối
    sx1 = clamp(sx1, 0, W - 1)
    sy1 = clamp(sy1, 0, H - 1)
    sx2 = clamp(sx2, 0, W - 1)
    sy2 = clamp(sy2, 0, H - 1)

    # đảm bảo có diện tích dương
    if sx2 <= sx1:
        sx2 = min(W - 1, sx1 + 1)
    if sy2 <= sy1:
        sy2 = min(H - 1, sy1 + 1)

    return sx1, sy1, sx2, sy2


def need_clean(obj):
    """Kiểm tra xem có cần clean không (có tuple hoặc numpy)"""
    if isinstance(obj, dict):
        return any(need_clean(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(need_clean(v) for v in obj)
    elif isinstance(obj, (np.generic, np.ndarray, tuple, torch.Tensor)):
        return True
    return False

def clean_data(obj):
    """Đệ quy convert numpy -> python type và tuple -> list"""
    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(v) for v in obj]
    elif isinstance(obj, tuple):
        return [clean_data(v) for v in obj]   # tuple -> list
    elif isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):  
        if obj.ndim == 0:  # scalar tensor
            return obj.item()
        else:              # multi-dim tensor
            return obj.tolist()
    else:
        return obj
    
def arrow_angle(red, yellow):
    x1, y1 = red
    x2, y2 = yellow
    dx, dy = x2 - x1, y2 - y1
    theta = math.atan2(dy, dx)   # CCW, rad
    deg = math.degrees(theta)    # [-180,180]
    angle = (360 - deg) % 360    # đổi sang CW
    return angle

def rotate_image(frame, angle):
    """Xoay toàn bộ ảnh quanh tâm"""
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    return rotated, M

def transform_points(points, M):
    """Biến đổi toạ độ keypoints/boxes theo ma trận xoay"""
    pts = np.array(points).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    pts_ones = np.hstack([pts, ones])
    transformed = M.dot(pts_ones.T).T
    return transformed

"""
16082026 - KHAI - Add function to convert bbox in cropped images to coordinates in original image
"""
def transform_bbox_to_original_coords(bbox, rotation_matrix, crop_origin=(0, 0), frame_shape=None):
    """Map a bbox from the rotated crop back to original image coordinates."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ox, oy = crop_origin
    pts = np.array([
        [x1 + ox, y1 + oy],
        [x2 + ox, y1 + oy],
        [x2 + ox, y2 + oy],
        [x1 + ox, y2 + oy],
    ], dtype=np.float32).reshape(-1, 1, 2)

    inv_M = cv2.invertAffineTransform(rotation_matrix)
    transformed = cv2.transform(pts, inv_M).reshape(-1, 2)

    x_min = int(np.floor(transformed[:, 0].min()))
    y_min = int(np.floor(transformed[:, 1].min()))
    x_max = int(np.ceil(transformed[:, 0].max()))
    y_max = int(np.ceil(transformed[:, 1].max()))

    if frame_shape is not None:
        h, w = frame_shape[:2]
        x_min = max(0, min(w - 1, x_min))
        y_min = max(0, min(h - 1, y_min))
        x_max = max(0, min(w - 1, x_max))
        y_max = max(0, min(h - 1, y_max))

    return [x_min, y_min, x_max, y_max]


def union_box(box1, box2):
    """Lấy box chung từ 2 box"""
    x1 = min(int(box1[0]), int(box2[0]))
    y1 = min(int(box1[1]), int(box2[1]))
    x2 = max(int(box1[2]), int(box2[2]))
    y2 = max(int(box1[3]), int(box2[3]))
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    xA = max(int(box1[0]), int(box2[0]))
    yA = max(int(box1[1]), int(box2[1]))
    xB = min(int(box1[2]), int(box2[2]))
    yB = min(int(box1[3]), int(box2[3]))

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou
