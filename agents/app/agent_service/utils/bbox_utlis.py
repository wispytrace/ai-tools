
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2

def resize_bbox(bbox: BBox, scale: float) -> BBox:
    """围绕中心缩放 bbox"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    new_w = w * scale
    new_h = h * scale
    return [
        int(cx - new_w / 2),
        int(cy - new_h / 2),
        int(cx + new_w / 2),
        int(cy + new_h / 2)
    ]

def count_contact_pixels(img: np.ndarray, bbox: BBox, band: int = 2) -> int:
    """计算 bbox 边界附近与前景接触的像素数"""
    if img.size == 0:
        raise ValueError("Empty image")
    if band < 0:
        raise ValueError("band must be >= 0")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        g_min, g_max = g.min(), g.max()
        gray = ((g - g_min) / (g_max - g_min + 1e-8) * 255).astype(np.uint8)

    _, inv_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg = inv_bin > 0
    H, W = fg.shape
    x0, y0, x1, y1 = map(int, bbox)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W - 1))
    y0 = max(0, min(y0, H - 1))
    y1 = max(0, min(y1, H - 1))
    if x0 >= x1 or y0 >= y1:
        return 0

    contact_region = np.zeros_like(fg, dtype=bool)

    # 上下边带
    if y1 > y0:
        r0 = max(0, y0 - band)
        r1 = min(H - 1, y0 + band)
        contact_region[r0:r1 + 1, x0:x1 + 1] = True
        r0 = max(0, y1 - band)
        r1 = min(H - 1, y1 + band)
        contact_region[r0:r1 + 1, x0:x1 + 1] = True

    # 左右边带
    if x1 > x0:
        c0 = max(0, x0 - band)
        c1 = min(W - 1, x0 + band)
        contact_region[y0:y1 + 1, c0:c1 + 1] = True
        c0 = max(0, x1 - band)
        c1 = min(W - 1, x1 + band)
        contact_region[y0:y1 + 1, c0:c1 + 1] = True

    return int(np.sum(fg & contact_region))

def find_optimal_bbox(image_path: str, bbox: BBox, scale_list: List[float]) -> BBox:
    """寻找边缘接触最少的最优 bbox（去噪）"""
    img = cv2.imread(image_path)
    optimal = bbox
    min_contact = float('inf')
    for scale in scale_list:
        scaled = resize_bbox(bbox, scale)
        contact = count_contact_pixels(img, scaled)
        if contact < min_contact:
            min_contact = contact
            optimal = scaled
    return optimal

def crop_image(image_path: str, bbox: BBox, save_path: str) -> None:
    """裁剪并保存图像"""
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    H, W = img.shape[:2]
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid bbox for cropping")
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(save_path, cropped)
    return save_path, W, H  # width, height

def visualize_bboxes(image_path: str, bboxes: List[BBox], save_path: str) -> None:
    """在图像上绘制 bbox 并保存"""
    img = cv2.imread(image_path)
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite(save_path, img)
    return os.path.abspath(save_path)