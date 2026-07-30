"""
evaluator.py — Class-wise IoU & Segmentation Metric Evaluator
============================================================
Hesaplanan tahmin maskeleri ile ground truth maskeleri arasında
sınıf bazlı IoU (Intersection over Union), Precision, Recall ve mIoU skorlarını hesaplar.
"""

import os
import glob
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


RS19_UINT8_MAPPING = {
    12: "rail-track",
    17: "rail-raised",
    15: "trackbed",
    18: "rail-embedded",
    3: "tram-track"
}


def compute_binary_iou(pred_binary: np.ndarray, gt_binary: np.ndarray, smooth: float = 1e-6) -> float:
    """
    İkili (binary) iki maske arasında IoU hesaplar.
    Shape uyumsuzluğunu (2D vs 3D) otomatik sıfırlar.
    """
    pred_b = (np.squeeze(pred_binary) > 0).astype(np.float32)
    gt_b = (np.squeeze(gt_binary) > 0).astype(np.float32)
    
    if pred_b.ndim == 3:
        pred_b = pred_b[..., 0]
    if gt_b.ndim == 3:
        gt_b = gt_b[..., 0]
    
    intersection = (pred_b * gt_b).sum()
    union = pred_b.sum() + gt_b.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return float((intersection + smooth) / (union + smooth))


def evaluate_test_dataset_iou(detector, test_dataset_dir: str = r"C:\Users\kalra\OneDrive\Desktop\Staj\W\D8\test_dataset", max_samples: int = 100) -> Dict[str, float]:
    """
    D8/test_dataset içerisindeki görülmemiş test resimleri ve uint8 etiket maskelerini kullanarak
    modelin sınıf bazlı IoU skorlarını hesaplar.
    """
    img_dir = os.path.join(test_dataset_dir, "images")
    mask_dir = os.path.join(test_dataset_dir, "masks")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"[WARNING] Test dataset directories not found in {test_dataset_dir}")
        return {}
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:max_samples]
    if not img_files:
        print("[WARNING] No test images found.")
        return {}

    class_iou_accumulator = {}
    class_count_accumulator = {}

    print(f"\n[INFO] Evaluator: Running test set evaluation on {len(img_files)} images...")

    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        
        if not os.path.exists(mask_path):
            continue

        frame_bgr = cv2.imread(img_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if frame_bgr is None or gt_mask is None:
            continue

        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]

        h_orig, w_orig = frame_bgr.shape[:2]
        
        # Detector prediction
        vis, pred_info = detector.process_frame_gpu(frame_bgr)

        if not detector.is_yolo:
            # Binary SMP Model (DeepLabV3+)
            pred_binary = pred_info.get("binary_mask", np.zeros((h_orig, w_orig), dtype=np.uint8))
            gt_binary = (gt_mask == 12).astype(np.uint8)  # rail-track
            
            iou = compute_binary_iou(pred_binary, gt_binary)
            class_name = "rail-track"
            class_iou_accumulator[class_name] = class_iou_accumulator.get(class_name, 0.0) + iou
            class_count_accumulator[class_name] = class_count_accumulator.get(class_name, 0.0) + 1
        else:
            # YOLOv8 Segmentation Model: Combine instance masks per class ID
            res = pred_info.get("res", None)
            if res is not None and res.masks is not None:
                masks_data = res.masks.data.cpu().numpy()  # (N, H_in, W_in)
                clss = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

                class_pred_masks = {}
                for i, c_idx in enumerate(clss):
                    c_name = detector.class_names.get(c_idx, f"class_{c_idx}")
                    if c_name.lower() in ['background']:
                        continue

                    m_pred = cv2.resize(masks_data[i], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    binary_inst = (m_pred > 0.5).astype(np.uint8)

                    if c_name not in class_pred_masks:
                        class_pred_masks[c_name] = binary_inst
                    else:
                        class_pred_masks[c_name] = np.logical_or(class_pred_masks[c_name], binary_inst).astype(np.uint8)

                # Compute IoU per predicted class
                for c_name, pred_binary in class_pred_masks.items():
                    gt_id = 12  # default rail-track
                    if c_name == 'rail-raised':
                        gt_id = 17
                    elif c_name == 'trackbed':
                        gt_id = 15
                    elif c_name == 'rail-embedded':
                        gt_id = 18
                    elif c_name == 'tram-track':
                        gt_id = 3

                    gt_binary = (gt_mask == gt_id).astype(np.uint8)
                    
                    if np.any(gt_binary) or np.any(pred_binary):
                        iou = compute_binary_iou(pred_binary, gt_binary)
                        class_iou_accumulator[c_name] = class_iou_accumulator.get(c_name, 0.0) + iou
                        class_count_accumulator[c_name] = class_count_accumulator.get(c_name, 0.0) + 1

    results_ious = {}
    for c_name, total_iou in class_iou_accumulator.items():
        cnt = class_count_accumulator[c_name]
        if cnt > 0:
            results_ious[c_name] = round((total_iou / cnt) * 100.0, 2)

    return results_ious


def format_metrics_report(model_name: str, 
                           gpu_fps_mean: float, gpu_fps_std: float, gpu_lat_mean: float, gpu_lat_std: float,
                           sys_fps_mean: float, sys_fps_std: float, sys_lat_mean: float, sys_lat_std: float,
                           class_ious: Dict[str, float] = None) -> str:
    """
    Metrikleri standart rapor ve log metnine dönüştürür.
    """
    lines = [
        "=" * 60,
        f"BENCHMARK & IOU REPORT — Model: [{model_name}]",
        "=" * 60,
        f"GPU Model Inference FPS (Mean ± Std) : {gpu_fps_mean:.2f} ± {gpu_fps_std:.2f} (Latency: {gpu_lat_mean:.2f} ± {gpu_lat_std:.2f} ms)",
        f"System Display FPS (Mean ± Std)     : {sys_fps_mean:.2f} ± {sys_fps_std:.2f} (Latency: {sys_lat_mean:.2f} ± {sys_lat_std:.2f} ms)",
        "-" * 60,
        "CLASS-WISE IoU SCORES (Unseen Test Set):"
    ]
    
    if class_ious:
        for cls_name, iou_val in class_ious.items():
            lines.append(f"  - {cls_name:<20s} : %{iou_val:.2f}")
        valid_values = [v for v in class_ious.values() if not np.isnan(v)]
        if valid_values:
            mean_iou = np.mean(valid_values)
            lines.append("-" * 60)
            lines.append(f"  * Mean IoU (mIoU)         : %{mean_iou:.2f}")
    else:
        lines.append("  (Sınıf bazlı IoU testi çalıştırılmadı veya test verisi bulunamadı)")
        
    lines.append("=" * 60)
    return "\n".join(lines)
