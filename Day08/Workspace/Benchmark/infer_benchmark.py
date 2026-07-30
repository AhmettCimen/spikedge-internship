"""
infer_benchmark_d8.py — Day 8 & Day 9 GPU-Accelerated Benchmark & Class-wise IoU Evaluator
=============================================================================================
- Supports PyTorch (.pt), ONNX (.onnx), PyTorch SMP (.pth), and TFLite (.tflite) models.
- Uses NVIDIA CUDA GPU acceleration for maximum FPS and zero MemoryError allocation crashes.
- Measures real-time Pure GPU Inference FPS/ms and System Display Loop FPS/ms.
- Displays live HUD overlay during video playback.
- Evaluates class-wise IoU scores per detected class.
- Saves versioned logs to Outputs/Logs/[model_name].log
- Appends benchmark results to W/D9/leaderboard.csv
"""

import os
import sys

# Ensure current script directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import time
import csv
import argparse
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Local import
from evaluator import compute_binary_iou, format_metrics_report

# Enable cuDNN benchmark for kernel optimization if CUDA is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def select_weights_via_gui(initial_dir=r"C:/Users/kalra/OneDrive/Desktop/Staj/W/D6",
                           title="Select Model Weights File (.pt / .onnx / .pth / .tflite)"):
    if not os.path.exists(initial_dir):
        initial_dir = os.path.dirname(__file__)

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir,
        filetypes=[
            ("All Model Weights (*.pt;*.onnx;*.pth;*.tflite)", "*.pt;*.onnx;*.pth;*.tflite"),
            ("YOLO PyTorch Weights (*.pt)", "*.pt"),
            ("ONNX Models (*.onnx)", "*.onnx"),
            ("PyTorch SMP Weights (*.pth)", "*.pth"),
            ("TensorFlow Lite Weights (*.tflite)", "*.tflite"),
            ("All Files (*.*)", "*.*")
        ]
    )
    root.destroy()
    return file_path


def select_video_via_gui(initial_dir=r"C:/Users/kalra/OneDrive/Desktop/Staj/src",
                         title="Select Video File"):
    if not os.path.exists(initial_dir):
        initial_dir = os.path.dirname(__file__)

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir,
        filetypes=[
            ("Video Files (*.mp4;*.avi;*.mov;*.mkv)", "*.mp4;*.avi;*.mov;*.mkv"),
            ("All Files (*.*)", "*.*")
        ]
    )
    root.destroy()
    return file_path


class D8GPUPerformantDetector:
    """
    Day 8 / Day 9 High-Performance GPU Detector supporting PyTorch, ONNX, and TFLite.
    """
    def __init__(self, 
                 weights_path: str, 
                 arch: str = "deeplabv3plus", 
                 encoder: str = "resnet50", 
                 img_size: int = 640,
                 threshold: float = 0.25,
                 device: str = "cuda"):
        
        self.weights_path = weights_path
        self.img_size = (img_size, img_size)
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        device_label = "NVIDIA GPU (CUDA)" if self.device.type == "cuda" else "CPU"
        
        ext = os.path.splitext(weights_path)[1].lower()
        stem = Path(weights_path).stem
        
        # Handle TFLite memory limit fallback to matching PT/ONNX file if needed
        if ext == ".tflite":
            pt_candidate = os.path.join(os.path.dirname(os.path.dirname(weights_path)), f"{stem.replace('_float32', '').replace('_float16', '')}.pt")
            onnx_candidate = os.path.join(os.path.dirname(os.path.dirname(weights_path)), f"{stem.replace('_float32', '').replace('_float16', '')}.onnx")
            
            if os.path.exists(pt_candidate):
                weights_path = pt_candidate
                ext = ".pt"
                print(f"[INFO] TFLite Memory Limit Bypass: Routing to PyTorch GPU model ({weights_path})")
            elif os.path.exists(onnx_candidate):
                weights_path = onnx_candidate
                ext = ".onnx"
                print(f"[INFO] TFLite Memory Limit Bypass: Routing to ONNX GPU model ({weights_path})")

        self.is_yolo = ext in [".pt", ".onnx", ".tflite"] or "yolo" in weights_path.lower()
        
        if self.is_yolo:
            from ultralytics import YOLO
            from ultralytics.utils.plotting import colors
            
            # Set custom palette colors so trackbed (class 2 & 3) is brown (139, 69, 19)
            colors.palette[2] = (139, 69, 19)
            colors.palette[3] = (139, 69, 19)
            
            self.model_name = Path(weights_path).stem
            print("\n" + "=" * 60)
            print(f"[INFO] Initializing YOLOv8 GPU Segmentation Detector ({ext.upper()})")
            print(f"  - Device           : {device_label}")
            print(f"  - Model Name       : {self.model_name}")
            print(f"  - Input Resolution : {self.img_size[1]}x{self.img_size[0]}")
            print(f"  - Weights File     : {weights_path}")
            print("=" * 60)

            self.yolo_model = YOLO(weights_path)
            self.class_names = getattr(self.yolo_model, "names", {0: "rail-track", 1: "rail-raised", 2: "trackbed"})
            
            if self.device.type == "cuda":
                print("[INFO] Running GPU Warmup...")
                dummy = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                for _ in range(5):
                    _ = self.yolo_model.predict(dummy, imgsz=self.img_size[0], conf=self.threshold, device=self.device, verbose=False)
                torch.cuda.synchronize()
                print("[SUCCESS] NVIDIA GPU Ready.\n")
        else:
            # PyTorch SMP Model
            import importlib
            models_module = importlib.import_module("models")
            create_model = getattr(models_module, "create_model")
            
            self.model_name = f"{arch}+{encoder}"
            self.class_names = {1: "rail-track"}
            print("\n" + "=" * 60)
            print("[INFO] Initializing PyTorch SMP GPU Detector")
            print(f"  - Device           : {device_label}")
            print(f"  - Architecture     : {arch} ({encoder})")
            print(f"  - Input Resolution : {self.img_size[1]}x{self.img_size[0]}")
            print(f"  - Weights File     : {weights_path}")
            print("=" * 60)

            self.model = create_model(architecture=arch, encoder=encoder, in_channels=3, classes=1)
            self.model.to(self.device)
            self.model.eval()

            if os.path.exists(weights_path):
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["state_dict"])
                    elif isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint)
                    else:
                        self.model.load_state_dict(checkpoint)
                    print("[SUCCESS] Model weights loaded successfully.")
                except Exception as e:
                    print(f"[WARNING] Failed to load model weights: {e}")
            else:
                print(f"[WARNING] Model weights file not found: {weights_path}")

            if self.device.type == "cuda":
                print("[INFO] Running GPU Warmup...")
                dummy = torch.zeros((1, 3, self.img_size[0], self.img_size[1]), device=self.device)
                with torch.amp.autocast('cuda'):
                    for _ in range(10):
                        _ = self.model(dummy)
                torch.cuda.synchronize()
                print("[SUCCESS] NVIDIA GPU Ready.\n")

    def process_frame_gpu(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Executes GPU inference.
        Returns:
            vis_bgr: Processed BGR visualization
            pred_info: Information dictionary containing masks/classes
        """
        if self.is_yolo:
            results = self.yolo_model.predict(
                frame_bgr, 
                imgsz=self.img_size[0], 
                conf=self.threshold, 
                device=self.device, 
                verbose=False
            )
            res = results[0]
            vis_bgr = res.plot(boxes=False)
            return vis_bgr, {"res": res}
        else:
            h_orig, w_orig = frame_bgr.shape[:2]
            t_bgr = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
            t_rgb = t_bgr[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t_input = torch.nn.functional.interpolate(t_rgb, size=self.img_size, mode='bilinear', align_corners=False)

            if self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = self.model(t_input)
                    prob = torch.sigmoid(logits)
            else:
                logits = self.model(t_input)
                prob = torch.sigmoid(logits)

            prob_orig = torch.nn.functional.interpolate(prob, size=(h_orig, w_orig), mode='bilinear', align_corners=False).squeeze()
            binary_mask_gpu = (prob_orig > self.threshold).to(torch.uint8)
            binary_mask = binary_mask_gpu.cpu().numpy()

            vis_bgr = frame_bgr.copy()
            mask_indices = binary_mask > 0

            if np.any(mask_indices):
                vis_bgr[mask_indices] = cv2.addWeighted(vis_bgr[mask_indices], 0.75, 
                                                        np.full_like(vis_bgr[mask_indices], (150, 255, 0)), 0.25, 0)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_bgr, contours, -1, (0, 255, 255), 2)

            return vis_bgr, {"binary_mask": binary_mask}


def save_unique_log(model_name: str, content: str, log_dir=r"C:/Users/kalra/OneDrive/Desktop/Staj/Outputs/Logs"):
    os.makedirs(log_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    log_path = os.path.join(log_dir, f"[{safe_name}].log")
    
    counter = 1
    while os.path.exists(log_path):
        log_path = os.path.join(log_dir, f"[{safe_name}]_{counter}.log")
        counter += 1
        
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
        
    return log_path


def append_to_leaderboard(model_name: str, gpu_fps: float, gpu_lat: float, sys_fps: float, sys_lat: float,
                          leaderboard_path=r"C:/Users/kalra/OneDrive/Desktop/Staj/W/D9/leaderboard.csv"):
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    file_exists = os.path.exists(leaderboard_path)
    
    fieldnames = ["Timestamp", "Model_Name", "GPU_FPS", "GPU_Latency_ms", "System_FPS", "System_Latency_ms"]
    
    with open(leaderboard_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(leaderboard_path) == 0:
            writer.writeheader()
        writer.writerow({
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Model_Name": model_name,
            "GPU_FPS": round(gpu_fps, 2),
            "GPU_Latency_ms": round(gpu_lat, 2),
            "System_FPS": round(sys_fps, 2),
            "System_Latency_ms": round(sys_lat, 2)
        })
    print(f"[LEADERBOARD UPDATED] -> {leaderboard_path}")


def run_d8_benchmark(video_path: str, detector: D8GPUPerformantDetector, show_display: bool = True):
    model_name = detector.model_name

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = "D9 GPU Benchmark & Live Evaluator"
    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1100, 620)

    frame_idx = 0
    paused = False

    gpu_latencies_ms = []
    system_latencies_ms = []
    
    t_start = time.time()
    prev_loop_time = time.time()

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 1. Pure GPU Model Inference Timing
            if detector.device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu0 = time.time()

            vis, pred_info = detector.process_frame_gpu(frame)

            if detector.device.type == "cuda":
                torch.cuda.synchronize()
            gpu_time_ms = (time.time() - t_gpu0) * 1000.0
            gpu_latencies_ms.append(gpu_time_ms)
            gpu_instant_fps = 1000.0 / max(gpu_time_ms, 1e-5)

            # 2. System End-to-End Display Timing
            curr_loop_time = time.time()
            system_time_sec = curr_loop_time - prev_loop_time
            prev_loop_time = curr_loop_time
            system_time_ms = system_time_sec * 1000.0
            system_instant_fps = 1.0 / max(system_time_sec, 1e-6)
            system_latencies_ms.append(system_time_ms)

            # HUD Overlay
            if show_display:
                hud = vis.copy()
                cv2.rectangle(hud, (10, 10), (340, 90), (0, 0, 0), -1)
                vis = cv2.addWeighted(vis, 0.75, hud, 0.25, 0)

                cv2.putText(vis, f"GPU Infer: {gpu_instant_fps:.1f} FPS ({gpu_time_ms:.1f} ms)", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(vis, f"Display Loop: {system_instant_fps:.1f} FPS ({system_time_ms:.1f} ms)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        if show_display:
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # Space
                paused = not paused
            elif key == ord('d') or key == 83:
                skip = int(5.0 * orig_fps)
                new_pos = min(total_frames - 1, frame_idx + skip)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            elif key == ord('a') or key == 81:
                skip = int(5.0 * orig_fps)
                new_pos = max(0, frame_idx - skip)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    if len(gpu_latencies_ms) > 0:
        gpu_lats = np.array(gpu_latencies_ms)
        gpu_fps_vals = 1000.0 / np.maximum(gpu_lats, 1e-5)
        gpu_mean_fps, gpu_std_fps = float(np.mean(gpu_fps_vals)), float(np.std(gpu_fps_vals))
        gpu_mean_lat, gpu_std_lat = float(np.mean(gpu_lats)), float(np.std(gpu_lats))

        sys_lats = np.array(system_latencies_ms)
        sys_fps_vals = 1000.0 / np.maximum(sys_lats, 1e-5)
        sys_mean_fps, sys_std_fps = float(np.mean(sys_fps_vals)), float(np.std(sys_fps_vals))
        sys_mean_lat, sys_std_lat = float(np.mean(sys_lats)), float(np.std(sys_lats))

        # Evaluate Class-wise IoU over D8 test dataset
        from evaluator import evaluate_test_dataset_iou
        class_ious = evaluate_test_dataset_iou(detector)

        report_text = format_metrics_report(
            model_name=model_name,
            gpu_fps_mean=gpu_mean_fps, gpu_fps_std=gpu_std_fps,
            gpu_lat_mean=gpu_mean_lat, gpu_lat_std=gpu_std_lat,
            sys_fps_mean=sys_mean_fps, sys_fps_std=sys_std_fps,
            sys_lat_mean=sys_mean_lat, sys_lat_std=sys_std_lat,
            class_ious=class_ious
        )

        print("\n" + report_text)

        log_saved = save_unique_log(model_name, report_text)
        print(f"\n[LOG SAVED] -> {log_saved}")
        
        append_to_leaderboard(model_name, gpu_mean_fps, gpu_mean_lat, sys_mean_fps, sys_mean_lat)


def main():
    parser = argparse.ArgumentParser(description="Day 8 & Day 9 GPU Benchmark & Evaluator")
    parser.add_argument("--weights", type=str, default=None, help="Model weights file (.pt, .onnx, .pth, or .tflite)")
    parser.add_argument("--file", type=str, default=None, help="Input video file path")
    parser.add_argument("--img-size", type=int, default=640, help="GPU input resolution (e.g. 640, 768)")
    parser.add_argument("--arch", type=str, default="deeplabv3plus", help="Model architecture (if SMP)")
    parser.add_argument("--encoder", type=str, default="resnet50", help="Model encoder (if SMP)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Segmentation threshold / conf")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")

    args = parser.parse_args()

    selected_weights = args.weights
    if selected_weights is None:
        print("Please select model weights file (.pt / .onnx / .pth / .tflite)...")
        selected_weights = select_weights_via_gui()
        if not selected_weights:
            print("No weights file selected. Exiting.")
            return

    selected_file = args.file
    if selected_file is None:
        print("Please select a video file...")
        selected_file = select_video_via_gui()
        if not selected_file:
            print("No video file selected. Exiting.")
            return

    detector = D8GPUPerformantDetector(
        weights_path=selected_weights,
        arch=args.arch,
        encoder=args.encoder,
        img_size=args.img_size,
        threshold=args.threshold,
        device=args.device
    )

    run_d8_benchmark(selected_file, detector, show_display=not args.no_display)


if __name__ == "__main__":
    main()
