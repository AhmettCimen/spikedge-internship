import argparse
import time
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Video veya Resim Dosyası Seçin",
        filetypes=[
            ("Medya Dosyaları", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
            ("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv"),
            ("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
            ("Tüm Dosyalar", "*.*")
        ]
    )
    root.destroy()
    return file_path

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Rail Detection & Track Path Segmentation on Video/Image.")
    parser.add_argument("--weights", type=str, default="runs/segment/runs/segment/rail_model/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--input", type=str, default=None, help="Input video file path, image path, or camera index. If omitted, file dialog opens.")
    parser.add_argument("--output", type=str, default="output_rail_demo.mp4", help="Output video/image path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--show", action="store_true", default=True, help="Display window during processing (default: True)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable display window during processing")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (e.g. 0, 0,1) or cpu (default: 0)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 for all)")
    return parser.parse_args()

def process_frame(frame, model, conf_thresh=0.35, imgsz=640, device="0"):
    results = model.predict(frame, conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
    annotated = frame.copy()
    h, w = frame.shape[:2]

    overlay = annotated.copy()

    track_color = (255, 191, 0)   # Cyan / Light Blue for rail-track path (BGR: 255, 191, 0)
    rail_color = (0, 255, 255)    # Yellow for metal rail lines (BGR: 0, 255, 255)

    if results.masks is not None and len(results.masks) > 0:
        masks = results.masks.data.cpu().numpy() # shape (N, mask_h, mask_w)
        boxes = results.boxes.data.cpu().numpy() # shape (N, 6) -> x1, y1, x2, y2, conf, cls

        for i, box in enumerate(boxes):
            cls_id = int(box[5])
            mask = masks[i]
            # Resize mask to original frame dimensions
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            bin_mask = (mask_resized > 0.5).astype(np.uint8)

            if cls_id == 0:
                # Class 0: rail-track (Fill polygon area)
                overlay[bin_mask == 1] = track_color

            elif cls_id == 1:
                # Class 1: rail-line (Draw highlighted rail curves)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.drawContours(annotated, [cnt], -1, rail_color, thickness=3)

    # Blend track path overlay with semi-transparency (alpha=0.45)
    cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

    # Draw track boundaries
    if results.masks is not None and len(results.masks) > 0:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()
        for i, box in enumerate(boxes):
            if int(box[5]) == 0: # rail-track
                mask = masks[i]
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                bin_mask = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.polylines(annotated, [cnt], isClosed=False, color=(0, 255, 0), thickness=2)

    return annotated

def main():
    args = parse_args()

    if not args.input:
        print("Girdi dosyası/videosu belirtilmedi. Dosya seçim penceresi açılıyor...")
        args.input = select_file()
        if not args.input:
            print("Herhangi bir dosya seçilmedi. İşlem iptal edildi.")
            return

    weights_path = args.weights
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_paths = [
        weights_path,
        os.path.join(script_dir, weights_path),
        os.path.join(script_dir, "runs", "segment", "runs", "segment", "rail_model", "weights", "best.pt"),
        os.path.join(script_dir, "runs", "segment", "rail_model", "weights", "best.pt"),
        os.path.join(script_dir, "best.pt"),
        "runs/segment/runs/segment/rail_model/weights/best.pt",
        "runs/segment/rail_model/weights/best.pt",
        "best.pt",
    ]
    found_weights = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            found_weights = candidate
            break

    if found_weights:
        weights_path = found_weights
    else:
        print(f"Warning: Custom weights '{weights_path}' not found. Using pretrained 'yolov8s-seg.pt' for demo.")
        weights_path = os.path.join(script_dir, "yolov8s-seg.pt")

    model = YOLO(weights_path)
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print(f"Bilgi: Sisteminizdeki PyTorch kurulumunda CUDA (GPU) sürücüsü aktif/yüklü olmadığı için CPU modunda çalıştırılıyor.")
        device = "cpu"
    print(f"Loaded model from: {weights_path} on device: {device}")

    # Check if input is image file
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img = cv2.imread(args.input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")
        
        t0 = time.time()
        out_img = process_frame(img, model, conf_thresh=args.conf, imgsz=args.imgsz, device=device)
        fps = 1.0 / (time.time() - t0)

        cv2.putText(out_img, f"FPS: {fps:.1f} ({str(device).upper()})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(args.output, out_img)
        print(f"Processed single image in {time.time() - t0:.3f}s. Saved output to: {args.output}")

        if args.show:
            cv2.imshow("Rail & Track Segmentation", out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Input is video or webcam
    input_source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    print(f"Processing video stream: {args.input} ({width}x{height} @ {fps_in:.1f} FPS)...")

    win_name = "Real-Time Rail & Track Segmentation"
    if args.show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames limit: {args.max_frames}")
                break

            t0 = time.time()
            annotated_frame = process_frame(frame, model, conf_thresh=args.conf, imgsz=args.imgsz, device=device)
            proc_time = time.time() - t0
            curr_fps = 1.0 / max(proc_time, 0.001)

            cv2.putText(annotated_frame, f"FPS: {curr_fps:.1f} | Real-Time Track Seg", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            out_writer.write(annotated_frame)
            frame_count += 1

            if args.show:
                cv2.imshow(win_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nKullanıcı tarafından durduruldu ('q' tuşuna basıldı).")
                    break
    finally:
        cap.release()
        out_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / max(total_time, 0.001)
    print(f"\nProcessing complete! Processed {frame_count} frames in {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
