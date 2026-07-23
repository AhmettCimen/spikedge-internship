import os
import glob
import json
import shutil
import argparse
import numpy as np
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert RailSem19 dataset to YOLO Segmentation format.")
    parser.add_argument("--archive-dir", type=str, default="archive", help="Path to archive directory")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output dataset directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set split ratio")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return parser.parse_args()

def convert_dataset():
    args = parse_args()
    np.random.seed(args.seed)

    jpg_dir = os.path.join(args.archive_dir, "jpgs", "rs19_val")
    uint8_dir = os.path.join(args.archive_dir, "uint8", "rs19_val")
    json_dir = os.path.join(args.archive_dir, "jsons", "rs19_val")

    if not os.path.exists(jpg_dir):
        raise FileNotFoundError(f"Image directory not found: {jpg_dir}")

    # Gather all frames
    jpg_files = glob.glob(os.path.join(jpg_dir, "*.jpg"))
    jpg_files.sort()

    if args.limit > 0:
        jpg_files = jpg_files[:args.limit]

    total_files = len(jpg_files)
    print(f"Found {total_files} images to process.")

    # Shuffle and split train / val
    indices = np.arange(total_files)
    np.random.shuffle(indices)
    val_count = int(total_files * args.val_split)
    val_indices = set(indices[:val_count])

    # Output directories
    out_img_train = os.path.join(args.output_dir, "images", "train")
    out_img_val = os.path.join(args.output_dir, "images", "val")
    out_lbl_train = os.path.join(args.output_dir, "labels", "train")
    out_lbl_val = os.path.join(args.output_dir, "labels", "val")

    for d in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
        os.makedirs(d, exist_ok=True)

    stats = {"rail_track_count": 0, "rail_line_count": 0, "processed_frames": 0}

    for idx, jpg_path in enumerate(tqdm(jpg_files, desc="Converting RailSem19")):
        frame_id = os.path.splitext(os.path.basename(jpg_path))[0]
        png_path = os.path.join(uint8_dir, f"{frame_id}.png")
        json_path = os.path.join(json_dir, f"{frame_id}.json")

        is_val = idx in val_indices
        target_img_dir = out_img_val if is_val else out_img_train
        target_lbl_dir = out_lbl_val if is_val else out_lbl_train

        # Copy image file
        dst_img_path = os.path.join(target_img_dir, f"{frame_id}.jpg")
        if not os.path.exists(dst_img_path):
            shutil.copy2(jpg_path, dst_img_path)

        lbl_lines = []

        # Read uint8 label map
        if os.path.exists(png_path):
            mask_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                h, w = mask_img.shape

                # 1) Class 0: rail-track (label ID == 12)
                track_mask = (mask_img == 12).astype(np.uint8) * 255
                if np.any(track_mask):
                    contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 150:
                            continue
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        if len(approx) >= 3:
                            pts = approx.reshape(-1, 2)
                            norm_pts = []
                            for px, py in pts:
                                norm_pts.extend([round(px / w, 6), round(py / h, 6)])
                            line_str = "0 " + " ".join(map(str, norm_pts))
                            lbl_lines.append(line_str)
                            stats["rail_track_count"] += 1

                # 2) Class 1: rail-line from mask (label ID == 17 or 18)
                rail_mask = ((mask_img == 17) | (mask_img == 18)).astype(np.uint8) * 255
                
                # Also overlay polylines from JSON onto rail_mask if available
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            jdata = json.load(f)
                        for obj in jdata.get("objects", []):
                            if "polyline-pair" in obj:
                                for rail_pts in obj["polyline-pair"]:
                                    pts_arr = np.around(np.array(rail_pts)).astype(np.int32)
                                    if len(pts_arr) > 1:
                                        cv2.polylines(rail_mask, [pts_arr], False, 255, thickness=6)
                            elif "polyline" in obj and obj.get("label") in ["rail", "guard-rail"]:
                                pts_arr = np.around(np.array(obj["polyline"])).astype(np.int32)
                                if len(pts_arr) > 1:
                                    cv2.polylines(rail_mask, [pts_arr], False, 255, thickness=6)
                    except Exception:
                        pass

                if np.any(rail_mask):
                    contours, _ = cv2.findContours(rail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 20:
                            continue
                        epsilon = 0.002 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        if len(approx) >= 3:
                            pts = approx.reshape(-1, 2)
                            norm_pts = []
                            for px, py in pts:
                                norm_pts.extend([round(px / w, 6), round(py / h, 6)])
                            line_str = "1 " + " ".join(map(str, norm_pts))
                            lbl_lines.append(line_str)
                            stats["rail_line_count"] += 1

        # Write label txt
        lbl_txt_path = os.path.join(target_lbl_dir, f"{frame_id}.txt")
        with open(lbl_txt_path, "w") as f:
            f.write("\n".join(lbl_lines))

        stats["processed_frames"] += 1

    # Write dataset.yaml
    yaml_content = f"""path: {os.path.abspath(args.output_dir).replace('\\', '/')}
train: images/train
val: images/val

names:
  0: rail-track
  1: rail-line
"""
    yaml_path = os.path.join(args.output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n--- Dataset Conversion Summary ---")
    print(f"Processed Frames    : {stats['processed_frames']}")
    print(f"Rail Track Polygons : {stats['rail_track_count']}")
    print(f"Rail Line Polygons  : {stats['rail_line_count']}")
    print(f"Dataset Config Saved: {yaml_path}")

if __name__ == "__main__":
    convert_dataset()
