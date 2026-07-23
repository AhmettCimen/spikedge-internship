import argparse
import os
import torch
from ultralytics import YOLO  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Segmentation model on RailSem19 dataset.")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8s-seg.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda', 'cpu', '0', etc.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for Windows compatibility)")
    parser.add_argument("--project", type=str, default="runs/segment", help="Save project directory")
    parser.add_argument("--name", type=str, default="rail_model", help="Experiment name")
    return parser.parse_args()

def train():
    args = parse_args()

    # Determine device
    if not args.device:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"=== Starting YOLO Segmentation Training ===")
    print(f"Model       : {args.model}")
    print(f"Dataset Config: {args.data}")
    print(f"Epochs      : {args.epochs}")
    print(f"Image Size  : {args.imgsz}")
    print(f"Batch Size  : {args.batch}")
    print(f"Workers     : {args.workers}")
    print(f"Device      : {device} (CUDA Available: {torch.cuda.is_available()})")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True
    )

    print(f"\nTraining completed! Best weights saved to: {os.path.join(args.project, args.name, 'weights', 'best.pt')}")

if __name__ == "__main__":
    train()
