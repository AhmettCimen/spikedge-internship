"""
convert_to_tflite.py — Model Weights to TensorFlow Lite Converter & GUI Tool
==============================================================================
Converts PyTorch YOLOv8 (.pt) and PyTorch SMP DeepLabV3+ (.pth) model weights 
to optimized TensorFlow Lite (.tflite) models.

Features:
- Interactive Graphical User Interface (GUI) when launched without CLI arguments.
- Command-line interface (CLI) for automated script invocation.
- Auto-detects model architecture (.pt vs .pth).
- Cross-platform support (handles Windows Ultralytics LiteRT export restrictions via ONNX2TF fallback).
- Supports FP32, FP16, and INT8 precision quantization.

Usage (GUI):
    python convert_to_tflite.py

Usage (CLI):
    python convert_to_tflite.py --weights best.pt --precision FP32 --output best.tflite
    python convert_to_tflite.py --weights deeplab_resnet50.pth --precision FP16 --arch deeplabv3plus --encoder resnet50
"""

import os
import sys
import argparse
import tempfile
import shutil
import threading
import numpy as np
import torch
from pathlib import Path
from typing import Optional

# Ensure workspace directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import create_model


def convert_yolo_to_tflite(weights_path: str,
                           output_path: str,
                           img_size: int = 640,
                           precision: str = "FP32") -> str:
    """
    Converts YOLOv8 (.pt) model to TensorFlow Lite (.tflite) with ONNX2TF fallback for Windows compatibility.

    Args:
        weights_path: Path to .pt weight file
        output_path: Target path for output .tflite model
        img_size: Input resolution
        precision: Precision mode ('FP32', 'FP16', or 'INT8')

    Returns:
        Path to converted .tflite model
    """
    from ultralytics import YOLO

    precision_upper = precision.upper()
    print("\n" + "=" * 60)
    print(f"[CONVERTER] Loading YOLO Model: {weights_path}")
    print(f"  - Input Resolution : {img_size}x{img_size}")
    print(f"  - Precision Mode   : {precision_upper}")
    print("=" * 60)

    model = YOLO(weights_path)
    half_flag = (precision_upper == "FP16")
    int8_flag = (precision_upper == "INT8")

    # 1. Attempt direct Ultralytics export if not on Windows
    if sys.platform != "win32":
        try:
            print("[CONVERTER] Attempting direct Ultralytics TFLite export...")
            exported_path = model.export(
                format='tflite',
                imgsz=img_size,
                half=half_flag,
                int8=int8_flag,
                verbose=True
            )

            if exported_path and os.path.exists(exported_path):
                out_dir = os.path.dirname(os.path.abspath(output_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                if os.path.abspath(exported_path) != os.path.abspath(output_path):
                    shutil.move(exported_path, output_path)
                print(f"\n[SUCCESS] YOLO TFLite model saved to: {output_path}")
                return output_path
        except Exception as e:
            print(f"[WARNING] Direct Ultralytics TFLite export failed ({e}). Switching to ONNX -> TFLite conversion...")
    else:
        print("[CONVERTER] Windows OS detected. Using high-compatibility ONNX -> TFLite conversion pipeline...")

    # 2. Fallback: Export YOLO to ONNX -> ONNX2TF TFLite
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[CONVERTER] Step 1/2: Exporting YOLO to ONNX format...")
        onnx_exported = model.export(
            format='onnx',
            imgsz=img_size,
            dynamic=False,
            simplify=True,
            verbose=False
        )

        if not onnx_exported or not os.path.exists(onnx_exported):
            raise RuntimeError("YOLO ONNX export failed.")

        print(f"[SUCCESS] YOLO ONNX export complete: {onnx_exported}")

        print(f"[CONVERTER] Step 2/2: Converting YOLO ONNX model to TFLite ({precision_upper})...")
        tflite_result = convert_onnx_to_tflite_onnx2tf(
            onnx_path=onnx_exported,
            output_path=output_path,
            precision=precision_upper,
            img_size=img_size
        )
        return tflite_result


def convert_smp_to_tflite(weights_path: str,
                          output_path: str,
                          arch: str = "deeplabv3plus",
                          encoder: str = "resnet50",
                          img_size: int = 640,
                          precision: str = "FP32") -> str:
    """
    Converts PyTorch SMP (.pth) model to TFLite via ONNX -> TFLite conversion pipeline.

    Args:
        weights_path: Path to .pth weight file
        output_path: Target path for output .tflite model
        arch: Model architecture name (e.g., 'deeplabv3plus')
        encoder: Backbone encoder name (e.g., 'resnet50')
        img_size: Input resolution
        precision: Precision mode ('FP32', 'FP16', or 'INT8')

    Returns:
        Path to converted .tflite model
    """
    precision_upper = precision.upper()
    print("\n" + "=" * 60)
    print(f"[CONVERTER] Initializing PyTorch SMP Model ({arch} + {encoder})")
    print(f"  - Weights File     : {weights_path}")
    print(f"  - Input Resolution : {img_size}x{img_size}")
    print(f"  - Precision Mode   : {precision_upper}")
    print("=" * 60)

    # 1. Build PyTorch model and load weights
    model = create_model(architecture=arch, encoder=encoder, in_channels=3, classes=1)
    model.eval()

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("[SUCCESS] PyTorch state_dict loaded.")
    else:
        print(f"[WARNING] Weight file not found at {weights_path}. Converting untrained architecture...")

    # 2. Export PyTorch model to ONNX format in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        print(f"[CONVERTER] Step 1/2: Exporting PyTorch model to ONNX format...")

        dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=13,
            input_names=['input_rgb'],
            output_names=['output_logits'],
            dynamic_axes=None  # Static shape for optimal TFLite memory allocation
        )
        print(f"[SUCCESS] ONNX export complete: {onnx_path}")

        # 3. Convert ONNX model to TFLite
        print(f"[CONVERTER] Step 2/2: Converting ONNX model to TFLite ({precision_upper})...")
        tflite_result = convert_onnx_to_tflite_onnx2tf(
            onnx_path=onnx_path,
            output_path=output_path,
            precision=precision_upper,
            img_size=img_size
        )
        return tflite_result


def convert_onnx_to_tflite_onnx2tf(onnx_path: str,
                                   output_path: str,
                                   precision: str = "FP32",
                                   img_size: int = 640) -> str:
    """
    Converts ONNX graph to TFLite using onnx2tf converter with exact precision target matching.
    """
    precision_upper = precision.upper()
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_out_dir:
        try:
            import onnx2tf
            print(f"[CONVERTER] Executing onnx2tf converter engine ({precision_upper})...")
            
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=tmp_out_dir,
                output_integer_quantized_tflite=(precision_upper == "INT8"),
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True
            )

            # Collect all generated .tflite files
            all_tflites = []
            for root, _, files in os.walk(tmp_out_dir):
                for f in files:
                    if f.endswith(".tflite"):
                        all_tflites.append(os.path.join(root, f))

            selected_tflite = None
            if precision_upper == "FP16":
                for f in all_tflites:
                    if "float16" in f.lower() or "fp16" in f.lower():
                        selected_tflite = f
                        break
            elif precision_upper == "INT8":
                for f in all_tflites:
                    if "int" in f.lower() or "quant" in f.lower():
                        selected_tflite = f
                        break

            if not selected_tflite:
                # For FP32 mode, prioritize float32 model over float16
                for f in all_tflites:
                    if "float32" in f.lower() or "fp32" in f.lower():
                        selected_tflite = f
                        break
                if not selected_tflite and all_tflites:
                    non_fp16 = [f for f in all_tflites if "float16" not in f.lower()]
                    selected_tflite = non_fp16[0] if non_fp16 else all_tflites[0]

            if selected_tflite and os.path.exists(selected_tflite):
                shutil.copy2(selected_tflite, output_path)
                print(f"[SUCCESS] TFLite model generated ({precision_upper}): {output_path}")
                return output_path
            else:
                raise RuntimeError("No matching TFLite file produced by onnx2tf.")

        except Exception as err:
            print(f"[WARNING] onnx2tf failed: {err}.")

    raise RuntimeError("ONNX to TFLite conversion failed. Please check dependencies.")


def auto_convert(weights_path: str,
                 output_path: str = "best.tflite",
                 arch: str = "deeplabv3plus",
                 encoder: str = "resnet50",
                 img_size: int = 640,
                 precision: str = "FP32") -> str:
    """
    Auto-detects model type (.pt vs .pth) and converts it to TFLite.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Source weight file does not exist: {weights_path}")

    ext = os.path.splitext(weights_path)[1].lower()

    if ext == ".pt" or "yolo" in os.path.basename(weights_path).lower():
        return convert_yolo_to_tflite(
            weights_path=weights_path,
            output_path=output_path,
            img_size=img_size,
            precision=precision
        )
    elif ext == ".pth" or "deeplab" in os.path.basename(weights_path).lower():
        return convert_smp_to_tflite(
            weights_path=weights_path,
            output_path=output_path,
            arch=arch,
            encoder=encoder,
            img_size=img_size,
            precision=precision
        )
    else:
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and any(k in checkpoint for k in ["model", "model_state_dict"]):
                return convert_smp_to_tflite(
                    weights_path=weights_path, output_path=output_path,
                    arch=arch, encoder=encoder, img_size=img_size, precision=precision
                )
        except Exception:
            pass
        return convert_yolo_to_tflite(
            weights_path=weights_path, output_path=output_path,
            img_size=img_size, precision=precision
        )


# ==============================================================================
# Graphical User Interface (GUI)
# ==============================================================================

def launch_gui():
    """
    Launches interactive Tkinter GUI window for selecting model weights and target output path.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("TensorFlow Lite Model Converter — D9 Workspace")
    root.geometry("640x540")
    root.minsize(600, 500)

    # Style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Custom colors
    BG_COLOR = "#f5f6f8"
    ACCENT_COLOR = "#2563eb"
    TEXT_COLOR = "#1e293b"
    
    root.configure(bg=BG_COLOR)

    # Title Banner
    banner_frame = tk.Frame(root, bg="#1e293b", padx=15, pady=15)
    banner_frame.pack(fill="x")

    title_label = tk.Label(
        banner_frame, 
        text="TensorFlow Lite Model Converter", 
        font=("Segoe UI", 14, "bold"), 
        fg="white", 
        bg="#1e293b"
    )
    title_label.pack(anchor="w")

    sub_label = tk.Label(
        banner_frame, 
        text="Convert PyTorch YOLOv8 (.pt) & PyTorch SMP (.pth) to TFLite format", 
        font=("Segoe UI", 9), 
        fg="#94a3b8", 
        bg="#1e293b"
    )
    sub_label.pack(anchor="w")

    # Main Container
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    # Variables
    weights_path_var = tk.StringVar(value="")
    output_path_var = tk.StringVar(value="")
    precision_var = tk.StringVar(value="FP32")
    img_size_var = tk.IntVar(value=640)
    arch_var = tk.StringVar(value="deeplabv3plus")
    encoder_var = tk.StringVar(value="resnet50")
    status_var = tk.StringVar(value="Select model weight file to begin...")

    # Helper Browse Functions
    def browse_weights():
        file_selected = filedialog.askopenfilename(
            title="Select Input Model Weights File (.pt / .pth)",
            initialdir=current_dir,
            filetypes=[
                ("All Model Weights (*.pt;*.pth)", "*.pt;*.pth"),
                ("YOLO Weights (*.pt)", "*.pt"),
                ("PyTorch SMP Weights (*.pth)", "*.pth"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if file_selected:
            weights_path_var.set(file_selected)
            # Auto populate output path if empty
            if not output_path_var.get():
                stem = Path(file_selected).stem
                parent_dir = os.path.dirname(file_selected)
                output_path_var.set(os.path.join(parent_dir, f"{stem}.tflite"))

    def browse_output():
        initial_file = os.path.basename(output_path_var.get()) if output_path_var.get() else "best.tflite"
        initial_dir = os.path.dirname(output_path_var.get()) if output_path_var.get() else current_dir
        
        file_selected = filedialog.asksaveasfilename(
            title="Save Output TFLite File As",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".tflite",
            filetypes=[
                ("TensorFlow Lite Model (*.tflite)", "*.tflite"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if file_selected:
            output_path_var.set(file_selected)

    # Form UI Elements
    
    # 1. Model Weights Section
    weights_lbl = ttk.Label(main_frame, text="Input Model Weights (.pt / .pth):", font=("Segoe UI", 10, "bold"))
    weights_lbl.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

    weights_entry = ttk.Entry(main_frame, textvariable=weights_path_var, width=50)
    weights_entry.grid(row=1, column=0, sticky="ew", padx=(0, 10), ipady=3)

    weights_btn = ttk.Button(main_frame, text="Browse...", command=browse_weights)
    weights_btn.grid(row=1, column=1, sticky="w")

    # 2. Output Path Section
    output_lbl = ttk.Label(main_frame, text="Target Output TFLite Path (.tflite):", font=("Segoe UI", 10, "bold"))
    output_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(15, 5))

    output_entry = ttk.Entry(main_frame, textvariable=output_path_var, width=50)
    output_entry.grid(row=3, column=0, sticky="ew", padx=(0, 10), ipady=3)

    output_btn = ttk.Button(main_frame, text="Browse Save...", command=browse_output)
    output_btn.grid(row=3, column=1, sticky="w")

    # 3. Settings Grid
    settings_frame = ttk.LabelFrame(main_frame, text=" Conversion Settings ", padding=12)
    settings_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(20, 10))

    # Precision Choice
    prec_lbl = ttk.Label(settings_frame, text="Precision Mode:")
    prec_lbl.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)

    prec_combo = ttk.Combobox(settings_frame, textvariable=precision_var, values=["FP32", "FP16", "INT8"], state="readonly", width=12)
    prec_combo.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=5)

    # Resolution
    res_lbl = ttk.Label(settings_frame, text="Input Resolution:")
    res_lbl.grid(row=0, column=2, sticky="w", padx=(0, 10), pady=5)

    res_combo = ttk.Combobox(settings_frame, textvariable=img_size_var, values=[512, 640, 768, 1024], width=12)
    res_combo.grid(row=0, column=3, sticky="w", pady=5)

    # Architecture (for .pth)
    arch_lbl = ttk.Label(settings_frame, text="Architecture (.pth):")
    arch_lbl.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)

    arch_combo = ttk.Combobox(settings_frame, textvariable=arch_var, values=["deeplabv3plus", "unet", "unetplusplus", "fpn", "pspnet"], width=12)
    arch_combo.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=5)

    # Encoder (for .pth)
    enc_lbl = ttk.Label(settings_frame, text="Encoder (.pth):")
    enc_lbl.grid(row=1, column=2, sticky="w", padx=(0, 10), pady=5)

    enc_combo = ttk.Combobox(settings_frame, textvariable=encoder_var, values=["resnet50", "resnet34", "efficientnet-b3", "mobilenet_v2"], width=12)
    enc_combo.grid(row=1, column=3, sticky="w", pady=5)

    # Status Bar
    status_lbl = ttk.Label(main_frame, textvariable=status_var, font=("Segoe UI", 9, "italic"), foreground="#475569")
    status_lbl.grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 10))

    # Progress bar
    progress_bar = ttk.Progressbar(main_frame, mode="indeterminate")
    progress_bar.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 15))

    # Convert Action Handler
    def start_conversion():
        w_path = weights_path_var.get().strip()
        o_path = output_path_var.get().strip()

        if not w_path or not os.path.exists(w_path):
            messagebox.showerror("Error", "Please select a valid model weights file (.pt or .pth).")
            return

        if not o_path:
            messagebox.showerror("Error", "Please select an output location for the .tflite model.")
            return

        # Disable controls during conversion
        convert_btn.config(state="disabled")
        progress_bar.start(10)
        status_var.set("Converting model... Please wait (this may take 15-30 seconds)...")

        def worker():
            try:
                result = auto_convert(
                    weights_path=w_path,
                    output_path=o_path,
                    arch=arch_var.get(),
                    encoder=encoder_var.get(),
                    img_size=img_size_var.get(),
                    precision=precision_var.get()
                )
                root.after(0, lambda: on_success(result))
            except Exception as e:
                err_msg = str(e)
                root.after(0, lambda: on_error(err_msg))

        threading.Thread(target=worker, daemon=True).start()

    def on_success(out_file):
        progress_bar.stop()
        convert_btn.config(state="normal")
        status_var.set(f"SUCCESS: Model converted and saved to {out_file}")
        messagebox.showinfo("Conversion Success", f"TensorFlow Lite model successfully generated!\n\nSaved at:\n{out_file}")

    def on_error(err):
        progress_bar.stop()
        convert_btn.config(state="normal")
        status_var.set(f"ERROR: Conversion failed.")
        messagebox.showerror("Conversion Failed", f"An error occurred during conversion:\n\n{err}")

    # Convert Button
    convert_btn = tk.Button(
        main_frame, 
        text="  CONVERT TO TFLITE  ", 
        font=("Segoe UI", 11, "bold"), 
        bg="#2563eb", 
        fg="white", 
        activebackground="#1d4ed8",
        activeforeground="white",
        relief="flat",
        bd=0,
        padx=15,
        pady=8,
        command=start_conversion
    )
    convert_btn.grid(row=7, column=0, columnspan=2, pady=10)

    # Grid Weights
    main_frame.columnconfigure(0, weight=1)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Convert .pt / .pth Model Weights to TensorFlow Lite (.tflite)")
    parser.add_argument("--weights", type=str, default=None, help="Input model weights file (.pt or .pth)")
    parser.add_argument("--output", type=str, default="best.tflite", help="Output .tflite file path (default: best.tflite)")
    parser.add_argument("--precision", type=str, default="FP32", choices=["FP32", "FP16", "INT8"], help="Quantization / precision mode")
    parser.add_argument("--img-size", type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("--arch", type=str, default="deeplabv3plus", help="SMP model architecture (if .pth)")
    parser.add_argument("--encoder", type=str, default="resnet50", help="SMP model encoder (if .pth)")
    parser.add_argument("--gui", action="store_true", help="Force launch GUI interface")

    args = parser.parse_args()

    # Launch GUI if no weights provided or if --gui is passed
    if args.weights is None or args.gui:
        launch_gui()
    else:
        output_path = auto_convert(
            weights_path=args.weights,
            output_path=args.output,
            arch=args.arch,
            encoder=args.encoder,
            img_size=args.img_size,
            precision=args.precision
        )
        print(f"\n[DONE] Conversion complete. Model saved at: {output_path}")


if __name__ == "__main__":
    main()
