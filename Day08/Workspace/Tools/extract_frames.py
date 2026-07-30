import os
import sys
import time
import subprocess
import threading
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

DEFAULT_OUTPUT_DIR = r"C:\Users\kalra\OneDrive\Desktop\Staj\W\D6\Workspace3\screenshots"

def extract_frames(video_path, output_dir, target_width=1920, target_height=1080, progress_callback=None, stop_event=None):
    """
    Extracts 1 frame per second from video_path and saves it to output_dir as a 1080p image.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or total_frames <= 0:
        # Fallback if FPS cannot be read properly
        fps = 60.0 if fps <= 0 else fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    frame_interval = max(1, int(round(fps)))
    total_seconds = int(total_frames / fps) if fps > 0 else 0
    
    saved_count = 0
    current_frame_idx = 0
    
    while True:
        if stop_event and stop_event.is_set():
            break
            
        # Set frame position for exact second sampling
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break
        
        current_second = int(current_frame_idx / fps)
        
        # Check resolution and resize to 1080p (1920x1080) if needed
        h, w = frame.shape[:2]
        if (w, h) != (target_width, target_height):
            frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            frame_resized = frame
            
        # Generate clean filename with second timestamp
        mins = current_second // 60
        secs = current_second % 60
        hours = mins // 60
        mins = mins % 60
        
        time_str = f"{hours:02d}h{mins:02d}m{secs:02d}s" if hours > 0 else f"{mins:02d}m{secs:02d}s"
        filename = f"frame_sec_{current_second:05d}_{time_str}.png"
        save_path = os.path.join(output_dir, filename)
        
        # Save high quality PNG
        cv2.imwrite(save_path, frame_resized)
        saved_count += 1
        
        if progress_callback:
            percent = min(100.0, (current_frame_idx / max(1, total_frames)) * 100.0)
            progress_callback(saved_count, total_seconds, current_second, percent)
            
        current_frame_idx += frame_interval
        if current_frame_idx >= total_frames:
            break
            
    cap.release()
    return saved_count


class VideoExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("1080p Video Screenshot Extractor (1 FPS)")
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        
        # Dark modern style styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Variables
        self.video_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.status_var = tk.StringVar(value="Lütfen bir video dosyası seçin.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.is_processing = False
        self.stop_event = threading.Event()
        
        self._build_ui()
        
        # Automatically prompt file dialog after GUI window renders
        self.root.after(300, self.select_video)
        
    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header Label
        header_label = ttk.Label(
            main_frame,
            text="🎬 Video Frame Ekstraktörü (1080p - 1 FPS)",
            font=("Segoe UI", 14, "bold")
        )
        header_label.pack(anchor="w", pady=(0, 15))
        
        # Video File Selection Group
        video_group = ttk.LabelFrame(main_frame, text=" Video Seçimi ", padding=10)
        video_group.pack(fill=tk.X, pady=(0, 15))
        
        v_entry = ttk.Entry(video_group, textvariable=self.video_path_var, width=50)
        v_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        v_btn = ttk.Button(video_group, text="Video Seç...", command=self.select_video)
        v_btn.pack(side=tk.RIGHT)
        
        # Output Directory Selection Group
        output_group = ttk.LabelFrame(main_frame, text=" Kayıt Dizini (Workspace3 Screenshots) ", padding=10)
        output_group.pack(fill=tk.X, pady=(0, 15))
        
        o_entry = ttk.Entry(output_group, textvariable=self.output_dir_var, width=50)
        o_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        o_btn = ttk.Button(output_group, text="Klasör Seç...", command=self.select_output_dir)
        o_btn.pack(side=tk.RIGHT)
        
        # Info Box
        info_label = ttk.Label(
            main_frame,
            text="ℹ️ Açıklama: Seçilen video üzerinde her saniyede 1 frame (1 FPS)\n"
                 "   1080p (1920x1080) çözünürlükte screenshots klasörüne kaydedilir.",
            font=("Segoe UI", 9),
            foreground="#444444"
        )
        info_label.pack(anchor="w", pady=(0, 15))
        
        # Progress Group
        progress_group = ttk.LabelFrame(main_frame, text=" Durum ve İlerleme ", padding=10)
        progress_group.pack(fill=tk.X, pady=(0, 15))
        
        self.progress_bar = ttk.Progressbar(
            progress_group,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        status_label = ttk.Label(progress_group, textvariable=self.status_var, font=("Segoe UI", 9))
        status_label.pack(anchor="w")
        
        # Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_btn = ttk.Button(btn_frame, text="🚀 İşlemi Başlat", command=self.start_extraction)
        self.start_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.open_folder_btn = ttk.Button(btn_frame, text="📁 Kayıt Klasörünü Aç", command=self.open_output_folder)
        self.open_folder_btn.pack(side=tk.LEFT)
        
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="İşlenecek Videoyu Seçin",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.webm"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.status_var.set(f"Seçilen video: {os.path.basename(file_path)}")
            
    def select_output_dir(self):
        folder_path = filedialog.askdirectory(
            title="Kayıt Klasörünü Seçin",
            initialdir=self.output_dir_var.get()
        )
        if folder_path:
            self.output_dir_var.set(folder_path)
            
    def open_output_folder(self):
        out_dir = self.output_dir_var.get()
        os.makedirs(out_dir, exist_ok=True)
        if os.name == 'nt':
            os.startfile(out_dir)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', out_dir])
        else:
            subprocess.Popen(['xdg-open', out_dir])
            
    def start_extraction(self):
        video_path = self.video_path_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        
        if not video_path:
            messagebox.showwarning("Uyarı", "Lütfen önce bir video dosyası seçin!")
            self.select_video()
            return
            
        if not os.path.isfile(video_path):
            messagebox.showerror("Hata", f"Seçilen video dosyası bulunamadı:\n{video_path}")
            return
            
        if self.is_processing:
            return
            
        self.is_processing = True
        self.stop_event.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("İşlem başlatılıyor...")
        self.progress_var.set(0.0)
        
        # Run extraction in background thread so GUI remains responsive
        threading.Thread(
            target=self._run_thread,
            args=(video_path, output_dir),
            daemon=True
        ).start()
        
    def _run_thread(self, video_path, output_dir):
        start_time = time.time()
        
        def update_progress(saved, total_secs, current_sec, percent):
            mins, secs = divmod(current_sec, 60)
            t_mins, t_secs = divmod(total_secs, 60)
            msg = f"İşleniyor: {saved} frame kaydedildi | {mins:02d}:{secs:02d} / {t_mins:02d}:{t_secs:02d} (%{percent:.1f})"
            self.root.after(0, lambda: (self.status_var.set(msg), self.progress_var.set(percent)))
            
        try:
            total_saved = extract_frames(
                video_path=video_path,
                output_dir=output_dir,
                target_width=1920,
                target_height=1080,
                progress_callback=update_progress,
                stop_event=self.stop_event
            )
            
            elapsed = time.time() - start_time
            done_msg = f"Tamamlandı! Toplam {total_saved} adet 1080p frame ({elapsed:.1f} saniyede) kaydedildi."
            
            self.root.after(0, lambda: self._on_success(done_msg, output_dir))
            
        except Exception as e:
            err_msg = f"Hata oluştu: {str(e)}"
            self.root.after(0, lambda: self._on_error(err_msg))
            
    def _on_success(self, msg, output_dir):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set(msg)
        self.progress_var.set(100.0)
        
        answer = messagebox.askyesno(
            "İşlem Tamamlandı",
            f"{msg}\n\nKayıt klasörünü açmak ister misiniz?\n({output_dir})"
        )
        if answer:
            self.open_output_folder()
            
    def _on_error(self, err_msg):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set(err_msg)
        messagebox.showerror("Hata", err_msg)


def main():
    parser = argparse.ArgumentParser(description="1080p Video Screenshot Extractor (1 FPS)")
    parser.add_argument("--video", type=str, help="Video dosyası yolu")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Ekran görüntülerinin kaydedileceği klasör")
    args = parser.parse_args()
    
    if args.video:
        # CLI Mode
        print(f"Video: {args.video}")
        print(f"Kayıt Klasörü: {args.output}")
        def cli_progress(saved, total_secs, current_sec, percent):
            print(f"\r[İlerleme %{percent:.1f}] Kaydedilen Frame: {saved}", end="", flush=True)
            
        count = extract_frames(args.video, args.output, progress_callback=cli_progress)
        print(f"\nİşlem tamamlandı. Toplam {count} adet 1080p frame kaydedildi.")
    else:
        # GUI Mode
        root = tk.Tk()
        app = VideoExtractorGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
