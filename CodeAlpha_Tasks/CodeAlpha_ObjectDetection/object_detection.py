"""
CodeAlpha Internship - Task 4: Object Detection and Tracking
Uses YOLOv8 (ultralytics) + ByteTrack for real-time detection and tracking
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys

# ── Install dependencies ──────────────────────────────────────────────────────
def install_if_missing(pkg, import_name=None):
    import importlib, subprocess
    name = import_name or pkg.split("[")[0].replace("-", "_")
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

install_if_missing("ultralytics")
install_if_missing("opencv-python", "cv2")
install_if_missing("numpy")
install_if_missing("Pillow", "PIL")

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


# ── Tracker (Simple Centroid-based SORT-like tracker) ────────────────────────
class SimpleTracker:
    """Lightweight centroid tracker for object persistence across frames."""
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.objects = {}   # id -> centroid
        self.lost = {}      # id -> frames lost
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, class_name, confidence)
        Returns: list of (x1, y1, x2, y2, track_id, class_name, confidence)
        """
        if not detections:
            # Increment lost count for all
            for tid in list(self.lost.keys()):
                self.lost[tid] += 1
                if self.lost[tid] > self.max_lost:
                    del self.objects[tid]
                    del self.lost[tid]
            return []

        centroids = np.array([
            [(d[0]+d[2])//2, (d[1]+d[3])//2] for d in detections
        ], dtype=float)

        if not self.objects:
            for i, c in enumerate(centroids):
                self.objects[self.next_id] = c
                self.lost[self.next_id] = 0
                self.next_id += 1
        else:
            obj_ids = list(self.objects.keys())
            obj_cents = np.array([self.objects[i] for i in obj_ids])
            # Distance matrix
            D = np.linalg.norm(obj_cents[:, None] - centroids[None], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            used_rows, used_cols = set(), set()
            assigned = {}
            for r in rows:
                c = cols[r]
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] < 100:  # max distance threshold
                    tid = obj_ids[r]
                    assigned[tid] = c
                    used_rows.add(r)
                    used_cols.add(c)
            # Update matched
            for tid, c_idx in assigned.items():
                self.objects[tid] = centroids[c_idx]
                self.lost[tid] = 0
            # New objects for unmatched detections
            for c_idx in range(len(centroids)):
                if c_idx not in used_cols:
                    self.objects[self.next_id] = centroids[c_idx]
                    self.lost[self.next_id] = 0
                    self.next_id += 1
            # Remove long-lost objects
            for tid in obj_ids:
                if tid not in assigned:
                    self.lost[tid] = self.lost.get(tid, 0) + 1
                    if self.lost[tid] > self.max_lost:
                        del self.objects[tid]
                        del self.lost[tid]

        # Assemble output
        results = []
        obj_ids = list(self.objects.keys())
        obj_cents = [self.objects[i] for i in obj_ids]
        for i, det in enumerate(detections):
            if i < len(obj_ids):
                tid = obj_ids[i]
                results.append((*det[:4], tid, det[4], det[5]))
        return results


# ── YOLO Colors ───────────────────────────────────────────────────────────────
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def get_color(cls_id):
    return COLORS[cls_id % len(COLORS)]


# ── Main App ──────────────────────────────────────────────────────────────────
class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 CodeAlpha Object Detection & Tracking")
        self.root.geometry("1100x700")
        self.root.configure(bg="#0a0a0a")
        self.model = None
        self.tracker = SimpleTracker()
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.detected_objects = {}
        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # Left panel: video feed
        left = tk.Frame(self.root, bg="#0a0a0a")
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        tk.Label(left, text="🎯 Object Detection & Tracking",
                 font=("Consolas", 16, "bold"), fg="#00ff88", bg="#0a0a0a").pack()
        tk.Label(left, text="CodeAlpha AI Internship — Task 4 | YOLOv8 + Centroid Tracker",
                 font=("Consolas", 8), fg="#555", bg="#0a0a0a").pack(pady=(0, 8))

        self.canvas = tk.Label(left, bg="#111", width=780, height=500,
                               text="📷 Video feed will appear here",
                               font=("Consolas", 12), fg="#333")
        self.canvas.pack(pady=5)

        # Buttons
        btn_frame = tk.Frame(left, bg="#0a0a0a")
        btn_frame.pack(pady=8)

        self.webcam_btn = tk.Button(btn_frame, text="📷 Webcam", command=self._start_webcam,
                                    bg="#00ff88", fg="#0a0a0a", font=("Consolas", 10, "bold"),
                                    relief="flat", padx=12, pady=6, cursor="hand2")
        self.webcam_btn.grid(row=0, column=0, padx=6)

        self.video_btn = tk.Button(btn_frame, text="📁 Open Video", command=self._open_video,
                                   bg="#0088ff", fg="white", font=("Consolas", 10, "bold"),
                                   relief="flat", padx=12, pady=6, cursor="hand2")
        self.video_btn.grid(row=0, column=1, padx=6)

        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", command=self._stop,
                                  bg="#ff4444", fg="white", font=("Consolas", 10, "bold"),
                                  relief="flat", padx=12, pady=6, cursor="hand2",
                                  state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=6)

        # Right panel: stats
        right = tk.Frame(self.root, bg="#111", width=280)
        right.pack(side="right", fill="y", padx=(0, 10), pady=10)
        right.pack_propagate(False)

        tk.Label(right, text="📊 Detection Stats",
                 font=("Consolas", 12, "bold"), fg="#00ff88", bg="#111").pack(pady=(15, 5))

        # Model status
        self.model_label = tk.Label(right, text="⏳ Loading YOLOv8...",
                                    font=("Consolas", 9), fg="#ffaa00", bg="#111")
        self.model_label.pack(pady=3)

        # Stats
        stats_frame = tk.Frame(right, bg="#1a1a1a", relief="groove", bd=1)
        stats_frame.pack(fill="x", padx=10, pady=8)

        self.fps_label = tk.Label(stats_frame, text="FPS: --",
                                  font=("Consolas", 11, "bold"), fg="#00ff88", bg="#1a1a1a")
        self.fps_label.pack(anchor="w", padx=10, pady=2)

        self.frame_label = tk.Label(stats_frame, text="Frames: 0",
                                    font=("Consolas", 9), fg="#888", bg="#1a1a1a")
        self.frame_label.pack(anchor="w", padx=10)

        self.det_label = tk.Label(stats_frame, text="Detections: 0",
                                  font=("Consolas", 9), fg="#888", bg="#1a1a1a")
        self.det_label.pack(anchor="w", padx=10, pady=(0, 8))

        # Confidence threshold
        tk.Label(right, text="Confidence Threshold",
                 font=("Consolas", 9), fg="#aaa", bg="#111").pack(pady=(5, 0))
        self.conf_var = tk.DoubleVar(value=0.4)
        ttk.Scale(right, from_=0.1, to=0.9, variable=self.conf_var,
                  orient="horizontal", length=200).pack()
        self.conf_display = tk.Label(right, textvariable=self.conf_var,
                                     font=("Consolas", 9, "bold"), fg="#00ff88", bg="#111")
        self.conf_display.pack()

        # Objects detected list
        tk.Label(right, text="🏷 Objects in Frame",
                 font=("Consolas", 10, "bold"), fg="#00ff88", bg="#111").pack(pady=(10, 3))
        self.obj_listbox = tk.Listbox(right, font=("Consolas", 9), bg="#1a1a1a",
                                      fg="#cccccc", relief="flat", height=12,
                                      selectbackground="#333")
        self.obj_listbox.pack(fill="x", padx=10, pady=5)

        # Status bar
        self.status = tk.Label(self.root, text="Ready — Load YOLOv8 model...",
                               font=("Consolas", 8), fg="#555", bg="#0a0a0a", anchor="w")
        self.status.pack(fill="x", side="bottom", padx=10, pady=3)

    def _load_model(self):
        def load():
            try:
                self.model = YOLO("yolov8n.pt")  # nano model, auto-downloads
                self.root.after(0, lambda: self.model_label.config(
                    text="✅ YOLOv8n Loaded", fg="#00ff88"))
                self.root.after(0, lambda: self.status.config(
                    text="✅ YOLOv8n model ready. Start webcam or open a video."))
            except Exception as e:
                self.root.after(0, lambda: self.model_label.config(
                    text=f"❌ Model Error", fg="#ff4444"))
                self.root.after(0, lambda: messagebox.showerror("Model Error", str(e)))
        threading.Thread(target=load, daemon=True).start()

    def _start_webcam(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            return
        self.running = True
        self.webcam_btn.config(state="disabled")
        self.video_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="📷 Webcam running...")
        threading.Thread(target=self._process_frames, daemon=True).start()

    def _open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not path:
            return
        if self.running:
            self._stop()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", "Cannot open video file.")
            return
        self.running = True
        self.webcam_btn.config(state="disabled")
        self.video_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text=f"▶ Playing: {os.path.basename(path)}")
        threading.Thread(target=self._process_frames, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.webcam_btn.config(state="normal")
        self.video_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text="⏹ Stopped.")

    def _process_frames(self):
        import time
        prev_time = time.time()
        self.tracker = SimpleTracker()
        self.frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, self._stop)
                break

            self.frame_count += 1
            conf_thresh = self.conf_var.get()

            if self.model is None:
                continue

            # Run YOLO detection
            results = self.model(frame, verbose=False, conf=conf_thresh)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]
                detections.append((x1, y1, x2, y2, cls_name, conf))

            # Update tracker
            tracked = self.tracker.update(detections)

            # Draw on frame
            current_objects = {}
            for item in tracked:
                x1, y1, x2, y2, tid, cls_name, conf = item
                color = get_color(tid)
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Label background
                label = f"{cls_name} #{tid} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                # Track centroid
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, color, -1)
                current_objects[cls_name] = current_objects.get(cls_name, 0) + 1

            # FPS overlay
            curr_time = time.time()
            fps = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f} | Detections: {len(tracked)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

            # Convert for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((780, 500))
            photo = ImageTk.PhotoImage(img)

            # Update GUI (thread-safe)
            fc = self.frame_count
            fps_v = fps
            objs = dict(current_objects)
            n_det = len(tracked)

            def update_ui(p=photo, f=fc, fv=fps_v, o=objs, nd=n_det):
                self.canvas.config(image=p, text="")
                self.canvas.image = p
                self.fps_label.config(text=f"FPS: {fv:.1f}")
                self.frame_label.config(text=f"Frames: {f}")
                self.det_label.config(text=f"Detections: {nd}")
                self.obj_listbox.delete(0, "end")
                for cls, cnt in sorted(o.items()):
                    self.obj_listbox.insert("end", f"  • {cls}: {cnt}")

            self.root.after(0, update_ui)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
