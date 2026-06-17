"""
ALPR Desktop GUI Application — Tkinter-based.

A modern dark-themed desktop application for Vietnamese license plate recognition.
No web server required — runs entirely as a standalone desktop app.

Usage:
    python app_gui.py
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ALPRPipeline
from src.utils import draw_plate_boxes, draw_char_boxes, resolve_model_path


# ─── Color Palette (Dark Theme) ───────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0f172a",
    "bg_secondary": "#1e293b",
    "bg_card": "#1e1b4b",
    "bg_input": "#334155",
    "accent": "#38bdf8",
    "accent_hover": "#7dd3fc",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border": "#334155",
    "border_accent": "rgba(56, 189, 248, 0.3)",
}


class ALPRApp(tk.Tk):
    """Main application window for the ALPR GUI."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Vietnamese ALPR — License Plate Recognition")
        self.configure(bg=COLORS["bg_primary"])
        self.geometry("1280x800")
        self.minsize(1000, 650)

        # State
        self.pipeline = None
        self.current_image = None       # OpenCV BGR
        self.current_image_path = None
        self.results = []

        # Configure styles
        self._setup_styles()

        # Build UI
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

        # Load models in background
        self.after(100, self._load_models_async)

    # ─── Styles ────────────────────────────────────────────────────────────────

    def _setup_styles(self) -> None:
        """Configure ttk styles for the dark theme."""
        style = ttk.Style(self)
        style.theme_use("clam")

        # General
        style.configure(".", background=COLORS["bg_primary"], foreground=COLORS["text_primary"])

        # Frames
        style.configure("Card.TFrame", background=COLORS["bg_secondary"])
        style.configure("Header.TFrame", background=COLORS["bg_card"])

        # Labels
        style.configure(
            "Title.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["accent"],
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["text_secondary"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=COLORS["bg_primary"],
            foreground=COLORS["text_primary"],
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Result.TLabel",
            background=COLORS["bg_secondary"],
            foreground=COLORS["success"],
            font=("Consolas", 22, "bold"),
        )
        style.configure(
            "Raw.TLabel",
            background=COLORS["bg_secondary"],
            foreground=COLORS["warning"],
            font=("Consolas", 16),
        )
        style.configure(
            "Info.TLabel",
            background=COLORS["bg_secondary"],
            foreground=COLORS["text_secondary"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["text_muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Metric.TLabel",
            background=COLORS["bg_secondary"],
            foreground=COLORS["accent"],
            font=("Consolas", 11, "bold"),
        )

        # Buttons
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground=COLORS["bg_primary"],
            font=("Segoe UI", 11, "bold"),
            padding=(20, 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent_hover"])],
        )
        style.configure(
            "Secondary.TButton",
            background=COLORS["bg_input"],
            foreground=COLORS["text_primary"],
            font=("Segoe UI", 10),
            padding=(15, 8),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", COLORS["border"])],
        )

        # Scale (slider)
        style.configure(
            "Accent.Horizontal.TScale",
            background=COLORS["bg_secondary"],
            troughcolor=COLORS["bg_input"],
        )

        # Separator
        style.configure("Accent.TSeparator", background=COLORS["border"])

    # ─── Header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        """Build the application header bar."""
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill=tk.X, padx=0, pady=0)

        inner = ttk.Frame(header, style="Header.TFrame")
        inner.pack(fill=tk.X, padx=20, pady=12)

        ttk.Label(
            inner, text="🚗  Vietnamese ALPR System", style="Title.TLabel",
        ).pack(side=tk.LEFT)

        ttk.Label(
            inner,
            text="Two-stage YOLOv8 Pipeline  •  Plate Detection → Character Recognition → Vietnamese Format Correction",
            style="Subtitle.TLabel",
        ).pack(side=tk.LEFT, padx=(20, 0))

    # ─── Main Content ──────────────────────────────────────────────────────────

    def _build_main_content(self) -> None:
        """Build the main content area with left controls and right results."""
        main = tk.Frame(self, bg=COLORS["bg_primary"])
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 0))

        # Left panel: Controls + Image
        left = tk.Frame(main, bg=COLORS["bg_primary"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self._build_controls(left)
        self._build_image_panel(left)

        # Right panel: Results
        right = tk.Frame(main, bg=COLORS["bg_primary"], width=420)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        right.pack_propagate(False)

        self._build_results_panel(right)

    def _build_controls(self, parent: tk.Frame) -> None:
        """Build the controls panel (file select, sliders, run button)."""
        ctrl = tk.Frame(parent, bg=COLORS["bg_secondary"], highlightthickness=1,
                        highlightbackground=COLORS["border"])
        ctrl.pack(fill=tk.X, pady=(0, 8))

        inner = tk.Frame(ctrl, bg=COLORS["bg_secondary"])
        inner.pack(fill=tk.X, padx=16, pady=12)

        # Row 1: File selection
        row1 = tk.Frame(inner, bg=COLORS["bg_secondary"])
        row1.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(
            row1, text="📂  Browse Image", style="Accent.TButton",
            command=self._browse_image,
        ).pack(side=tk.LEFT)

        ttk.Button(
            row1, text="📁  Load Sample", style="Secondary.TButton",
            command=self._load_sample,
        ).pack(side=tk.LEFT, padx=(10, 0))

        self.btn_run = ttk.Button(
            row1, text="▶  Run Detection", style="Accent.TButton",
            command=self._run_detection_async, state=tk.DISABLED,
        )
        self.btn_run.pack(side=tk.RIGHT)

        self.lbl_filename = tk.Label(
            row1, text="No image loaded", bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"], font=("Segoe UI", 9),
        )
        self.lbl_filename.pack(side=tk.LEFT, padx=(15, 0))

        # Row 2: Confidence sliders
        row2 = tk.Frame(inner, bg=COLORS["bg_secondary"])
        row2.pack(fill=tk.X)

        # Plate confidence
        tk.Label(
            row2, text="Plate Conf:", bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"], font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        self.plate_conf_var = tk.DoubleVar(value=0.4)
        self.plate_conf_slider = ttk.Scale(
            row2, from_=0.1, to=1.0, variable=self.plate_conf_var,
            orient=tk.HORIZONTAL, length=120, style="Accent.Horizontal.TScale",
            command=lambda v: self.plate_conf_label.config(text=f"{float(v):.2f}"),
        )
        self.plate_conf_slider.pack(side=tk.LEFT, padx=(5, 0))
        self.plate_conf_label = tk.Label(
            row2, text="0.40", bg=COLORS["bg_secondary"],
            fg=COLORS["accent"], font=("Consolas", 9, "bold"), width=4,
        )
        self.plate_conf_label.pack(side=tk.LEFT, padx=(2, 15))

        # Char confidence
        tk.Label(
            row2, text="Char Conf:", bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"], font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        self.char_conf_var = tk.DoubleVar(value=0.3)
        self.char_conf_slider = ttk.Scale(
            row2, from_=0.1, to=1.0, variable=self.char_conf_var,
            orient=tk.HORIZONTAL, length=120, style="Accent.Horizontal.TScale",
            command=lambda v: self.char_conf_label.config(text=f"{float(v):.2f}"),
        )
        self.char_conf_slider.pack(side=tk.LEFT, padx=(5, 0))
        self.char_conf_label = tk.Label(
            row2, text="0.30", bg=COLORS["bg_secondary"],
            fg=COLORS["accent"], font=("Consolas", 9, "bold"), width=4,
        )
        self.char_conf_label.pack(side=tk.LEFT, padx=(2, 0))

    def _build_image_panel(self, parent: tk.Frame) -> None:
        """Build the image display area."""
        frame = tk.Frame(parent, bg=COLORS["bg_secondary"], highlightthickness=1,
                         highlightbackground=COLORS["border"])
        frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(
            frame, bg=COLORS["bg_secondary"], highlightthickness=0,
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Placeholder text
        self.image_canvas.create_text(
            400, 250, text="Load an image to begin detection",
            fill=COLORS["text_muted"], font=("Segoe UI", 14),
            tags="placeholder",
        )

        # Bind resize
        self.image_canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_results_panel(self, parent: tk.Frame) -> None:
        """Build the results display panel on the right side."""
        # Header
        header = tk.Frame(parent, bg=COLORS["bg_primary"])
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header, text="🔍  Detection Results", style="Section.TLabel").pack(
            side=tk.LEFT,
        )

        self.lbl_plate_count = tk.Label(
            header, text="", bg=COLORS["bg_primary"],
            fg=COLORS["text_muted"], font=("Segoe UI", 10),
        )
        self.lbl_plate_count.pack(side=tk.RIGHT)

        # Scrollable results container
        canvas_frame = tk.Frame(parent, bg=COLORS["bg_primary"])
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.results_canvas = tk.Canvas(
            canvas_frame, bg=COLORS["bg_primary"], highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                  command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg=COLORS["bg_primary"])
        self.results_window = self.results_canvas.create_window(
            (0, 0), window=self.results_inner, anchor=tk.NW,
        )

        self.results_inner.bind("<Configure>", self._on_results_configure)
        self.results_canvas.bind("<Configure>", self._on_results_canvas_configure)

        # Mouse wheel scrolling
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Initial placeholder
        self.lbl_no_results = tk.Label(
            self.results_inner,
            text="No results yet.\nLoad an image and click\n'Run Detection' to begin.",
            bg=COLORS["bg_primary"], fg=COLORS["text_muted"],
            font=("Segoe UI", 11), justify=tk.CENTER,
        )
        self.lbl_no_results.pack(pady=80)

    # ─── Status Bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        """Build the bottom status bar."""
        status = ttk.Frame(self, style="Header.TFrame")
        status.pack(fill=tk.X, side=tk.BOTTOM)

        inner = tk.Frame(status, bg=COLORS["bg_card"])
        inner.pack(fill=tk.X, padx=10, pady=4)

        self.lbl_status = tk.Label(
            inner, text="⏳ Loading models...", bg=COLORS["bg_card"],
            fg=COLORS["text_muted"], font=("Segoe UI", 9),
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_latency = tk.Label(
            inner, text="", bg=COLORS["bg_card"],
            fg=COLORS["accent"], font=("Consolas", 9),
        )
        self.lbl_latency.pack(side=tk.RIGHT)

    # ─── Model Loading ─────────────────────────────────────────────────────────

    def _load_models_async(self) -> None:
        """Load YOLO models in a background thread."""
        def _load():
            try:
                plate_path = str(PROJECT_ROOT / "models" / "final_best.pt")
                char_path = str(PROJECT_ROOT / "models" / "final_char_yolo.pt")

                self.pipeline = ALPRPipeline(plate_path, char_path)
                self.after(0, self._on_models_loaded)
            except Exception as e:
                self.after(0, lambda: self._on_model_error(str(e)))

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def _on_models_loaded(self) -> None:
        """Callback when models are loaded successfully."""
        self.lbl_status.config(text="✅ Models loaded — Ready", fg=COLORS["success"])
        if self.current_image is not None:
            self.btn_run.config(state=tk.NORMAL)

    def _on_model_error(self, error: str) -> None:
        """Callback when model loading fails."""
        self.lbl_status.config(
            text=f"❌ Model load failed: {error}", fg=COLORS["error"],
        )
        messagebox.showerror(
            "Model Error",
            f"Failed to load YOLO models.\n\n{error}\n\n"
            "Please ensure model weights exist in the 'models/' directory.\n"
            "See models/README.md for download instructions.",
        )

    # ─── Image Loading ─────────────────────────────────────────────────────────

    def _browse_image(self) -> None:
        """Open a file dialog to select an image."""
        path = filedialog.askopenfilename(
            title="Select Vehicle Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_image(path)

    def _load_sample(self) -> None:
        """Load a sample image from the data/test/images directory."""
        sample_dir = PROJECT_ROOT / "data" / "test" / "images"
        if not sample_dir.exists():
            sample_dir = PROJECT_ROOT / "data" / "samples"

        if not sample_dir.exists():
            messagebox.showinfo("No Samples", "No sample images found in data/ directory.")
            return

        path = filedialog.askopenfilename(
            title="Select Sample Image",
            initialdir=str(sample_dir),
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        """Load and display an image."""
        try:
            self.current_image = cv2.imread(path)
            if self.current_image is None:
                raise ValueError("Cannot decode image")
            self.current_image_path = path
            self.lbl_filename.config(
                text=os.path.basename(path), fg=COLORS["text_primary"],
            )
            self._display_image(self.current_image)
            self._clear_results()

            if self.pipeline is not None:
                self.btn_run.config(state=tk.NORMAL)

            self.lbl_status.config(
                text=f"📷 Loaded: {os.path.basename(path)}", fg=COLORS["text_primary"],
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _display_image(self, image: np.ndarray, fit: bool = True) -> None:
        """Display a BGR image on the canvas."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if fit:
            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                pil_img.thumbnail((canvas_w - 4, canvas_h - 4), Image.Resampling.LANCZOS)

        self._current_photo = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            self.image_canvas.winfo_width() // 2,
            self.image_canvas.winfo_height() // 2,
            image=self._current_photo, anchor=tk.CENTER,
        )

    # ─── Detection ─────────────────────────────────────────────────────────────

    def _run_detection_async(self) -> None:
        """Run detection in a background thread."""
        if self.pipeline is None or self.current_image is None:
            return

        self.btn_run.config(state=tk.DISABLED)
        self.lbl_status.config(text="🔄 Running AI detection...", fg=COLORS["warning"])

        def _run():
            try:
                t_start = time.perf_counter()
                results = self.pipeline.process_image(
                    self.current_image,
                    conf_plate=self.plate_conf_var.get(),
                    conf_char=self.char_conf_var.get(),
                )
                t_total = (time.perf_counter() - t_start) * 1000
                self.after(0, lambda: self._on_detection_done(results, t_total))
            except Exception as e:
                self.after(0, lambda: self._on_detection_error(str(e)))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _on_detection_done(self, results: list[dict], total_ms: float) -> None:
        """Callback when detection completes."""
        self.results = results
        self.btn_run.config(state=tk.NORMAL)

        # Draw annotated image
        if results:
            plates_for_draw = [
                {"bbox": r["bbox"], "conf": r["plate_conf"]}
                for r in results
            ]
            annotated = draw_plate_boxes(self.current_image, plates_for_draw)
            self._display_image(annotated)

        # Update results panel
        self._show_results(results)

        # Update status
        n = len(results)
        self.lbl_status.config(
            text=f"✅ Found {n} plate{'s' if n != 1 else ''}", fg=COLORS["success"],
        )
        self.lbl_latency.config(text=f"⚡ {total_ms:.0f} ms total")
        self.lbl_plate_count.config(text=f"{n} detected")

    def _on_detection_error(self, error: str) -> None:
        """Callback when detection fails."""
        self.btn_run.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"❌ Error: {error}", fg=COLORS["error"])
        messagebox.showerror("Detection Error", f"An error occurred:\n{error}")

    # ─── Results Display ───────────────────────────────────────────────────────

    def _clear_results(self) -> None:
        """Clear the results panel."""
        for widget in self.results_inner.winfo_children():
            widget.destroy()

        self.lbl_plate_count.config(text="")
        self.lbl_latency.config(text="")

    def _show_results(self, results: list[dict]) -> None:
        """Populate the results panel with detection results."""
        self._clear_results()

        if not results:
            tk.Label(
                self.results_inner,
                text="⚠ No plates detected.\nTry lowering the\nconfidence thresholds.",
                bg=COLORS["bg_primary"], fg=COLORS["warning"],
                font=("Segoe UI", 11), justify=tk.CENTER,
            ).pack(pady=80)
            return

        for idx, res in enumerate(results):
            self._create_result_card(idx, res)

    def _create_result_card(self, idx: int, result: dict) -> None:
        """Create a single result card for a detected plate."""
        card = tk.Frame(
            self.results_inner, bg=COLORS["bg_secondary"],
            highlightthickness=1, highlightbackground=COLORS["border"],
        )
        card.pack(fill=tk.X, pady=(0, 10), padx=2)

        inner = tk.Frame(card, bg=COLORS["bg_secondary"])
        inner.pack(fill=tk.X, padx=14, pady=12)

        # Title row
        title_row = tk.Frame(inner, bg=COLORS["bg_secondary"])
        title_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            title_row, text=f"📍  Plate #{idx + 1}",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT)

        tk.Label(
            title_row, text=f"conf: {result['plate_conf']:.2f}",
            bg=COLORS["bg_secondary"], fg=COLORS["text_muted"],
            font=("Consolas", 9),
        ).pack(side=tk.RIGHT)

        # Plate crop image
        crop = result["crop"]
        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            # Scale crop to fit card width
            max_w = 380
            ratio = min(max_w / pil_crop.width, 100 / pil_crop.height)
            new_size = (int(pil_crop.width * ratio), int(pil_crop.height * ratio))
            pil_crop = pil_crop.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(pil_crop)

            crop_label = tk.Label(
                inner, image=photo, bg=COLORS["bg_secondary"],
            )
            crop_label.image = photo  # Prevent garbage collection
            crop_label.pack(pady=(0, 10))

        # Annotated crop with char boxes
        if crop.size > 0 and result.get("characters"):
            annotated_crop = draw_char_boxes(crop, result["characters"])
            crop_ann_rgb = cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB)
            pil_ann = Image.fromarray(crop_ann_rgb)

            ratio = min(max_w / pil_ann.width, 80 / pil_ann.height)
            new_size = (int(pil_ann.width * ratio), int(pil_ann.height * ratio))
            pil_ann = pil_ann.resize(new_size, Image.Resampling.LANCZOS)

            photo_ann = ImageTk.PhotoImage(pil_ann)
            ann_label = tk.Label(
                inner, image=photo_ann, bg=COLORS["bg_secondary"],
            )
            ann_label.image = photo_ann
            ann_label.pack(pady=(0, 10))

        # Separator
        tk.Frame(inner, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=4)

        # Raw OCR text
        tk.Label(
            inner, text="Raw OCR:", bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"], font=("Segoe UI", 9),
            anchor=tk.W,
        ).pack(fill=tk.X)

        raw_text = result["raw_text"] or "[empty]"
        tk.Label(
            inner, text=raw_text, bg=COLORS["bg_secondary"],
            fg=COLORS["warning"], font=("Consolas", 16),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))

        # Corrected text (main result)
        tk.Label(
            inner, text="Corrected:", bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"], font=("Segoe UI", 9),
            anchor=tk.W,
        ).pack(fill=tk.X)

        corrected = result["corrected_text"] or "[empty]"
        result_frame = tk.Frame(
            inner, bg=COLORS["bg_primary"], highlightthickness=2,
            highlightbackground=COLORS["success"],
        )
        result_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            result_frame, text=corrected, bg=COLORS["bg_primary"],
            fg=COLORS["success"], font=("Consolas", 22, "bold"),
            pady=8,
        ).pack()

        # Timing metrics
        timing = result.get("timing", {})
        metrics_row = tk.Frame(inner, bg=COLORS["bg_secondary"])
        metrics_row.pack(fill=tk.X)

        for label, value in [
            ("Plate", f"{timing.get('plate_ms', 0):.0f}ms"),
            ("Char", f"{timing.get('char_ms', 0):.0f}ms"),
            ("BBox", str(result["bbox"])),
        ]:
            tk.Label(
                metrics_row, text=f"{label}: {value}",
                bg=COLORS["bg_secondary"], fg=COLORS["text_muted"],
                font=("Consolas", 8),
            ).pack(side=tk.LEFT, padx=(0, 12))

    # ─── Event Handlers ────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event: tk.Event) -> None:
        """Redraw image when canvas is resized."""
        if self.results:
            plates_for_draw = [
                {"bbox": r["bbox"], "conf": r["plate_conf"]}
                for r in self.results
            ]
            annotated = draw_plate_boxes(self.current_image, plates_for_draw)
            self._display_image(annotated)
        elif self.current_image is not None:
            self._display_image(self.current_image)

    def _on_results_configure(self, event: tk.Event) -> None:
        """Update scroll region when results panel content changes."""
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_results_canvas_configure(self, event: tk.Event) -> None:
        """Resize inner frame to match canvas width."""
        self.results_canvas.itemconfig(self.results_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mouse wheel scrolling for results panel."""
        self.results_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def main() -> None:
    """Launch the ALPR desktop GUI application."""
    app = ALPRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
