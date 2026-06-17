"""
Utility functions for the ALPR system.

Provides helpers for image loading, path resolution, logging setup,
and visualization of detection results.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_model_path(relative_path: str) -> Path:
    """
    Resolve a model path relative to the project root.

    Args:
        relative_path: Path relative to project root (e.g. 'models/final_best.pt').

    Returns:
        Absolute Path object.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    full_path = get_project_root() / relative_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {full_path}\n"
            f"Please download model weights. See models/README.md for instructions."
        )
    return full_path


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        BGR image as numpy array.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    return image


def draw_plate_boxes(
    image: np.ndarray,
    plates: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw bounding boxes around detected license plates on an image.

    Args:
        image: BGR image (will not be modified in-place).
        plates: List of plate dicts with 'bbox' and 'conf' keys.
        color: BGR color for the bounding box.
        thickness: Line thickness in pixels.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    for idx, plate in enumerate(plates):
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"Plate #{idx + 1} ({plate['conf']:.2f})"
        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
    return annotated


def draw_char_boxes(
    crop: np.ndarray,
    chars: list[dict],
    box_color: tuple[int, int, int] = (255, 0, 0),
    text_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw bounding boxes around detected characters on a plate crop.

    Args:
        crop: BGR plate crop image (will not be modified in-place).
        chars: List of char dicts with normalized coordinates.
        box_color: BGR color for character bounding boxes.
        text_color: BGR color for character labels.

    Returns:
        Annotated copy of the crop.
    """
    annotated = crop.copy()
    h_crop, w_crop = crop.shape[:2]

    for char in chars:
        xc = char["x_center"]
        yc = char["y_center"]
        w = char["w"]
        h = char["h"]

        x1 = int((xc - w / 2) * w_crop)
        y1 = int((yc - h / 2) * h_crop)
        x2 = int((xc + w / 2) * w_crop)
        y2 = int((yc + h / 2) * h_crop)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)
        cv2.putText(
            annotated, char["char"], (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1,
        )
    return annotated


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure and return a logger for the ALPR system.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("alpr")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
