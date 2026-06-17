"""
ALPR Pipeline — Two-stage Vietnamese License Plate Recognition.

This module provides the core `ALPRPipeline` class that orchestrates:
  1. Plate detection using a YOLOv8 object detection model.
  2. Character recognition using a second YOLOv8 model (35 classes).
  3. Post-processing: row grouping + Vietnamese format correction.

Usage:
    >>> from src.pipeline import ALPRPipeline
    >>> pipeline = ALPRPipeline("models/final_best.pt", "models/final_char_yolo.pt")
    >>> results = pipeline.run("path/to/car_image.jpg")
"""

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.postprocessor import VietnamesePlatePostprocessor
from src.utils import load_image, setup_logging


class ALPRPipeline:
    """
    Automatic License Plate Recognition pipeline for Vietnamese plates.

    Two-stage approach:
      - Stage 1: YOLOv8n detects license plate regions in the full image.
      - Stage 2: YOLOv8s recognizes individual characters within each plate crop.
      - Post-processing: Characters are grouped into rows and corrected
        according to Vietnamese civilian plate formatting rules.

    Attributes:
        plate_model: YOLO model for plate detection.
        char_model: YOLO model for character recognition.
        postprocessor: Vietnamese plate post-processing logic.
        logger: Logger instance for this pipeline.
    """

    def __init__(
        self,
        plate_model_path: Union[str, Path],
        char_model_path: Union[str, Path],
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ALPR pipeline with pre-trained YOLO models.

        Args:
            plate_model_path: Path to the YOLOv8 plate detection weights (.pt).
            char_model_path: Path to the YOLOv8 character recognition weights (.pt).
            verbose: Enable verbose logging output.

        Raises:
            FileNotFoundError: If either model file does not exist.
        """
        self.logger = setup_logging(verbose)

        plate_path = Path(plate_model_path)
        char_path = Path(char_model_path)

        if not plate_path.exists():
            raise FileNotFoundError(f"Plate model not found: {plate_path}")
        if not char_path.exists():
            raise FileNotFoundError(f"Char model not found: {char_path}")

        self.logger.info("Loading plate detection model: %s", plate_path.name)
        self.plate_model = YOLO(str(plate_path))

        self.logger.info("Loading character recognition model: %s", char_path.name)
        self.char_model = YOLO(str(char_path))

        self.postprocessor = VietnamesePlatePostprocessor()
        self.logger.info("ALPR pipeline initialized successfully.")

    def detect_plates(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.4,
    ) -> list[dict]:
        """
        Detect license plate regions in an image.

        Args:
            image: BGR image as numpy array.
            conf_threshold: Minimum confidence score for plate detections.

        Returns:
            List of dicts, each containing:
              - 'crop': BGR plate crop (numpy array)
              - 'bbox': [x1, y1, x2, y2] pixel coordinates
              - 'conf': Detection confidence score
        """
        results = self.plate_model(image, conf=conf_threshold, verbose=False)
        plates = []

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = image[y1:y2, x1:x2]

                plates.append({
                    "crop": crop,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                })

        self.logger.debug("Detected %d plate(s) at conf >= %.2f", len(plates), conf_threshold)
        return plates

    def detect_characters(
        self,
        plate_crop: np.ndarray,
        conf_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Recognize characters within a license plate crop.

        Args:
            plate_crop: BGR plate crop image.
            conf_threshold: Minimum confidence for character detections.

        Returns:
            List of dicts, each containing:
              - 'char': Predicted character string
              - 'x_center', 'y_center': Normalized center coordinates
              - 'w', 'h': Normalized width and height
              - 'conf': Detection confidence score
        """
        if plate_crop.size == 0:
            return []

        results = self.char_model(plate_crop, conf=conf_threshold, verbose=False)
        detected = []

        h_crop, w_crop = plate_crop.shape[:2]

        for result in results:
            for box in result.boxes:
                xywh = box.xywh[0].cpu().numpy()
                x_center, y_center, w, h = xywh

                class_id = int(box.cls[0].cpu().numpy())
                char_val = self.char_model.names[class_id]
                conf = float(box.conf[0].cpu().numpy())

                # Normalize coordinates to [0, 1] range
                detected.append({
                    "char": char_val,
                    "x_center": x_center / w_crop,
                    "y_center": y_center / h_crop,
                    "w": w / w_crop,
                    "h": h / h_crop,
                    "conf": conf,
                })

        self.logger.debug("Detected %d character(s) at conf >= %.2f", len(detected), conf_threshold)
        return detected

    def process_image(
        self,
        image: np.ndarray,
        conf_plate: float = 0.4,
        conf_char: float = 0.3,
    ) -> list[dict]:
        """
        Run the full ALPR pipeline on a single image.

        Args:
            image: BGR image as numpy array.
            conf_plate: Plate detection confidence threshold.
            conf_char: Character recognition confidence threshold.

        Returns:
            List of result dicts, each containing:
              - 'bbox': [x1, y1, x2, y2] plate bounding box
              - 'crop': BGR plate crop
              - 'characters': List of detected character dicts
              - 'raw_text': Raw OCR output string
              - 'corrected_text': Post-processed plate string
              - 'plate_conf': Plate detection confidence
              - 'timing': Dict with 'plate_ms' and 'char_ms' latency
        """
        results = []

        t_plate_start = time.perf_counter()
        plates = self.detect_plates(image, conf_threshold=conf_plate)
        t_plate_ms = (time.perf_counter() - t_plate_start) * 1000

        for plate in plates:
            crop = plate["crop"]
            plate_h = crop.shape[0]

            t_char_start = time.perf_counter()
            chars = self.detect_characters(crop, conf_threshold=conf_char)
            t_char_ms = (time.perf_counter() - t_char_start) * 1000

            post_result = self.postprocessor.process(chars, plate_h)

            results.append({
                "bbox": plate["bbox"],
                "crop": crop,
                "characters": chars,
                "raw_text": post_result["raw_text"],
                "corrected_text": post_result["corrected_text"],
                "plate_conf": plate["conf"],
                "timing": {
                    "plate_ms": round(t_plate_ms, 1),
                    "char_ms": round(t_char_ms, 1),
                },
            })

        return results

    def run(
        self,
        image_path: str,
        conf_plate: float = 0.4,
        conf_char: float = 0.3,
    ) -> list[dict]:
        """
        Run the full pipeline on an image file path.

        Convenience wrapper that loads the image then calls process_image().

        Args:
            image_path: Path to the input image.
            conf_plate: Plate detection confidence threshold.
            conf_char: Character recognition confidence threshold.

        Returns:
            List of result dicts (see process_image for schema).

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If image cannot be decoded.
        """
        image = load_image(image_path)
        self.logger.info("Processing: %s", image_path)
        return self.process_image(image, conf_plate, conf_char)
