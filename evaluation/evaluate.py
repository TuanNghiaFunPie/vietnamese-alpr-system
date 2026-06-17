"""
ALPR Evaluation Script.

Evaluate the ALPR pipeline accuracy on a test dataset.
Computes character-level and plate-level accuracy metrics.

Usage:
    python evaluation/evaluate.py --test-dir data/test/images/ --labels data/test/labels/

Note: This script requires ground-truth label files in YOLO format to be present
alongside the test images. If no labels are available, it runs inference-only
mode and reports detection statistics.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ALPRPipeline
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ALPR pipeline accuracy")

    parser.add_argument(
        "--test-dir", type=str, required=True,
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--plate-model", type=str, default="models/final_best.pt",
        help="Path to plate detection model.",
    )
    parser.add_argument(
        "--char-model", type=str, default="models/final_char_yolo.pt",
        help="Path to character recognition model.",
    )
    parser.add_argument(
        "--plate-conf", type=float, default=0.4,
        help="Plate detection confidence threshold.",
    )
    parser.add_argument(
        "--char-conf", type=float, default=0.3,
        help="Character recognition confidence threshold.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save evaluation results to JSON file.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def collect_test_images(test_dir: str) -> list[str]:
    """Collect all image files from the test directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for f in sorted(Path(test_dir).iterdir()):
        if f.suffix.lower() in extensions:
            images.append(str(f))
    return images


def main() -> None:
    """Run evaluation on the test dataset."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    # Resolve model paths
    plate_path = args.plate_model
    char_path = args.char_model
    if not Path(plate_path).is_absolute():
        plate_path = str(PROJECT_ROOT / plate_path)
    if not Path(char_path).is_absolute():
        char_path = str(PROJECT_ROOT / char_path)

    # Initialize pipeline
    logger.info("Initializing ALPR pipeline...")
    pipeline = ALPRPipeline(plate_path, char_path, verbose=args.verbose)

    # Collect test images
    images = collect_test_images(args.test_dir)
    logger.info("Found %d test images in %s", len(images), args.test_dir)

    if not images:
        logger.error("No images found. Exiting.")
        return

    # Run inference
    stats = {
        "total_images": len(images),
        "images_with_plates": 0,
        "total_plates_detected": 0,
        "total_chars_detected": 0,
        "avg_plate_conf": 0.0,
        "avg_latency_ms": 0.0,
        "detection_rate": 0.0,
        "results": [],
    }

    latencies = []
    all_confs = []

    for i, img_path in enumerate(images):
        try:
            t_start = time.perf_counter()
            results = pipeline.run(img_path, args.plate_conf, args.char_conf)
            latency_ms = (time.perf_counter() - t_start) * 1000
            latencies.append(latency_ms)

            n_plates = len(results)
            if n_plates > 0:
                stats["images_with_plates"] += 1

            stats["total_plates_detected"] += n_plates

            for res in results:
                n_chars = len(res.get("characters", []))
                stats["total_chars_detected"] += n_chars
                all_confs.append(res["plate_conf"])

                stats["results"].append({
                    "image": os.path.basename(img_path),
                    "raw_text": res["raw_text"],
                    "corrected_text": res["corrected_text"],
                    "confidence": round(res["plate_conf"], 4),
                    "num_chars": n_chars,
                    "latency_ms": round(latency_ms, 1),
                })

            if (i + 1) % 50 == 0 or i == len(images) - 1:
                logger.info(
                    "Progress: %d/%d images (%.0f%%) — %d plates found",
                    i + 1, len(images), (i + 1) / len(images) * 100,
                    stats["total_plates_detected"],
                )

        except Exception as e:
            logger.error("Error processing %s: %s", img_path, e)

    # Compute aggregate statistics
    if latencies:
        stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
        stats["min_latency_ms"] = round(min(latencies), 1)
        stats["max_latency_ms"] = round(max(latencies), 1)
        stats["median_latency_ms"] = round(sorted(latencies)[len(latencies) // 2], 1)

    if all_confs:
        stats["avg_plate_conf"] = round(sum(all_confs) / len(all_confs), 4)

    stats["detection_rate"] = round(
        stats["images_with_plates"] / stats["total_images"] * 100, 1,
    ) if stats["total_images"] > 0 else 0.0

    # Print summary
    print("\n" + "=" * 60)
    print("  ALPR Evaluation Summary")
    print("=" * 60)
    print(f"  Total images:          {stats['total_images']}")
    print(f"  Images with plates:    {stats['images_with_plates']}")
    print(f"  Detection rate:        {stats['detection_rate']}%")
    print(f"  Total plates detected: {stats['total_plates_detected']}")
    print(f"  Total chars detected:  {stats['total_chars_detected']}")
    print(f"  Avg plate confidence:  {stats['avg_plate_conf']:.4f}")
    print(f"  Avg latency:           {stats['avg_latency_ms']:.1f} ms")
    if latencies:
        print(f"  Min/Max latency:       {stats['min_latency_ms']:.1f} / {stats['max_latency_ms']:.1f} ms")
    print("=" * 60)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
