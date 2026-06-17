"""
ALPR Command-Line Interface.

Run Vietnamese license plate recognition on single images or batch directories.

Usage:
    # Single image
    python app_cli.py --image path/to/car.jpg

    # Batch processing
    python app_cli.py --dir path/to/images/ --output results.json

    # With custom thresholds
    python app_cli.py --image car.jpg --plate-conf 0.5 --char-conf 0.4

    # Save annotated images
    python app_cli.py --image car.jpg --save-annotated output/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import cv2

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ALPRPipeline
from src.utils import (
    resolve_model_path,
    load_image,
    draw_plate_boxes,
    draw_char_boxes,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vietnamese ALPR — Automatic License Plate Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_cli.py --image photo.jpg
  python app_cli.py --dir ./test_images/ --format json --output results.json
  python app_cli.py --image photo.jpg --save-annotated ./output/
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Path to a single input image.",
    )
    input_group.add_argument(
        "--dir", type=str,
        help="Path to a directory of images for batch processing.",
    )

    # Model options
    parser.add_argument(
        "--plate-model", type=str, default="models/final_best.pt",
        help="Path to plate detection model (default: models/final_best.pt).",
    )
    parser.add_argument(
        "--char-model", type=str, default="models/final_char_yolo.pt",
        help="Path to character recognition model (default: models/final_char_yolo.pt).",
    )

    # Threshold options
    parser.add_argument(
        "--plate-conf", type=float, default=0.4,
        help="Plate detection confidence threshold (default: 0.4).",
    )
    parser.add_argument(
        "--char-conf", type=float, default=0.3,
        help="Character recognition confidence threshold (default: 0.3).",
    )

    # Output options
    parser.add_argument(
        "--format", type=str, choices=["text", "json", "csv"], default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to file (for json/csv formats).",
    )
    parser.add_argument(
        "--save-annotated", type=str, default=None,
        help="Directory to save annotated images with bounding boxes.",
    )

    # Misc
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose (debug) logging.",
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple:
    """Resolve model paths, making them absolute if relative."""
    plate_path = Path(args.plate_model)
    char_path = Path(args.char_model)

    # If relative, resolve from project root
    if not plate_path.is_absolute():
        plate_path = PROJECT_ROOT / plate_path
    if not char_path.is_absolute():
        char_path = PROJECT_ROOT / char_path

    return str(plate_path), str(char_path)


def collect_images(args: argparse.Namespace) -> list[str]:
    """Collect image paths from --image or --dir argument."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if args.image:
        return [args.image]

    images = []
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"Error: Directory not found: {args.dir}", file=sys.stderr)
        sys.exit(1)

    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in extensions:
            images.append(str(f))

    if not images:
        print(f"Warning: No images found in {args.dir}", file=sys.stderr)

    return images


def format_text_output(image_path: str, results: list[dict]) -> str:
    """Format results as human-readable text."""
    lines = [f"\n{'='*60}"]
    lines.append(f"  Image: {os.path.basename(image_path)}")
    lines.append(f"{'='*60}")

    if not results:
        lines.append("  No license plates detected.")
        return "\n".join(lines)

    for idx, res in enumerate(results):
        lines.append(f"\n  Plate #{idx + 1}:")
        lines.append(f"    Bounding Box : {res['bbox']}")
        lines.append(f"    Confidence   : {res['plate_conf']:.3f}")
        lines.append(f"    Raw OCR      : {res['raw_text'] or '[empty]'}")
        lines.append(f"    Corrected    : {res['corrected_text'] or '[empty]'}")
        lines.append(f"    Latency      : plate={res['timing']['plate_ms']:.1f}ms, "
                      f"char={res['timing']['char_ms']:.1f}ms")

    return "\n".join(lines)


def format_json_record(image_path: str, results: list[dict]) -> dict:
    """Format results as a JSON-serializable dict."""
    plates = []
    for res in results:
        plates.append({
            "bbox": res["bbox"],
            "confidence": round(res["plate_conf"], 4),
            "raw_text": res["raw_text"],
            "corrected_text": res["corrected_text"],
            "timing_ms": res["timing"],
        })
    return {
        "image": os.path.basename(image_path),
        "num_plates": len(plates),
        "plates": plates,
    }


def save_annotated_image(
    image_path: str,
    results: list[dict],
    output_dir: str,
) -> None:
    """Save image with plate bounding boxes drawn."""
    image = load_image(image_path)

    # Draw plate boxes
    plates_for_draw = [
        {"bbox": r["bbox"], "conf": r["plate_conf"]}
        for r in results
    ]
    annotated = draw_plate_boxes(image, plates_for_draw)

    # Add corrected text labels
    for idx, res in enumerate(results):
        x1, y1, x2, y2 = res["bbox"]
        label = res["corrected_text"] or res["raw_text"] or "?"
        cv2.putText(
            annotated, label, (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, annotated)


def main() -> None:
    """Main entry point for the CLI application."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    # Resolve model paths
    plate_model_path, char_model_path = resolve_paths(args)

    # Initialize pipeline
    try:
        pipeline = ALPRPipeline(
            plate_model_path, char_model_path,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Download model weights — see models/README.md for instructions.", file=sys.stderr)
        sys.exit(1)

    # Collect images
    image_paths = collect_images(args)
    logger.info("Processing %d image(s)...", len(image_paths))

    all_json_records = []
    csv_lines = ["image,plate_index,raw_text,corrected_text,confidence,bbox"]

    t_total_start = time.perf_counter()

    for img_path in image_paths:
        try:
            results = pipeline.run(img_path, args.plate_conf, args.char_conf)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Skipping %s: %s", img_path, e)
            continue

        # Text output (always printed to stdout)
        if args.format == "text":
            print(format_text_output(img_path, results))

        # JSON accumulation
        if args.format == "json":
            record = format_json_record(img_path, results)
            all_json_records.append(record)
            if not args.output:
                print(json.dumps(record, indent=2, ensure_ascii=False))

        # CSV accumulation
        if args.format == "csv":
            for idx, res in enumerate(results):
                bbox_str = f"{res['bbox']}"
                csv_lines.append(
                    f"{os.path.basename(img_path)},{idx+1},"
                    f"{res['raw_text']},{res['corrected_text']},"
                    f"{res['plate_conf']:.4f},\"{bbox_str}\""
                )

        # Save annotated image
        if args.save_annotated:
            save_annotated_image(img_path, results, args.save_annotated)

    t_total_ms = (time.perf_counter() - t_total_start) * 1000

    # Save to file if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        if args.format == "json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_json_records, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        elif args.format == "csv":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(csv_lines))
            print(f"Results saved to {args.output}")

    logger.info("Done. Total time: %.1f ms (%d images)", t_total_ms, len(image_paths))


if __name__ == "__main__":
    main()
