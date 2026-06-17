"""
Vietnamese License Plate Post-processing Module.

Handles character sorting (row grouping by Y-axis, left-to-right by X-axis)
and rule-based error correction following Vietnamese civilian plate formats.

Vietnamese civilian plate structure:
  - Positions 0-1: Province code (digits)
  - Position 2: Series letter (letter)
  - Position 3: Sub-series (letter or digit, kept as-is)
  - Positions 4+: Registration number (digits)

Examples: 29A-123.45, 30E1-567.89, 51F-038.92
"""

from typing import Optional


class VietnamesePlatePostprocessor:
    """Post-processor for Vietnamese license plate OCR results."""

    # Common OCR confusion mappings
    LETTER_TO_DIGIT = {
        "O": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
        "S": "5", "B": "8", "G": "6", "A": "4",
    }
    DIGIT_TO_LETTER = {
        "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
        "6": "G", "4": "A",
    }

    def sort_characters(
        self,
        chars: list[dict],
        plate_height_px: int,
        y_threshold_ratio: float = 0.25,
    ) -> str:
        """
        Group detected characters into rows and sort left-to-right.

        Characters are clustered into rows based on their Y-center proximity,
        then each row is sorted by X-center. Rows are joined with '-' to
        represent multi-line plates.

        Args:
            chars: List of character dicts with 'char', 'x_center', 'y_center' keys.
            plate_height_px: Height of the plate crop in pixels.
            y_threshold_ratio: Fraction of plate height used as row-split threshold.

        Returns:
            Sorted plate string (e.g. '29A-12345' for two-row plates).
        """
        if not chars:
            return ""

        y_threshold = y_threshold_ratio * plate_height_px

        # Sort by Y-center first
        chars_by_y = sorted(chars, key=lambda c: c["y_center"])

        rows: list[list[dict]] = []
        for char in chars_by_y:
            if not rows:
                rows.append([char])
            else:
                # Compare with average Y of the current last row
                avg_y = sum(c["y_center"] for c in rows[-1]) / len(rows[-1])
                if abs(char["y_center"] - avg_y) < y_threshold:
                    rows[-1].append(char)
                else:
                    rows.append([char])

        # Sort rows top-to-bottom, characters left-to-right within each row
        rows.sort(key=lambda r: sum(c["y_center"] for c in r) / len(r))

        row_texts = []
        for row in rows:
            sorted_row = sorted(row, key=lambda c: c["x_center"])
            row_texts.append("".join(c["char"] for c in sorted_row))

        return "-".join(row_texts)

    def apply_vietnamese_rules(self, raw_plate: str) -> str:
        """
        Apply Vietnamese civilian plate format rules to correct OCR errors.

        Rules enforced:
          - Positions 0-1: Must be digits (province code)
          - Position 2: Must be a letter (series)
          - Position 3: Keep as-is (can be letter or digit)
          - Positions 4+: Must be digits (registration number)

        The output is formatted with separators for readability:
          - Two-row plates: 'XXY-ZZZ.ZZ'
          - One-row plates: 'XXY-ZZZ.ZZ' or 'XXYY-ZZZ.ZZ'

        Args:
            raw_plate: Raw OCR string, possibly with '-' row separator.

        Returns:
            Corrected and formatted plate string.
        """
        clean = raw_plate.replace("-", "")
        n = len(clean)

        if n < 7:
            # Too short to match any valid Vietnamese plate format
            return raw_plate

        corrected = list(clean)

        # Rule 1: First two characters must be digits (province code)
        for i in range(2):
            if not corrected[i].isdigit():
                corrected[i] = self.LETTER_TO_DIGIT.get(corrected[i], corrected[i])

        # Rule 2: Third character must be a letter (series)
        if not corrected[2].isalpha():
            corrected[2] = self.DIGIT_TO_LETTER.get(corrected[2], corrected[2])

        # Rule 3: Fourth character can be letter or digit — keep as-is

        # Rule 4: Positions 4 onwards must be digits (registration number)
        for i in range(4, n):
            if not corrected[i].isdigit():
                corrected[i] = self.LETTER_TO_DIGIT.get(corrected[i], corrected[i])

        corrected_str = "".join(corrected)

        # Re-format with visual separators
        if "-" in raw_plate:
            # Preserve the original row split position
            parts = raw_plate.split("-")
            split_pos = len(parts[0])
            line1 = corrected_str[:split_pos]
            line2 = corrected_str[split_pos:]
            if len(line2) == 5:
                line2 = f"{line2[:3]}.{line2[3:]}"
            return f"{line1}-{line2}"
        else:
            # Single-row plate: split after series letter(s)
            split_idx = 3 if (len(clean) > 3 and clean[3].isdigit()) else 4
            part1 = corrected_str[:split_idx]
            part2 = corrected_str[split_idx:]
            if len(part2) == 5:
                part2 = f"{part2[:3]}.{part2[3:]}"
            return f"{part1}-{part2}"

    def process(
        self,
        chars: list[dict],
        plate_height_px: int,
        y_threshold_ratio: float = 0.25,
    ) -> dict:
        """
        Run the full post-processing pipeline: sort then correct.

        Args:
            chars: Detected character dicts.
            plate_height_px: Plate crop height in pixels.
            y_threshold_ratio: Row-split threshold ratio.

        Returns:
            Dict with 'raw_text' and 'corrected_text' keys.
        """
        raw_text = self.sort_characters(chars, plate_height_px, y_threshold_ratio)
        corrected_text = self.apply_vietnamese_rules(raw_text)
        return {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
        }
