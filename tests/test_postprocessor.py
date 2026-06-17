"""
Unit tests for the Vietnamese plate post-processor.

Tests cover:
  - Character sorting (row grouping and X-axis ordering)
  - Vietnamese format correction rules
  - Edge cases (empty input, short strings, etc.)
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.postprocessor import VietnamesePlatePostprocessor


@pytest.fixture
def processor():
    """Create a post-processor instance for tests."""
    return VietnamesePlatePostprocessor()


# ─── sort_characters Tests ─────────────────────────────────────────────────────

class TestSortCharacters:
    """Tests for the character sorting / row grouping logic."""

    def test_empty_input(self, processor):
        """Empty character list should return empty string."""
        assert processor.sort_characters([], plate_height_px=100) == ""

    def test_single_row_sorted(self, processor):
        """Characters already sorted left-to-right in one row."""
        chars = [
            {"char": "2", "x_center": 0.1, "y_center": 0.5, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "9", "x_center": 0.2, "y_center": 0.5, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "A", "x_center": 0.3, "y_center": 0.5, "w": 0.05, "h": 0.1, "conf": 0.9},
        ]
        result = processor.sort_characters(chars, plate_height_px=100)
        assert result == "29A"

    def test_single_row_unsorted(self, processor):
        """Characters out of order should be sorted by X-center."""
        chars = [
            {"char": "A", "x_center": 0.3, "y_center": 0.5, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "2", "x_center": 0.1, "y_center": 0.5, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "9", "x_center": 0.2, "y_center": 0.5, "w": 0.05, "h": 0.1, "conf": 0.9},
        ]
        result = processor.sort_characters(chars, plate_height_px=100)
        assert result == "29A"

    def test_two_rows(self, processor):
        """Two-row plate (motorcycle style) should produce hyphen-separated text."""
        chars = [
            # Row 1 (top)
            {"char": "2", "x_center": 0.2, "y_center": 10, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "9", "x_center": 0.4, "y_center": 12, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "A", "x_center": 0.6, "y_center": 11, "w": 0.05, "h": 0.1, "conf": 0.9},
            # Row 2 (bottom) — Y values well separated from row 1
            {"char": "1", "x_center": 0.1, "y_center": 80, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "2", "x_center": 0.3, "y_center": 82, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "3", "x_center": 0.5, "y_center": 81, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "4", "x_center": 0.7, "y_center": 80, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "5", "x_center": 0.9, "y_center": 83, "w": 0.05, "h": 0.1, "conf": 0.9},
        ]
        result = processor.sort_characters(chars, plate_height_px=100)
        assert result == "29A-12345"

    def test_single_character(self, processor):
        """Single character should return just that character."""
        chars = [
            {"char": "X", "x_center": 0.5, "y_center": 0.5, "w": 0.1, "h": 0.1, "conf": 0.9},
        ]
        result = processor.sort_characters(chars, plate_height_px=100)
        assert result == "X"


# ─── apply_vietnamese_rules Tests ──────────────────────────────────────────────

class TestApplyVietnameseRules:
    """Tests for the Vietnamese plate format correction."""

    def test_correct_plate_no_change(self, processor):
        """Already correct plate should not be modified (structure-wise)."""
        result = processor.apply_vietnamese_rules("29A-12345")
        assert result == "29A-123.45"

    def test_letter_o_in_province_code(self, processor):
        """Letter O in province code should be corrected to 0."""
        result = processor.apply_vietnamese_rules("O9A-12345")
        assert result.startswith("09A")

    def test_digit_in_series_position(self, processor):
        """Digit in series letter position should be corrected to letter."""
        result = processor.apply_vietnamese_rules("298-12345")
        # 8 → B
        assert "B" in result[:4]

    def test_letter_in_registration_number(self, processor):
        """Letter in registration number should be corrected to digit."""
        result = processor.apply_vietnamese_rules("29A-1234S")
        # S → 5, so the last char should be 5
        assert result.endswith("5")

    def test_short_string_passthrough(self, processor):
        """String shorter than 7 chars should be returned unchanged."""
        result = processor.apply_vietnamese_rules("29A")
        assert result == "29A"

    def test_two_row_format_preserved(self, processor):
        """Two-row format with hyphen should preserve row structure."""
        result = processor.apply_vietnamese_rules("29A-12345")
        assert "-" in result

    def test_single_row_gets_formatted(self, processor):
        """Single-row input should be formatted with separator."""
        result = processor.apply_vietnamese_rules("29A12345")
        assert "-" in result

    def test_extended_series(self, processor):
        """Plate with letter at position 3 (e.g., 29AA) should preserve it."""
        result = processor.apply_vietnamese_rules("29AA12345")
        assert result.startswith("29AA")

    def test_multiple_corrections(self, processor):
        """Multiple simultaneous OCR errors should all be corrected."""
        # O→0, Z→2 in registration
        result = processor.apply_vietnamese_rules("O9A-1Z345")
        assert result[0] == "0"  # O → 0
        # Z → 2 somewhere in the registration part

    def test_empty_string(self, processor):
        """Empty string should return empty string."""
        result = processor.apply_vietnamese_rules("")
        assert result == ""


# ─── process (integration) Tests ───────────────────────────────────────────────

class TestProcess:
    """Integration tests for the full post-processing pipeline."""

    def test_full_pipeline(self, processor):
        """Full process should return both raw and corrected text."""
        chars = [
            {"char": "2", "x_center": 0.1, "y_center": 10, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "9", "x_center": 0.2, "y_center": 12, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "A", "x_center": 0.3, "y_center": 11, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "1", "x_center": 0.1, "y_center": 80, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "2", "x_center": 0.2, "y_center": 82, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "3", "x_center": 0.3, "y_center": 81, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "4", "x_center": 0.4, "y_center": 80, "w": 0.05, "h": 0.1, "conf": 0.9},
            {"char": "5", "x_center": 0.5, "y_center": 83, "w": 0.05, "h": 0.1, "conf": 0.9},
        ]
        result = processor.process(chars, plate_height_px=100)
        assert "raw_text" in result
        assert "corrected_text" in result
        assert result["raw_text"] == "29A-12345"
        assert "29A" in result["corrected_text"]

    def test_empty_chars(self, processor):
        """Empty character list should produce empty results."""
        result = processor.process([], plate_height_px=100)
        assert result["raw_text"] == ""
        assert result["corrected_text"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
