# Vietnamese License Plate Format & Correction Rules

## Overview

This document describes the Vietnamese civilian license plate format and the rule-based correction logic used by the ALPR post-processor to fix common OCR misreading errors.

## Vietnamese Civilian Plate Format

### Structure

Vietnamese civilian (white/yellow) license plates follow a specific format:

```
[Province Code][Series Letter][Sub-series][Registration Number]
   2 digits      1 letter     0-1 char       5 digits

Example: 29A1-123.45
         ││││  │││││
         │││└──┘│││└─ Registration digits (5)
         ││└────┘││
         │└──────┘└── Sub-series (optional, can be letter or digit)
         └────────── Series letter
         └────────── Province code (2 digits)
```

### Plate Types

| Type | Format | Example | Notes |
|------|--------|---------|-------|
| **Car (1-row)** | `DDLDDDDDD` | `29A12345` | Single horizontal line |
| **Motorcycle (2-row)** | `DDL` / `DDDDD` | `29A` / `12345` | Two stacked lines |
| **Extended series** | `DDLLDDDDDD` | `29AA12345` | When series overflows |

Where: `D` = Digit, `L` = Letter

### Province Codes (Selected)

| Code | Province/City |
|------|---------------|
| 29, 30, 31, 32, 33 | Hà Nội |
| 41 | Hải Phòng |
| 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 | TP. Hồ Chí Minh |
| 43 | Đà Nẵng |
| 36 | Thanh Hóa |
| 34 | Hải Dương |
| 37 | Nghệ An |

## OCR Confusion Pairs

YOLOv8 character detection can confuse visually similar characters. The most common confusion pairs:

| Intended | Misread As | Direction |
|----------|-----------|-----------|
| `0` (zero) | `O`, `D` | Digit → Letter |
| `1` (one) | `I`, `L` | Digit → Letter |
| `2` (two) | `Z` | Digit → Letter |
| `5` (five) | `S` | Digit → Letter |
| `8` (eight) | `B` | Digit → Letter |
| `6` (six) | `G` | Digit → Letter |
| `4` (four) | `A` | Digit → Letter |

## Correction Rules

The post-processor applies positional rules based on the known plate structure:

### Rule 1: Province Code (positions 0-1) → Must be DIGITS
```python
# If a letter is found at positions 0-1, convert to digit
# Example: "O9A12345" → "09A12345"  (O → 0)
```

### Rule 2: Series Letter (position 2) → Must be LETTER
```python
# If a digit is found at position 2, convert to letter
# Example: "298I2345" → "29BI2345"  (8 → B)
```

### Rule 3: Sub-series (position 3) → Keep as-is
```python
# Position 3 can be either letter or digit
# Example: "29A1..." or "29AA..." — both valid
```

### Rule 4: Registration Number (positions 4+) → Must be DIGITS
```python
# If letters are found at positions 4+, convert to digits
# Example: "29A1234S" → "29A12345"  (S → 5)
```

## Output Formatting

After correction, plates are formatted with visual separators:

### Two-row plates (motorcycle)
```
Input:  "29A" + "12345"
Output: "29A-123.45"
```

### Single-row plates (car)
```
Input:  "29A12345"
Output: "29A-123.45"

Input:  "29AA12345"  
Output: "29AA-123.45"
```

## Limitations

1. **Military / Government plates**: Different format (red/blue plates), not currently handled
2. **Diplomatic plates**: White text on blue background, different structure
3. **Temporary plates**: Paper plates, not reliably detectable
4. **Severely damaged plates**: Physical damage reduces both detection and recognition accuracy
5. **Non-standard fonts**: Some aftermarket plates use non-standard character spacing
