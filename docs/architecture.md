# System Architecture

## Pipeline Overview

The Vietnamese ALPR system uses a **two-stage detection pipeline** followed by **rule-based post-processing**:

```
Input Image → [Stage 1: Plate Detection] → [Stage 2: Character Recognition] → [Post-processing] → Output
```

## Detailed Architecture

```mermaid
flowchart TD
    A["Input Image<br/>(Full vehicle photo)"] --> B["Stage 1: Plate Detection<br/>(YOLOv8n)"]
    B --> C{"Plates found?"}
    C -->|No| D["Return empty results"]
    C -->|Yes| E["Crop plate regions"]
    E --> F["Stage 2: Character Recognition<br/>(YOLOv8s, 35 classes)"]
    F --> G["Row Grouping<br/>(Y-axis clustering)"]
    G --> H["Left-to-Right Sorting<br/>(X-axis ordering)"]
    H --> I["Vietnamese Format Correction<br/>(Rule-based)"]
    I --> J["Formatted Output<br/>(e.g., 29A-123.45)"]
```

## Stage 1: Plate Detection

**Model**: YOLOv8n (nano variant — optimized for speed)

- **Input**: Full vehicle image (any resolution)
- **Output**: Bounding boxes of detected license plates
- **Classes**: 1 (license_plate)
- **Inference**: ~10-30ms per image on GPU

The detector locates rectangular plate regions in the image and returns their coordinates with confidence scores. Each detected region is then cropped for the next stage.

## Stage 2: Character Recognition

**Model**: YOLOv8s (small variant — balanced speed/accuracy)

- **Input**: Cropped license plate image
- **Output**: Bounding boxes and class labels for each character
- **Classes**: 35 (digits 0-9 + letters A-Z, excluding 'O')
- **Inference**: ~15-40ms per plate crop on GPU

Each character is detected with its position (center coordinates, width, height) and predicted class. The position information is critical for the sorting stage.

## Post-processing Pipeline

### Step 1: Row Grouping (Y-axis clustering)

Vietnamese plates can be:
- **Single-row** (cars): `29A12345`
- **Two-row** (motorcycles): Line 1: `29A`, Line 2: `12345`

Characters are grouped into rows by comparing their Y-center coordinates. If the Y-distance between two characters exceeds 25% of the plate height, they are placed in separate rows.

### Step 2: Left-to-Right Sorting (X-axis)

Within each row, characters are sorted by their X-center coordinate to produce the correct reading order.

### Step 3: Vietnamese Format Correction

OCR models can confuse visually similar characters (e.g., 'O' ↔ '0', 'B' ↔ '8'). The correction module enforces Vietnamese civilian plate structure:

| Position | Type | Example |
|----------|------|---------|
| 0-1 | Digits (province code) | `29` |
| 2 | Letter (series) | `A` |
| 3 | Letter or digit | `1` or `A` |
| 4+ | Digits (registration) | `12345` |

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant PD as Plate Detector
    participant CR as Char Recognizer
    participant PP as Post-processor

    U->>P: process_image(image)
    P->>PD: detect_plates(image, conf=0.4)
    PD-->>P: plates: [{crop, bbox, conf}]
    
    loop For each plate
        P->>CR: detect_characters(crop, conf=0.3)
        CR-->>P: chars: [{char, x, y, w, h, conf}]
        P->>PP: sort_characters(chars)
        PP-->>P: raw_text: "29A-12345"
        P->>PP: apply_vietnamese_rules(raw_text)
        PP-->>P: corrected: "29A-123.45"
    end
    
    P-->>U: results: [{bbox, raw, corrected, timing}]
```

## Performance Characteristics

| Component | Typical Latency | Hardware |
|-----------|----------------|----------|
| Plate Detection | 10-30 ms | GPU (CUDA) |
| Character Recognition | 15-40 ms | GPU (CUDA) |
| Post-processing | < 1 ms | CPU |
| **Total Pipeline** | **~30-70 ms** | GPU |

> CPU-only inference is approximately 5-10x slower.
