# Model Weights

This directory stores the pre-trained YOLO model weights required by the ALPR system.

## Required Files

| File | Size | Description |
|------|------|-------------|
| `final_best.pt` | ~6 MB | YOLOv8n plate detection model |
| `final_char_yolo.pt` | ~18 MB | YOLOv8s character recognition model (35 classes) |

## Download

These weights are not included in the Git repository due to their file size.

### Option 1: Google Drive
> Download from: [Google Drive Link] _(to be added)_

### Option 2: Train from scratch
See the training section in the main [README.md](../README.md) for instructions on training your own models using the included dataset configuration.

## Training Details

### Plate Detection Model (`final_best.pt`)
- **Architecture**: YOLOv8n (nano)
- **Task**: Object detection (1 class: license plate)
- **Input size**: 640×640
- **Training data**: Vietnamese vehicle images with plate annotations

### Character Recognition Model (`final_char_yolo.pt`)
- **Architecture**: YOLOv8s (small)
- **Task**: Object detection (35 classes: 0-9, A-Z excluding O)
- **Input size**: 640×640
- **Training data**: Cropped license plate images with character-level annotations

## Character Classes (35)

```
Digits: 0 1 2 3 4 5 6 7 8 9
Letters: A B C D E F G H I J K L M N P Q R S T U V W X Y Z
```

> **Note**: The letter 'O' is excluded to avoid confusion with digit '0'.
