# RGB Wood Detection Application

## Description

A comprehensive computer vision application for real-time wood detection using dual-camera setup. The system employs advanced RGB-based detection algorithms combined with shape analysis, morphological operations, and texture analysis to identify wood panels in industrial conveyor belt environments. Features robust detection capabilities with automatic ROI generation, confidence scoring, and real-time visualization.

## Features

- **Dual-Camera Detection**: Simultaneous monitoring from top and bottom perspectives
- **RGB Color Segmentation**: Wood detection using calibrated RGB color profiles for different panel types
- **Contour Detection**: Rectangular shape detection for precise wood plank identification
- **Morphological Operations**: Opening, closing, and dilation operations for noise reduction and mask refinement
- **Texture Analysis**: Standard deviation-based texture analysis for enhanced confidence
- **Automatic ROI Generation**: Dynamic region-of-interest creation based on detected wood
- **Confidence Scoring**: Multi-factor confidence calculation combining color, texture, shape, area, aspect ratio, and solidity
- **Real-time Visualization**: Live overlay of detection results with bounding boxes and metrics
- **Width Measurement**: Pixel-to-millimeter conversion for dimensional analysis
- **Interactive Controls**: Toggle detection, mask view, and frame capture during live operation

## Installation

### Prerequisites

- Python 3.7+
- OpenCV 4.x
- NumPy
- Two USB cameras (configured as video0 and video2)

### Dependencies

Install required packages:

```bash
pip install opencv-python numpy
```

### Setup

1. Clone or download the project files
2. Ensure cameras are connected and accessible
3. Run the application:

```bash
python rgb_wood_detector.py
```

For testing with sample images:

```bash
python test_wood_detection.py
```

## Usage

### Live Detection Mode

Run the main application for real-time wood detection:

```bash
python rgb_wood_detector.py
```

The application will:
- Initialize both cameras at 1280x720 resolution
- Display combined view of top and bottom cameras (resized to 852x480 for optimal viewing on 1080p screens)
- Show detection status and confidence scores
- Save frames to `captured_frames/` directory

### Batch Testing Mode

Process captured images for testing and validation:

```bash
python test_wood_detection.py
```

This will:
- Analyze all images in `captured_frames/` directory
- Generate visualizations in `detection_results/`
- Output detection statistics to console

## Controls

During live detection mode, use these keyboard controls:

- **`q`**: Quit the application
- **`d`**: Toggle detection processing on/off
- **`c`**: Toggle between camera view and detection mask view
- **`s`**: Save current frames to `captured_frames/` directory

## Configuration

### RGB Color Profiles

The system uses calibrated RGB ranges for different wood panel types:

**Top Panel Profile:**
- Lower bound: [169, 180, 176] (BGR)
- Upper bound: [211, 219, 210] (BGR)

**Bottom Panel Profile:**
- Lower bound: [134, 105, 109] (BGR)
- Upper bound: [207, 176, 183] (BGR)

### Pixel-to-MM Conversion

Width measurements use calibrated conversion factors:
- Top camera: 2.96 pixels per mm (at 31cm distance)
- Bottom camera: 3.18 pixels per mm
- Calibrate by measuring known distances in captured frames

### Camera Settings

Optimized camera parameters for consistent detection:
- Resolution: 1280x720 (720p)
- Top camera: brightness=0, contrast=32, saturation=64, hue=0, exposure=-6, white_balance=4520, gain=0
- Bottom camera: brightness=85, contrast=125, saturation=125, hue=0, exposure=-6, white_balance=4850, gain=0, sharpness=200, backlight_compensation=1

### Detection Parameters

- Minimum contour area: 2000 pixels
- Maximum contour area: 500000 pixels
- Aspect ratio range: 1.0 - 10.0
- Contour approximation: 0.025
- Morphological kernel size: 11
- Closing iterations: 3
- Opening iterations: 2

### Detection ROIs

Detection processing uses predefined regions of interest for optimal performance:
- Top camera: 100% width, 70% height (centered vertically)
- Bottom camera: 80% width, 80% height (centered)

## Technical Details

### Detection Pipeline

1. **Color Segmentation**: RGB-based masking with histogram equalization on V-channel
2. **Morphological Processing**: Closing, dilation, and opening operations for noise reduction
3. **Contour Analysis**: External contour detection with area, size, and aspect ratio filtering
4. **Shape Validation**: Polygon approximation with solidity and vertex count scoring
5. **Texture Analysis**: Local standard deviation calculation for wood texture validation
6. **Confidence Calculation**: Weighted scoring combining color (50%), texture (30%), and shape (20%) confidences

### Key Improvements

- **RGB Color Profiles**: Calibrated ranges for top and bottom panel types
- **Histogram Equalization**: Enhanced contrast for better color detection
- **Multi-scale Morphological Operations**: Configurable kernel sizes and iterations
- **Rectangular Contour Detection**: Precise shape filtering for wood planks
- **Automatic ROI Generation**: 10% padding around detected regions
- **Texture Confidence**: Standard deviation-based analysis for wood grain detection
- **Width Measurement**: Real-time mm conversion using calibrated pixel ratios

### Architecture

- **ColorWoodDetector Class**: Core detection algorithms and parameters
- **CameraHandler Class**: Camera initialization and configuration management
- **Detection Methods**:
  - `detect_wood_comprehensive()`: Main pipeline combining color, shape, and texture analysis
  - `detect_wood_by_color()`: RGB-based color segmentation
  - `detect_rectangular_contours()`: Contour filtering and validation
  - `visualize_detection()`: Result overlay and annotation
  - `_detect_wood_by_texture()`: Texture analysis using local variance

### Performance Characteristics

- **Resolution**: 1280x720 per camera
- **Frame Rate**: Real-time processing
- **Detection Range**: 2000-500000 pixels contour area
- **Aspect Ratio**: 1.0-10.0 for valid wood shapes
- **Confidence Threshold**: 0.0-1.0 with multi-factor scoring

### Output Information

- **Detection Results**: Boolean wood_detected flag, wood_count, confidence score (0.0-1.0)
- **Measurements**: Width in mm for detected wood (using bounding box dimensions)
- **Visualizations**: Bounding boxes, confidence labels, ROI overlays
- **Masks**: Binary masks showing detected regions
- **Console Output**: Real-time statistics every 30 frames with status and measurements

### Error Handling

- Camera initialization validation
- Image loading error detection
- Morphological operation fallbacks
- Exception handling in all detection methods