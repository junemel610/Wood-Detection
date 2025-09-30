# Wood Detection Application

## Description

A comprehensive computer vision application for real-time wood detection using dual-camera setup. The system employs advanced color-based detection algorithms combined with shape analysis to identify wood panels in industrial conveyor belt environments. Features robust detection capabilities with automatic ROI generation, confidence scoring, and real-time visualization.

## Features

- **Dual-Camera Detection**: Simultaneous monitoring from top and bottom perspectives
- **HSV Color Analysis**: Wood detection using calibrated color profiles for different panel types
- **Contour Detection**: 4-point rectangle detection for precise wood plank identification
- **Automatic ROI Generation**: Dynamic region-of-interest creation based on detected wood
- **Confidence Scoring**: Multi-factor confidence calculation combining area, aspect ratio, and solidity
- **Real-time Visualization**: Live overlay of detection results with bounding boxes and metrics
- **Width Measurement**: Pixel-to-millimeter conversion for dimensional analysis
- **Batch Processing**: Test mode for processing captured image sets
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
python color_wood_detector.py
```

For testing with sample images:

```bash
python test_wood_detection.py
```

## Usage

### Live Detection Mode

Run the main application for real-time wood detection:

```bash
python color_wood_detector.py
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

## Calibration

### HSV Color Profiles

The system uses calibrated HSV ranges for different wood panel types:

**Top Panel Profile:**
- Lower bound: [85, 10, 100]
- Upper bound: [125, 40, 240]

**Bottom Panel Profile:**
- Lower bound: [72, 3, 134]
- Upper bound: [75, 4, 135]

### Pixel-to-MM Conversion

Width measurements use calibrated conversion factors:
- Top camera: 2.96 pixels per mm (at 31cm distance)
- Bottom camera: 3.18 pixels per mm
- Calibrate by measuring known distances in captured frames

### Camera Settings

Optimized camera parameters for consistent detection:
- Resolution: 1280x720 (720p)
- Top camera: brightness=0, contrast=32, saturation=64
- Bottom camera: brightness=135, contrast=75, saturation=155

### Detection ROIs

Detection processing uses predefined regions of interest for optimal performance:
- Top camera: 100% width, 80% height (centered vertically)
- Bottom camera: 80% width, 80% height (centered)

## Technical Details

### Detection Pipeline

1. **Color Segmentation**: HSV-based masking with histogram equalization
2. **Morphological Processing**: Opening/closing operations for noise reduction
3. **Contour Analysis**: External contour detection with area/aspect filtering
4. **Shape Validation**: 4-point rectangle approximation with solidity scoring
5. **Confidence Calculation**: Weighted scoring based on multiple geometric factors

### Key Improvements

- **Adaptive Thresholding**: Dynamic HSV range adjustment for lighting variations
- **Histogram Equalization**: Enhanced V-channel processing for better contrast
- **Multi-scale Morphological Operations**: Kernel size 7 with 3 closing iterations
- **4-Point Rectangle Detection**: Precise corner point identification
- **Automatic ROI Generation**: 10% padding around detected wood regions
- **Confidence Scoring Algorithm**: Area (30%), aspect ratio (30%), solidity (20%), vertices (20%)
- **Width Measurement**: Real-time mm conversion for top and bottom camera detections

### Architecture

- **ColorWoodDetector Class**: Core detection algorithms and parameters
- **CameraHandler Class**: Camera initialization and configuration management
- **Detection Methods**:
  - `detect_wood_comprehensive()`: Main pipeline combining color and shape analysis
  - `detect_wood_by_color()`: HSV-based color segmentation
  - `detect_rectangular_contours()`: Contour filtering and validation
  - `visualize_detection()`: Result overlay and annotation

### Performance Characteristics

- **Resolution**: 1280x720 per camera
- **Frame Rate**: Real-time processing
- **Detection Range**: 3000-500000 pixels contour area
- **Aspect Ratio**: 1.0-10.0 for valid wood shapes
- **Confidence Threshold**: 0.0-1.0 with multi-factor scoring

### Output Formats

- **Live Display**: Side-by-side camera views with overlays
- **Saved Frames**: JPEG images in `captured_frames/`
- **Detection Results**: Visualization and masks in `detection_results/`
- **Console Output**: Real-time statistics every 30 frames

### Error Handling

- Camera initialization validation
- Image loading error detection
- Morphological operation fallbacks
- Exception handling in all detection methods