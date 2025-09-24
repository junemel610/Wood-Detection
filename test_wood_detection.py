#!/usr/bin/env python3
"""
Test script for wood detection on sample images
"""

import cv2
import numpy as np
import os
from color_wood_detector import ColorWoodDetector

def test_wood_detection():
    detector = ColorWoodDetector()

    # Create output directory
    output_dir = "detection_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test images from captured_frames
    sample_images = [
        "captured_frames/TopPanel_46.jpg",
        "captured_frames/BottomPanel_46.jpg",
        "captured_frames/TopPanel_195.jpg",
        "captured_frames/BottomPanel_195.jpg",
        "captured_frames/TopPanel_1707.jpg",
        "captured_frames/BottomPanel_1707.jpg"
    ]

    for image_path in sample_images:
        if not os.path.exists(image_path):
            print(f"❌ Image {image_path} not found, skipping")
            continue

        print(f"\n🖼️  Testing {image_path}")
        print("=" * 50)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load {image_path}")
            continue

        # Run detection
        result = detector.detect_wood_comprehensive(image)

        # Visualize and save
        vis_image = detector.visualize_detection(image, result)

        # Save visualization
        vis_path = os.path.join(output_dir, f"vis_{os.path.basename(image_path)}")
        cv2.imwrite(vis_path, vis_image)
        print(f"💾 Saved visualization to {vis_path}")

        # Save mask
        mask_path = os.path.join(output_dir, f"mask_{os.path.basename(image_path)}")
        cv2.imwrite(mask_path, result['color_mask'])
        print(f"💾 Saved mask to {mask_path}")

        print(f"📊 Summary: Detected={result['wood_detected']}, Count={result['wood_count']}, Confidence={result['confidence']:.2f}")

if __name__ == "__main__":
    test_wood_detection()