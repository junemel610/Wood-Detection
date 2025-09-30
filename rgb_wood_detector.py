#!/usr/bin/env python3
"""
Color-based Wood Detection with 4-Point Rectangle Detection
This module provides robust wood detection using:
1. RGB color analysis for wood tones
2. Contour detection for rectangular shapes
3. Automatic ROI generation
4. Adaptive thresholding
"""

import cv2
import numpy as np
import json
from typing import List, Tuple, Dict, Optional

class ColorWoodDetector:
    def __init__(self):
        self.wood_color_profiles = {
            'top_panel': {
                'rgb_lower': np.array([96, 100, 80]),  # BGR order
                'rgb_upper': np.array([230, 210, 200]),
                'name': 'Top Panel Wood'
            },
            'bottom_panel': {
                'rgb_lower': np.array([135, 145, 95]),  # RGB range for wood detection
                'rgb_upper': np.array([200, 220, 230]),  # RGB range for wood detection
                'name': 'Bottom Panel Wood'
            }
        }
        
        # Detection parameters
        self.min_contour_area = 2000      # Increased for more reliable detection with tighter RGB ranges
        self.max_contour_area = 500000    # Slightly reduced for typical wood plank sizes
        self.min_aspect_ratio = 1.0       # Tightened for more rectangular wood shapes
        self.max_aspect_ratio = 10.0      # Reduced for more typical plank proportions
        self.contour_approximation = 0.025 # Slightly tighter for better shape approximation
        
        # Morphological operations
        self.morph_kernel_size = 11
        self.closing_iterations = 3
        self.opening_iterations = 2

        # Pixel to mm conversion parameters for width measurement
        self.pixel_per_mm_top = 2.96     # Placeholder: calibrate based on top camera distance (31cm)
        self.pixel_per_mm_bottom = 3.18  # Placeholder: calibrate based on bottom camera distance

    def calculate_width_mm(self, bbox_pixels: int, camera: str = 'top') -> float:
        """Calculate width in mm from bounding box dimension in pixels using pixel_per_mm factors"""
        if camera == 'top':
            return bbox_pixels / self.pixel_per_mm_top
        elif camera == 'bottom':
            return bbox_pixels / self.pixel_per_mm_bottom
        else:
            raise ValueError("Camera must be 'top' or 'bottom'")

    def analyze_image_colors(self, image_path: str) -> Dict:
        """Analyze the color composition of the captured image"""
        print(f"🎨 Analyzing colors in: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        analysis = {
            "image_size": f"{w}x{h}",
            "wood_profiles_detected": {},
            "dominant_colors": {},
            "recommendations": []
        }

        # Test each wood color profile
        for profile_name, profile in self.wood_color_profiles.items():
            mask = cv2.inRange(rgb, profile['rgb_lower'], profile['rgb_upper'])
            pixels_detected = cv2.countNonZero(mask)
            percentage = (pixels_detected / (h * w)) * 100

            analysis["wood_profiles_detected"][profile_name] = {
                "pixels": pixels_detected,
                "percentage": round(percentage, 2),
                "detected": percentage > 1.0  # Consider detected if >1% of image
            }

            if percentage > 1.0:
                print(f"  ✅ {profile['name']}: {percentage:.1f}% of image")
            else:
                print(f"  ❌ {profile['name']}: {percentage:.1f}% of image")

        # Find dominant colors in RGB
        rgb_flat = rgb.reshape(-1, 3)
        r_values = rgb_flat[:, 0]
        g_values = rgb_flat[:, 1]
        b_values = rgb_flat[:, 2]

        analysis["dominant_colors"] = {
            "red_mean": int(np.mean(r_values)),
            "red_std": int(np.std(r_values)),
            "green_mean": int(np.mean(g_values)),
            "blue_mean": int(np.mean(b_values))
        }
        
        # Generate recommendations
        best_profiles = []
        for name, data in analysis["wood_profiles_detected"].items():
            if data["detected"] and data["percentage"] > 5:
                best_profiles.append((name, data["percentage"]))
        
        if best_profiles:
            best_profiles.sort(key=lambda x: x[1], reverse=True)
            analysis["recommendations"].append(f"Use {best_profiles[0][0]} profile as primary detection method")
        else:
            analysis["recommendations"].append("Consider creating custom color profile for this wood type")
            analysis["recommendations"].append(f"Dominant RGB: R={analysis['dominant_colors']['red_mean']}, G={analysis['dominant_colors']['green_mean']}, B={analysis['dominant_colors']['blue_mean']}")
        
        return analysis
    
    def detect_wood_by_color(self, image: np.ndarray, profile_names: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Detect wood using color profiles"""
        if profile_names is None:
            profile_names = list(self.wood_color_profiles.keys())

        # Apply histogram equalization on V channel for better lighting compensation
        hsv_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_temp)
        v = cv2.equalizeHist(v)
        hsv_temp = cv2.merge([h, s, v])
        rgb = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2BGR)

        # # Dynamically update RGB ranges based on dominant colors
        # self.update_rgb_ranges_based_on_dominant_colors(rgb)

        combined_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        detections = []

        print(f"🎨 Using profiles: {profile_names}")

        # Combine masks from selected profiles
        for profile_name in profile_names:
            if profile_name in self.wood_color_profiles:
                profile = self.wood_color_profiles[profile_name]
                mask = cv2.inRange(rgb, profile['rgb_lower'], profile['rgb_upper'])
                mask_pixels = cv2.countNonZero(mask)
                total_pixels = rgb.shape[0] * rgb.shape[1]
                mask_percentage = (mask_pixels / total_pixels) * 100
                print(f"  📊 {profile_name}: RGB range {profile['rgb_lower']} - {profile['rgb_upper']}, mask {mask_pixels} pixels ({mask_percentage:.1f}%)")
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        pre_morph_pixels = cv2.countNonZero(combined_mask)
        pre_morph_percentage = (pre_morph_pixels / total_pixels) * 100
        print(f"🔧 Pre-morph combined mask: {pre_morph_pixels} pixels ({pre_morph_percentage:.1f}%)")

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=self.closing_iterations)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=self.opening_iterations)

        post_morph_pixels = cv2.countNonZero(combined_mask)
        post_morph_percentage = (post_morph_pixels / total_pixels) * 100
        print(f"🔧 Post-morph combined mask: {post_morph_pixels} pixels ({post_morph_percentage:.1f}%)")

        # Additional logging for dominant colors
        rgb_flat = rgb.reshape(-1, 3)
        r_values = rgb_flat[:, 0]
        g_values = rgb_flat[:, 1]
        b_values = rgb_flat[:, 2]
        print(f"🎨 Dominant RGB in image: R={int(np.mean(r_values)):.0f}±{int(np.std(r_values)):.0f}, G={int(np.mean(g_values)):.0f}, B={int(np.mean(b_values)):.0f}")

        return combined_mask, detections

    def update_rgb_ranges_based_on_dominant_colors(self, rgb):
        """Dynamically adjust RGB ranges based on dominant colors in the image"""
        rgb_flat = rgb.reshape(-1, 3)
        r_mean = int(np.mean(rgb_flat[:, 0]))
        g_mean = int(np.mean(rgb_flat[:, 1]))
        b_mean = int(np.mean(rgb_flat[:, 2]))

        # Update profiles based on dominant colors
        self.wood_color_profiles['top_panel']['rgb_lower'] = np.array([max(0, r_mean - 30), max(0, g_mean - 30), max(0, b_mean - 30)])
        self.wood_color_profiles['top_panel']['rgb_upper'] = np.array([min(255, r_mean + 30), min(255, g_mean + 30), min(255, b_mean + 30)])
        self.wood_color_profiles['bottom_panel']['rgb_lower'] = np.array([max(0, r_mean - 30), max(0, g_mean - 30), max(0, b_mean - 30)])
        self.wood_color_profiles['bottom_panel']['rgb_upper'] = np.array([min(255, r_mean + 30), min(255, g_mean + 30), min(255, b_mean + 30)])
        print(f"🔧 Dynamically updated RGB ranges: R=[{r_mean-30}-{r_mean+30}], G=[{g_mean-30}-{g_mean+30}], B=[{b_mean-30}-{b_mean+30}]")
    
    def detect_rectangular_contours(self, mask: np.ndarray) -> List[Dict]:
        """Detect rectangular contours that could be wood planks"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"📐 Found {len(contours)} total contours")

        wood_candidates = []
        rejected_area = 0
        rejected_aspect = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                rejected_area += 1
                print(f"  ❌ Contour {i}: area {area:.0f} out of range [{self.min_contour_area}, {self.max_contour_area}]")
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)

            # Filter by aspect ratio (wood planks are typically rectangular)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                rejected_aspect += 1
                print(f"  ❌ Contour {i}: aspect {aspect_ratio:.2f} out of range [{self.min_aspect_ratio}, {self.max_aspect_ratio}]")
                continue

            # Approximate contour to polygon
            epsilon = self.contour_approximation * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Calculate additional metrics
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Get rotated rectangle for better angle detection
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            confidence = self._calculate_wood_confidence(area, aspect_ratio, solidity, len(approx))

            wood_candidate = {
                'contour': contour,
                'approx_points': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'vertices': len(approx),
                'rotated_rect': rect,
                'corner_points': box,
                'confidence': confidence
            }

            wood_candidates.append(wood_candidate)
            print(f"  ✅ Contour {i}: area {area:.0f}, aspect {aspect_ratio:.2f}, solidity {solidity:.2f}, confidence {confidence:.2f}")

        print(f"📊 Contour filtering: {len(contours)} total, {rejected_area} rejected by area, {rejected_aspect} by aspect, {len(wood_candidates)} candidates")

        # Sort by confidence
        wood_candidates.sort(key=lambda x: x['confidence'], reverse=True)

        return wood_candidates
    
    def _calculate_wood_confidence(self, area: float, aspect_ratio: float, solidity: float, vertices: int) -> float:
        """Calculate confidence score for wood detection"""
        confidence = 0.0
        
        # Area score (larger is better, up to a point)
        if 10000 <= area <= 100000:
            confidence += 0.3
        elif area > 5000:
            confidence += 0.2
        
        # Aspect ratio score (rectangular is better)
        if 2.0 <= aspect_ratio <= 6.0:
            confidence += 0.3
        elif 1.5 <= aspect_ratio <= 8.0:
            confidence += 0.2
        
        # Solidity score (more solid shapes are better)
        if solidity > 0.7:
            confidence += 0.2
        elif solidity > 0.5:
            confidence += 0.1
        
        # Vertex count score (4-6 vertices for rectangular shapes)
        if vertices == 4:
            confidence += 0.2
        elif 4 <= vertices <= 6:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_auto_roi(self, wood_candidates: List[Dict], image_shape: Tuple) -> Optional[Tuple[int, int, int, int]]:
        """Generate automatic ROI based on detected wood"""
        if not wood_candidates:
            return None
        
        # Use the highest confidence detection
        best_candidate = wood_candidates[0]
        x, y, w, h = best_candidate['bbox']
        
        # Add some padding around the detected wood
        padding_x = int(w * 0.1)  # 10% padding
        padding_y = int(h * 0.1)
        
        roi_x1 = max(0, x - padding_x)
        roi_y1 = max(0, y - padding_y)
        roi_x2 = min(image_shape[1], x + w + padding_x)
        roi_y2 = min(image_shape[0], y + h + padding_y)
        
        return (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)
    
    def detect_wood_comprehensive(self, image: np.ndarray, profile_names: List[str] = None, roi: Tuple[int, int, int, int] = None) -> Dict:
        """Comprehensive wood detection combining color and shape analysis"""

        print(f"🪵 Starting comprehensive wood detection on image shape: {image.shape}")

        # Step 1: Color-based detection with optional ROI
        if roi is not None:
            x, y, w, h = roi
            cropped = image[y:y+h, x:x+w]
            color_mask_cropped, _ = self.detect_wood_by_color(cropped, profile_names)
            color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            color_mask[y:y+h, x:x+w] = color_mask_cropped
        else:
            color_mask, _ = self.detect_wood_by_color(image, profile_names)

        mask_pixels = cv2.countNonZero(color_mask)
        total_pixels = image.shape[0] * image.shape[1]
        mask_percentage = (mask_pixels / total_pixels) * 100
        print(f"🎨 Color mask: {mask_pixels} pixels ({mask_percentage:.1f}%)")

        # Step 2: Find rectangular contours
        wood_candidates = self.detect_rectangular_contours(color_mask)
        print(f"📐 Found {len(wood_candidates)} wood candidates after contour filtering")

        # Step 3: Generate automatic ROI
        auto_roi = self.generate_auto_roi(wood_candidates, image.shape)
        if auto_roi:
            print(f"🎯 Auto ROI generated: {auto_roi}")
        else:
            print("❌ No auto ROI generated (no candidates)")

        # Step 4: Integrate texture analysis for enhanced confidence
        texture_confidence = self._detect_wood_by_texture(image)
        combined_confidence = (wood_candidates[0]['confidence'] + texture_confidence) / 2 if wood_candidates else texture_confidence

        # Step 5: Create result
        result = {
            'wood_detected': len(wood_candidates) > 0 or texture_confidence > 0.4,
            'wood_count': len(wood_candidates),
            'wood_candidates': wood_candidates,
            'auto_roi': auto_roi,
            'color_mask': color_mask,
            'confidence': combined_confidence,
            'texture_confidence': texture_confidence
        }

        print(f"✅ Detection complete: wood_detected={result['wood_detected']}, count={result['wood_count']}, confidence={result['confidence']:.2f}")

        return result
    
    def visualize_detection(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Create visualization of wood detection results"""
        vis_image = image.copy()
        
        # Draw all wood candidates
        for i, candidate in enumerate(detection_result['wood_candidates']):
            # Draw bounding box
            x, y, w, h = candidate['bbox']
            color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Best candidate in green, others in yellow
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence label
            label = f"Wood {i+1}: {candidate['confidence']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add metrics
            metrics = f"AR:{candidate['aspect_ratio']:.1f} S:{candidate['solidity']:.2f}"
            cv2.putText(vis_image, metrics, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw auto ROI
        if detection_result['auto_roi']:
            roi_x, roi_y, roi_w, roi_h = detection_result['auto_roi']
            cv2.rectangle(vis_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 3)
            cv2.putText(vis_image, "AUTO ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Add summary info
        summary = f"Wood Detected: {detection_result['wood_detected']} | Count: {detection_result['wood_count']} | Confidence: {detection_result['confidence']:.2f}"
        cv2.putText(vis_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image

    def detect_wood_presence(self, frame):
        color_conf = self._detect_wood_by_color(frame)
        texture_conf = self._detect_wood_by_texture(frame)
        shape_conf = self._detect_wood_by_shape(frame)
        
        # Combine confidences with weights (color most important for wood)
        combined_conf = (0.5 * color_conf + 0.3 * texture_conf + 0.2 * shape_conf)
        wood_detected = combined_conf > 0.3  # Lower threshold since multiple methods
        
        return wood_detected, combined_conf, {
            'color_confidence': color_conf,
            'texture_confidence': texture_conf,
            'shape_confidence': shape_conf
        }

    def detect_wood(self, frame):
        """
        Enhanced wood detection using the wood detection model.
        Falls back to visual detection if model is not available.
        Returns True if wood is detected, False otherwise.
        """
        wood_detected, confidence, _ = self.detect_wood_presence(frame)
        return wood_detected

    def _detect_wood_by_color(self, frame):
        """Detect wood using RGB color segmentation"""
        try:
            rgb_frame = frame

            # Define multiple wood color ranges to handle different wood types (refined for consistency)
            wood_ranges = [
                # Light wood (pine, birch) - RGB range
                ([150, 159, 109], [230, 210, 200]),
                # Medium wood (oak, maple) - RGB range
                ([130, 150, 160], [180, 200, 210]),
                # Dark wood (walnut, mahogany) - RGB range
                ([100, 120, 130], [160, 180, 190])
            ]
            
            combined_mask = None
            for lower, upper in wood_ranges:
                mask = cv2.inRange(rgb_frame, np.array(lower), np.array(upper))
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Clean up mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate percentage of wood-like pixels
            wood_pixel_count = cv2.countNonZero(combined_mask)
            total_pixels = frame.shape[0] * frame.shape[1]
            wood_percentage = (wood_pixel_count / total_pixels) * 100
            
            # Return confidence (normalized to 0-1)
            return min(wood_percentage / 20.0, 1.0)  # 20% wood pixels = 100% confidence
            
        except Exception as e:
            print(f"Error in color-based wood detection: {e}")
            return 0.0

    def _detect_wood_by_texture(self, frame):
        """Detect wood using basic texture analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate texture using standard deviation in local neighborhoods
            kernel_size = 15
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            # Calculate local standard deviation (texture measure)
            mean = cv2.blur(blurred.astype(np.float32), (kernel_size, kernel_size))
            sqr_mean = cv2.blur((blurred.astype(np.float32))**2, (kernel_size, kernel_size))
            texture_variance = sqr_mean - mean**2
            texture_std = np.sqrt(np.maximum(texture_variance, 0))
            
            # Wood typically has moderate texture (not too smooth, not too rough)
            # Calculate confidence based on texture distribution
            texture_mean = np.mean(texture_std)
            texture_confidence = 0.0
            
            # Optimal texture range for wood (adjust based on testing)
            if 10 < texture_mean < 40:
                texture_confidence = 1.0 - abs(texture_mean - 25) / 15.0
            
            return max(0.0, min(1.0, texture_confidence))
            
        except Exception as e:
            print(f"Error in texture-based wood detection: {e}")
            return 0.0

    def _detect_wood_by_shape(self, frame):
        """Detect wood using contour and shape analysis"""
        try:
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Analyze largest contours for rectangular/wood-like shapes
            frame_area = frame.shape[0] * frame.shape[1]
            shape_confidence = 0.0
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(contour)
                
                # Skip very small contours
                if area < frame_area * 0.05:
                    continue
                
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # Aspect ratio analysis
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Wood planks typically have certain aspect ratios
                # Adjust these ranges based on your conveyor setup
                if 0.3 < aspect_ratio < 5.0:  # Not too square, not too thin
                    # Calculate rectangularity (how close to rectangle)
                    rect_area = w * h
                    rectangularity = area / rect_area
                    
                    if rectangularity > 0.6:  # Reasonably rectangular
                        shape_confidence = max(shape_confidence, rectangularity)
            
            return min(1.0, shape_confidence)
            
        except Exception as e:
            print(f"Error in shape-based wood detection: {e}")
            return 0.0

class CameraHandler:
    def __init__(self):
        self.top_camera = None
        self.bottom_camera = None
        self.top_camera_index = 0  # Cam0
        self.bottom_camera_index = 2  # Cam2
        self.top_camera_settings = {
            'brightness': 0,
            'contrast': 32,
            'saturation': 64,
            'hue': 0,
            'exposure': -6,
            'white_balance': 4520,
            'gain': 0
        }
        self.bottom_camera_settings = {
            'brightness': 135,
            'contrast': 75,
            'saturation': 155,
            'hue': 0,
            'exposure': -6,
            'white_balance': 5400,
            'gain': 0
        }
    def initialize_cameras(self):
        try:
            self.top_camera = cv2.VideoCapture(self.top_camera_index)
            if not self.top_camera.isOpened():
                raise RuntimeError(f"Could not open top camera (Cam0 - index {self.top_camera_index})")
            self.bottom_camera = cv2.VideoCapture(self.bottom_camera_index)
            if not self.bottom_camera.isOpened():
                self.top_camera.release()
                raise RuntimeError(f"Could not open bottom camera (Cam2 - index {self.bottom_camera_index})")
            self.top_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.top_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._apply_camera_settings(self.top_camera, self.top_camera_settings)
            self._apply_camera_settings(self.bottom_camera, self.bottom_camera_settings)
            print("Cameras initialized successfully at 720p (1280x720)")
        except Exception as e:
            self.release_cameras()
            raise RuntimeError(f"Failed to initialize cameras: {str(e)}")
    def _apply_camera_settings(self, camera, settings):
        try:
            camera.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
            camera.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
            camera.set(cv2.CAP_PROP_SATURATION, settings['saturation'])
            camera.set(cv2.CAP_PROP_HUE, settings['hue'])
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            camera.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
            camera.set(cv2.CAP_PROP_AUTO_WB, 0)
            camera.set(cv2.CAP_PROP_WB_TEMPERATURE, settings['white_balance'])
            camera.set(cv2.CAP_PROP_GAIN, settings['gain'])
        except Exception as e:
            print(f"Warning: Some camera settings may not be supported: {e}")
    def release_cameras(self):
        if self.top_camera:
            self.top_camera.release()
            self.top_camera = None
        if self.bottom_camera:
            self.bottom_camera.release()
            self.bottom_camera = None
        print("Cameras released")

def main():
    camera_handler = CameraHandler()
    camera_handler.initialize_cameras()
    detector = ColorWoodDetector()

    # Define center ROIs (top camera 80% height, full width, centered; bottom camera 80% of frame, centered)
    frame_width = 1280
    frame_height = 720
    roi_height_top = int(frame_height * 0.8)
    roi_y_top = (frame_height - roi_height_top) // 2
    roi_width_top = frame_width
    roi_x_top = 0
    roi_top = (roi_x_top, roi_y_top, roi_width_top, roi_height_top)

    roi_width_bottom = int(frame_width * 0.8)
    roi_height_bottom = int(frame_height * 0.8)
    roi_x_bottom = (frame_width - roi_width_bottom) // 2
    roi_y_bottom = (frame_height - roi_height_bottom) // 2
    roi_bottom = (roi_x_bottom, roi_y_bottom, roi_width_bottom, roi_height_bottom)  # Same for both cameras

    cap0 = camera_handler.top_camera
    cap2 = camera_handler.bottom_camera
    if not cap0 or not cap2:
        print("❌ Could not open cameras")
        return
    print("🎥 Starting live wood detection from video0 and video2")
    print("Press 'q' to quit")
    frame_count = 0
    show_mask = False  # Ensure show_mask is always defined before the loop
    detection_enabled = True  # Toggle for detection processing
    # Directory to save captured frames
    save_dir = "captured_frames"
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        while True:
            ret0, frame0 = cap0.read()
            ret2, frame2 = cap2.read()
            if not ret0 or not ret2:
                print("❌ Failed to read frames from cameras")
                break
            frame_count += 1

            frame2_flipped = cv2.flip(frame2, 1)

            if detection_enabled:
                # Process frame from camera 0 (top)
                detection_result0 = detector.detect_wood_comprehensive(frame0, roi=roi_top)
                wood_detected0 = detection_result0['wood_detected']
                confidence0 = detection_result0['confidence']

                top_width_mm = None
                if wood_detected0 and detection_result0['wood_candidates']:
                    candidate = detection_result0['wood_candidates'][0]
                    _, _, _, h = candidate['bbox']
                    top_width_mm = detector.calculate_width_mm(h, 'top')

            else:
                # Skip detection processing for top
                detection_result0 = {'wood_candidates': [], 'color_mask': np.zeros(frame0.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
                wood_detected0 = False
                confidence0 = 0.0

                top_width_mm = None

            # Bottom camera: always skip wood detection
            detection_result2 = {'wood_candidates': [], 'color_mask': np.zeros(frame2_flipped.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
            wood_detected2 = False
            confidence2 = 0.0
            bottom_width_mm = None


            # Draw ROI bounding boxes
            cv2.rectangle(frame0, (roi_top[0], roi_top[1]), (roi_top[0] + roi_top[2], roi_top[1] + roi_top[3]), (0, 255, 0), 2)
            cv2.rectangle(frame2_flipped, (roi_bottom[0], roi_bottom[1]), (roi_bottom[0] + roi_bottom[2], roi_bottom[1] + roi_bottom[3]), (0, 255, 0), 2)
            
            # Draw bounding box on frame 0 for the best candidate only
            if detection_result0['wood_candidates']:
                candidate = detection_result0['wood_candidates'][0]
                x, y, w, h = candidate['bbox']
                color = (0, 255, 0)
                cv2.rectangle(frame0, (x, y), (x + w, y + h), color, 2)

                # Add confidence and width label
                label = f"Wood: {candidate['confidence']:.2f} | Width (vert): {top_width_mm:.1f}mm" if top_width_mm is not None else f"Wood: {candidate['confidence']:.2f}"
                cv2.putText(frame0, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            
            # Add text overlays
            # Camera 0
            if detection_enabled:
                status0 = "WOOD DETECTED" if wood_detected0 else "NO WOOD"
                color0 = (0, 255, 0) if wood_detected0 else (0, 0, 255)
            else:
                status0 = "DETECTION OFF"
                color0 = (0, 0, 255)
            cv2.putText(frame0, f"Camera 0 (Top): {status0}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color0, 2)
            cv2.putText(frame0, f"Count: {detection_result0['wood_count']} | Confidence: {confidence0:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame0, f"Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Camera 2
            status2 = "NO DETECTION"
            color2 = (128, 128, 128)
            cv2.putText(frame2, f"Camera 2 (Bottom): {status2}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)
            cv2.putText(frame2, f"Count: {detection_result2['wood_count']} | Confidence: {confidence2:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame2, f"Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize frames and masks to 480p for display
            display_width, display_height = 852, 480  # 480p (16:9)
            frame0_disp = cv2.resize(frame0, (display_width, display_height))
            frame2_disp = cv2.resize(frame2_flipped, (display_width, display_height))  # Already flipped
            mask0_disp = cv2.resize(detection_result0['color_mask'], (display_width, display_height))
            mask2_disp = cv2.resize(detection_result2['color_mask'], (display_width, display_height))  # Already flipped

            # No additional flip needed for bottom camera display

            # Stack frames side by side
            if show_mask:
                combined_disp = np.hstack((cv2.cvtColor(mask0_disp, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask2_disp, cv2.COLOR_GRAY2BGR)))
            else:
                combined_disp = np.hstack((frame0_disp, frame2_disp))

            cv2.imshow('Wood Detection (Top | Bottom)', combined_disp)

            # Print to console every 30 frames
            if frame_count % 30 == 0:
                width0_str = f", width={top_width_mm:.1f}mm" if top_width_mm is not None else ""
                width2_str = f", width={bottom_width_mm:.1f}mm" if bottom_width_mm is not None else ""
                print(f"Frame {frame_count}: Cam0={status0}({confidence0:.2f}{width0_str}), Cam2={status2}({confidence2:.2f}{width2_str})")

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detection_enabled = not detection_enabled
                print(f"Detection {'enabled' if detection_enabled else 'disabled'}")
            elif key == ord('c'):
                show_mask = not show_mask
                if show_mask:
                    print("Mask view enabled (press 'C' again to toggle off)")
                else:
                    print("Mask view disabled (press 'C' to toggle on)")
            elif key == ord('s'):
                # Save both frames
                top_path = os.path.join(save_dir, f"TopPanel_{frame_count}.jpg")
                bottom_path = os.path.join(save_dir, f"BottomPanel_{frame_count}.jpg")
                cv2.imwrite(top_path, frame0)
                cv2.imwrite(bottom_path, frame2)
                print(f"Saved TopPanel to {top_path}")
                print(f"Saved BottomPanel to {bottom_path}")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        camera_handler.release_cameras()
        cv2.destroyAllWindows()
        print("📷 Cameras released")


if __name__ == "__main__":
    main()
