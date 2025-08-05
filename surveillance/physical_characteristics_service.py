"""Physical Characteristics Analysis Service

Analyzes physical characteristics of detected persons including:
- Estimated height based on bounding box dimensions
- Clothing color analysis
- Body proportions
- Movement patterns
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
import colorsys

@dataclass
class PhysicalCharacteristics:
    """Represents physical characteristics of a person"""
    estimated_height_cm: Optional[float] = None
    dominant_clothing_colors: List[str] = None
    clothing_description: str = ""
    body_proportions: Dict[str, float] = None
    movement_speed: Optional[float] = None
    posture: str = "standing"
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.dominant_clothing_colors is None:
            self.dominant_clothing_colors = []
        if self.body_proportions is None:
            self.body_proportions = {}

class PhysicalCharacteristicsService:
    """Service for analyzing physical characteristics of detected persons"""
    
    def __init__(self):
        # Reference heights for calibration (in pixels to cm ratio)
        self.height_calibration = {
            'default_ratio': 0.5,  # Default: 1 pixel = 0.5 cm
            'camera_specific': {}  # Camera-specific calibrations
        }
        
        # Color mapping for clothing analysis
        self.color_names = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)],
            'purple': [(130, 50, 50), (170, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'gray': [(0, 0, 50), (180, 30, 200)]
        }
        
        print("[INFO] Physical Characteristics Service initialized")
    
    def analyze_person_characteristics(self, frame: np.ndarray, 
                                     person_bbox: Tuple[int, int, int, int],
                                     camera_id: str = "default") -> PhysicalCharacteristics:
        """Analyze physical characteristics of a detected person
        
        Args:
            frame: Full frame image
            person_bbox: Bounding box of the person (x1, y1, x2, y2)
            camera_id: Camera identifier for calibration
            
        Returns:
            PhysicalCharacteristics object with analyzed data
        """
        try:
            x1, y1, x2, y2 = person_bbox
            
            # Extract person region
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                return PhysicalCharacteristics()
            
            # Analyze different characteristics
            height = self._estimate_height(person_bbox, camera_id)
            colors = self._analyze_clothing_colors(person_roi)
            description = self._generate_clothing_description(colors, person_roi)
            proportions = self._analyze_body_proportions(person_bbox)
            
            return PhysicalCharacteristics(
                estimated_height_cm=height,
                dominant_clothing_colors=colors,
                clothing_description=description,
                body_proportions=proportions,
                confidence=0.8  # Base confidence
            )
            
        except Exception as e:
            print(f"[ERROR] Error analyzing physical characteristics: {e}")
            return PhysicalCharacteristics()
    
    def _estimate_height(self, person_bbox: Tuple[int, int, int, int], 
                        camera_id: str) -> Optional[float]:
        """Estimate person height based on bounding box
        
        Args:
            person_bbox: Bounding box (x1, y1, x2, y2)
            camera_id: Camera identifier
            
        Returns:
            Estimated height in centimeters
        """
        try:
            x1, y1, x2, y2 = person_bbox
            height_pixels = y2 - y1
            
            # Get calibration ratio for this camera
            ratio = self.height_calibration.get('camera_specific', {}).get(
                camera_id, self.height_calibration['default_ratio']
            )
            
            # Estimate height (assuming average person height is 170cm)
            # This is a simplified estimation - in real scenarios, you'd need proper calibration
            estimated_height = height_pixels * ratio
            
            # Clamp to reasonable human height range
            estimated_height = max(140, min(220, estimated_height))
            
            return round(estimated_height, 1)
            
        except Exception as e:
            print(f"[ERROR] Error estimating height: {e}")
            return None
    
    def _analyze_clothing_colors(self, person_roi: np.ndarray) -> List[str]:
        """Analyze dominant colors in person's clothing
        
        Args:
            person_roi: Person region of interest
            
        Returns:
            List of dominant color names
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            
            # Focus on torso area (middle 60% of the person)
            h, w = person_roi.shape[:2]
            torso_start = int(h * 0.2)
            torso_end = int(h * 0.8)
            torso_roi = hsv[torso_start:torso_end, :]
            
            detected_colors = []
            
            # Check each color range
            for color_name, ranges in self.color_names.items():
                total_pixels = 0
                
                if len(ranges) == 2:  # Single range
                    lower, upper = ranges
                    mask = cv2.inRange(torso_roi, np.array(lower), np.array(upper))
                    total_pixels = cv2.countNonZero(mask)
                elif len(ranges) == 4:  # Two ranges (for red)
                    lower1, upper1, lower2, upper2 = ranges
                    mask1 = cv2.inRange(torso_roi, np.array(lower1), np.array(upper1))
                    mask2 = cv2.inRange(torso_roi, np.array(lower2), np.array(upper2))
                    total_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
                
                # If color covers more than 15% of torso area, consider it dominant
                if total_pixels > (torso_roi.shape[0] * torso_roi.shape[1] * 0.15):
                    detected_colors.append((color_name, total_pixels))
            
            # Sort by pixel count and return top colors
            detected_colors.sort(key=lambda x: x[1], reverse=True)
            return [color for color, _ in detected_colors[:3]]
            
        except Exception as e:
            print(f"[ERROR] Error analyzing clothing colors: {e}")
            return []
    
    def _generate_clothing_description(self, colors: List[str], 
                                     person_roi: np.ndarray) -> str:
        """Generate a text description of clothing
        
        Args:
            colors: List of dominant colors
            person_roi: Person region of interest
            
        Returns:
            Text description of clothing
        """
        try:
            if not colors:
                return "Ropa no identificada"
            
            # Basic description based on colors
            if len(colors) == 1:
                return f"Ropa {colors[0]}"
            elif len(colors) == 2:
                return f"Ropa {colors[0]} y {colors[1]}"
            else:
                return f"Ropa {colors[0]}, {colors[1]} y {colors[2]}"
                
        except Exception as e:
            print(f"[ERROR] Error generating clothing description: {e}")
            return "Descripción no disponible"
    
    def _analyze_body_proportions(self, person_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Analyze body proportions
        
        Args:
            person_bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with proportion measurements
        """
        try:
            x1, y1, x2, y2 = person_bbox
            width = x2 - x1
            height = y2 - y1
            
            return {
                'aspect_ratio': round(height / width if width > 0 else 0, 2),
                'width_pixels': width,
                'height_pixels': height
            }
            
        except Exception as e:
            print(f"[ERROR] Error analyzing body proportions: {e}")
            return {}
    
    def set_camera_calibration(self, camera_id: str, pixels_per_cm: float):
        """Set height calibration for a specific camera
        
        Args:
            camera_id: Camera identifier
            pixels_per_cm: Calibration ratio (pixels per centimeter)
        """
        self.height_calibration['camera_specific'][camera_id] = pixels_per_cm
        print(f"[INFO] Height calibration set for camera {camera_id}: {pixels_per_cm} pixels/cm")
    
    def get_characteristics_summary(self, characteristics: PhysicalCharacteristics) -> str:
        """Get a human-readable summary of characteristics
        
        Args:
            characteristics: PhysicalCharacteristics object
            
        Returns:
            Human-readable summary string
        """
        summary_parts = []
        
        if characteristics.estimated_height_cm:
            summary_parts.append(f"Altura: ~{characteristics.estimated_height_cm}cm")
        
        if characteristics.clothing_description:
            summary_parts.append(characteristics.clothing_description)
        
        if characteristics.dominant_clothing_colors:
            colors_str = ", ".join(characteristics.dominant_clothing_colors)
            summary_parts.append(f"Colores: {colors_str}")
        
        return " | ".join(summary_parts) if summary_parts else "Sin características identificadas"

# Global instance
physical_characteristics_service = PhysicalCharacteristicsService()