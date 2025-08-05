"""Person Detection Service

Handles person detection using YOLO model.
Separated from the main surveillance service for better modularity.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

class PersonDetectionService:
    """Service for detecting persons in images using YOLO"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.6):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self._setup_yolo(model_path)
    
    def _setup_yolo(self, model_path: str):
        """Configure YOLO for person detection"""
        try:
            # Force YOLO to use CPU to avoid conflicts with InsightFace GPU
            self.model = YOLO(model_path)
            # Configure explicitly for CPU
            self.model.to('cpu')
            print(f"[INFO] YOLO model loaded for person detection (CPU): {model_path}")
        except Exception as e:
            print(f"[ERROR] Error loading YOLO model: {str(e)}")
            self.model = None
    
    def detect_persons(self, img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int, float]], int]:
        """Detect persons in an image using YOLO
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Tuple of (person_boxes, person_count)
            person_boxes: List of (x1, y1, x2, y2, confidence) tuples
            person_count: Number of persons detected
        """
        if self.model is None:
            return [], 0
        
        try:
            # Run YOLO detection on CPU
            results = self.model(img, device='cpu', verbose=False)
            
            person_boxes = []
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Class 0 in COCO dataset is 'person'
                        if int(box.cls[0]) == 0:  # person class
                            confidence = float(box.conf[0])
                            if confidence > self.confidence_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                person_boxes.append((x1, y1, x2, y2, confidence))
                                person_count += 1
            
            return person_boxes, person_count
            
        except Exception as e:
            print(f"[ERROR] Error in person detection: {e}")
            return [], 0
    
    def detect_persons_with_roi(self, img: np.ndarray, camera_id: str) -> Tuple[List[Tuple[int, int, int, int, float]], int]:
        """Detect persons with ROI support - DEPRECATED
        
        This method is deprecated. Use detect_persons() directly on ROI image
        and adjust coordinates in the calling code for better separation of concerns.
        
        Args:
            img: Input image as numpy array
            camera_id: Camera identifier for ROI configuration
            
        Returns:
            Tuple of (person_boxes, person_count)
        """
        # Deprecated: ROI handling should be done at the frame processing level
        return self.detect_persons(img)
    
    def draw_person_detections(self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """Draw person detection boxes on frame
        
        Args:
            frame: Image frame to draw on
            person_boxes: List of person detection boxes
            
        Returns:
            Frame with detections drawn
        """
        for x1, y1, x2, y2, conf in person_boxes:
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for person detection
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"[INFO] Person detection confidence threshold set to {threshold}")
        else:
            print(f"[ERROR] Invalid threshold value: {threshold}. Must be between 0.0 and 1.0")
    
    def get_detection_stats(self, person_boxes: List[Tuple[int, int, int, int, float]]) -> dict:
        """Get detection statistics
        
        Args:
            person_boxes: List of person detection boxes
            
        Returns:
            Dictionary with detection statistics
        """
        return {
            'person_count': len(person_boxes),
            'avg_confidence': sum(box[4] for box in person_boxes) / len(person_boxes) if person_boxes else 0.0,
            'max_confidence': max(box[4] for box in person_boxes) if person_boxes else 0.0,
            'min_confidence': min(box[4] for box in person_boxes) if person_boxes else 0.0
        }
    
    def is_model_loaded(self) -> bool:
        """Check if YOLO model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def reload_model(self, model_path: str = 'yolov8n.pt'):
        """Reload YOLO model
        
        Args:
            model_path: Path to YOLO model file
        """
        self._setup_yolo(model_path)

# Global instance
person_detection_service = PersonDetectionService(confidence_threshold = 0.90)