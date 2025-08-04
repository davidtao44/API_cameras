"""Frame Processing Service

Centralized service for processing camera frames.
Handles face detection, person detection, and ROI processing.
Shared by both streaming and surveillance services for optimization.
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import deque
import pytz

# Import modular services
from .face_detection_service import face_detection_service
from .person_detection_service import person_detection_service
from services.camera_stream import stream_recognizer

# Configure timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class FrameProcessingResult:
    """Container for frame processing results"""
    def __init__(self, camera_id: str, original_frame: np.ndarray, processed_frame: np.ndarray,
                 person_boxes: List, person_count: int, face_results: List[Dict],
                 detected_names: List[str], timestamp: datetime):
        self.camera_id = camera_id
        self.original_frame = original_frame
        self.processed_frame = processed_frame
        self.person_boxes = person_boxes
        self.person_count = person_count
        self.face_results = face_results
        self.detected_names = detected_names
        self.timestamp = timestamp
        self.face_count = len(face_results)

class FrameProcessingService:
    """Centralized frame processing service"""
    
    def __init__(self):
        # Processing cache and timing
        self.processing_cache = {}  # camera_id -> FrameProcessingResult
        self.cache_timestamps = {}  # camera_id -> timestamp
        self.cache_duration = 0.5  # Cache results for 500ms
        
        # Processing locks per camera
        self.processing_locks = {}  # camera_id -> threading.Lock
        
        # Frame queues for processing
        self.frame_queues = {}  # camera_id -> deque
        self.max_queue_size = 5
        
        # Processing threads
        self.processing_threads = {}  # camera_id -> threading.Thread
        self.processing_active = {}  # camera_id -> bool
        
        print("[INFO] Frame Processing Service initialized")
    
    def start_processing_for_camera(self, camera_id: str):
        """Start background processing for a camera"""
        if camera_id not in self.processing_locks:
            self.processing_locks[camera_id] = threading.Lock()
        
        if camera_id not in self.frame_queues:
            self.frame_queues[camera_id] = deque(maxlen=self.max_queue_size)
        
        if camera_id not in self.processing_active or not self.processing_active[camera_id]:
            self.processing_active[camera_id] = True
            thread = threading.Thread(
                target=self._background_processor,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.processing_threads[camera_id] = thread
            print(f"[INFO] Started background processing for camera {camera_id}")
    
    def stop_processing_for_camera(self, camera_id: str):
        """Stop background processing for a camera"""
        if camera_id in self.processing_active:
            self.processing_active[camera_id] = False
        
        if camera_id in self.processing_threads:
            thread = self.processing_threads[camera_id]
            if thread.is_alive():
                thread.join(timeout=2.0)
            del self.processing_threads[camera_id]
        
        # Clean up resources
        if camera_id in self.processing_cache:
            del self.processing_cache[camera_id]
        if camera_id in self.cache_timestamps:
            del self.cache_timestamps[camera_id]
        if camera_id in self.frame_queues:
            del self.frame_queues[camera_id]
        
        print(f"[INFO] Stopped background processing for camera {camera_id}")
    
    def submit_frame_for_processing(self, frame: np.ndarray, camera_id: str) -> bool:
        """Submit a frame for background processing"""
        if camera_id not in self.frame_queues:
            return False
        
        try:
            # Add frame to queue (will automatically remove oldest if full)
            self.frame_queues[camera_id].append(frame.copy())
            return True
        except Exception as e:
            print(f"[ERROR] Error submitting frame for {camera_id}: {e}")
            return False
    
    def get_latest_processing_result(self, camera_id: str) -> Optional[FrameProcessingResult]:
        """Get the latest processing result for a camera"""
        current_time = time.time()
        
        # Check if we have a recent cached result
        if (camera_id in self.processing_cache and 
            camera_id in self.cache_timestamps and
            current_time - self.cache_timestamps[camera_id] < self.cache_duration):
            return self.processing_cache[camera_id]
        
        return None
    
    def process_frame_immediately(self, frame: np.ndarray, camera_id: str) -> FrameProcessingResult:
        """Process a frame immediately (synchronous)"""
        return self._process_frame_internal(frame, camera_id)
    
    def _background_processor(self, camera_id: str):
        """Background thread for processing frames"""
        while self.processing_active.get(camera_id, False):
            try:
                # Get frame from queue
                if camera_id in self.frame_queues and self.frame_queues[camera_id]:
                    frame = self.frame_queues[camera_id].popleft()
                    
                    # Process the frame
                    result = self._process_frame_internal(frame, camera_id)
                    
                    # Cache the result
                    with self.processing_locks[camera_id]:
                        self.processing_cache[camera_id] = result
                        self.cache_timestamps[camera_id] = time.time()
                else:
                    # No frames to process, sleep briefly
                    time.sleep(0.01)
            
            except Exception as e:
                print(f"[ERROR] Error in background processor for {camera_id}: {e}")
                time.sleep(0.1)
    
    def _process_frame_internal(self, frame: np.ndarray, camera_id: str) -> FrameProcessingResult:
        """Internal frame processing logic with proper ROI integration"""
        timestamp = datetime.now(BOGOTA_TZ)
        
        # Apply ROI using stream_recognizer
        roi_img, roi_info = stream_recognizer._apply_roi(frame, camera_id)
        
        # Detect persons in ROI image and adjust coordinates
        person_boxes, person_count = person_detection_service.detect_persons(roi_img)
        if roi_info and person_boxes:
            person_boxes = self._adjust_person_boxes_for_roi(person_boxes, roi_info)
        
        # Detect and recognize faces using modular service in ROI
        face_results = face_detection_service.process_faces_in_image(roi_img)
        
        # Adjust face coordinates for original image if ROI was applied
        if roi_info and face_results:
            face_results = self._adjust_faces_for_roi(face_results, roi_info)
        
        # Extract detected names
        detected_names = [face['name'] for face in face_results]
        
        # Create processed frame with detections
        processed_frame = self._create_processed_frame(
            frame, person_boxes, face_results, camera_id, timestamp, roi_info
        )
        
        return FrameProcessingResult(
            camera_id=camera_id,
            original_frame=frame,
            processed_frame=processed_frame,
            person_boxes=person_boxes,
            person_count=person_count,
            face_results=face_results,
            detected_names=detected_names,
            timestamp=timestamp
        )
    
    def _adjust_person_boxes_for_roi(self, person_boxes: List, roi_info: Dict) -> List:
        """Adjust person detection coordinates from ROI to original image coordinates"""
        if not roi_info or not person_boxes:
            return person_boxes
        
        x_offset = roi_info.get('x', 0)
        y_offset = roi_info.get('y', 0)
        
        adjusted_boxes = []
        for box in person_boxes:
            if len(box) >= 4:  # (x1, y1, x2, y2, confidence)
                x1, y1, x2, y2 = box[:4]
                adjusted_box = (
                    x1 + x_offset,
                    y1 + y_offset,
                    x2 + x_offset,
                    y2 + y_offset
                ) + box[4:]  # Mantener confidence y otros datos
                adjusted_boxes.append(adjusted_box)
        
        return adjusted_boxes
    
    def _create_processed_frame(self, frame: np.ndarray, person_boxes: List, 
                               face_results: List[Dict], camera_id: str, 
                               timestamp: datetime, roi_info: Dict = None) -> np.ndarray:
        """Create processed frame with all detections drawn"""
        processed_frame = frame.copy()
        
        # Draw person detections
        processed_frame = person_detection_service.draw_person_detections(processed_frame, person_boxes)
        
        # Draw face detections
        for face_info in face_results:
            processed_frame = face_detection_service.draw_face_detection(processed_frame, face_info)
        
        # Add statistics with ROI info
        roi_status = "ROI: ON" if roi_info else "ROI: OFF"
        stats_text = f"Persons: {len(person_boxes)} | Faces: {len(face_results)} | {roi_status}"
        cv2.putText(processed_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp_str, (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw ROI
        processed_frame = stream_recognizer._draw_roi_on_frame(processed_frame, camera_id)
        
        return processed_frame
    
    def _adjust_faces_for_roi(self, face_results: List[Dict], roi_info: Dict) -> List[Dict]:
        """Adjust face coordinates from ROI to original image coordinates"""
        if not roi_info:
            return face_results
        
        x_offset = roi_info.get('x', 0)
        y_offset = roi_info.get('y', 0)
        
        adjusted_results = []
        for face_info in face_results:
            adjusted_face = face_info.copy()
            bbox = adjusted_face['bbox']
            # Adjust bbox coordinates
            adjusted_face['bbox'] = [
                bbox[0] + x_offset,
                bbox[1] + y_offset,
                bbox[2] + x_offset,
                bbox[3] + y_offset
            ]
            adjusted_results.append(adjusted_face)
        
        return adjusted_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "active_cameras": list(self.processing_active.keys()),
            "cache_status": {
                camera_id: {
                    "has_cache": camera_id in self.processing_cache,
                    "cache_age": time.time() - self.cache_timestamps.get(camera_id, 0)
                }
                for camera_id in self.processing_active.keys()
            },
            "queue_sizes": {
                camera_id: len(queue) 
                for camera_id, queue in self.frame_queues.items()
            }
        }

# Global instance
frame_processing_service = FrameProcessingService()