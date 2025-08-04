import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import threading

# Importar servicios existentes
from .person_detection_service import person_detection_service
from .face_detection_service import face_detection_service
from .person_tracking_service import person_tracking_service
from services.camera_stream import stream_recognizer
from config.settings import CAMERAS
import pytz

# Configure Bogotá, Colombia timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class EnhancedFrameProcessingResult:
    """Container for enhanced frame processing results with tracking"""
    def __init__(self, camera_id: str, original_frame: np.ndarray, processed_frame: np.ndarray,
                 person_boxes: List, person_count: int, face_results: List[Dict],
                 detected_names: List[str], timestamp: datetime, frame_id: int = 0,
                 tracking_results: List[Dict] = None):
        self.camera_id = camera_id
        self.original_frame = original_frame
        self.processed_frame = processed_frame
        self.person_boxes = person_boxes
        self.person_count = person_count
        self.face_results = face_results
        self.detected_names = detected_names
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.face_count = len(face_results)
        self.tracking_results = tracking_results or []
        
        # Extraer nombres persistentes del tracking
        self.persistent_names = []
        if tracking_results:
            for track in tracking_results:
                if track['is_persistent'] and track['name'] != "Unknown":
                    self.persistent_names.append(track['name'])
        
        # Combinar nombres detectados y persistentes
        self.all_detected_names = list(set(detected_names + self.persistent_names))

class EnhancedFrameProcessingService:
    """Enhanced frame processing service with person tracking capabilities"""
    
    def __init__(self):
        # Processing cache and timing
        self.processing_cache = {}  # camera_id -> EnhancedFrameProcessingResult
        self.cache_timestamps = {}  # camera_id -> timestamp
        self.cache_duration = 0.5  # Cache results for 500ms
        
        # Processing locks per camera
        self.processing_locks = {}  # camera_id -> threading.Lock
        
        # Frame queues for processing
        self.frame_queues = {}  # camera_id -> deque
        self.max_queue_size = 3  # Reduced queue size to prevent lag
        
        # Frame synchronization
        self.frame_counters = {}  # camera_id -> frame_counter
        self.last_processed_frame = {}  # camera_id -> frame_id
        
        # Processing threads
        self.processing_threads = {}  # camera_id -> threading.Thread
        self.processing_active = {}  # camera_id -> bool
        
        # Enhanced tracking settings
        self.enable_tracking = True
        self.tracking_mode = "enhanced"  # "basic" or "enhanced"
        
        print("[INFO] Enhanced Frame Processing Service initialized with tracking")
    
    def set_tracking_mode(self, mode: str):
        """Set tracking mode: 'basic' (original) or 'enhanced' (with tracking)"""
        if mode in ["basic", "enhanced"]:
            self.tracking_mode = mode
            print(f"[INFO] Tracking mode set to: {mode}")
        else:
            print(f"[WARNING] Invalid tracking mode: {mode}. Using 'enhanced'")
    
    def start_processing_for_camera(self, camera_id: str):
        """Start background processing for a camera"""
        if camera_id in self.processing_active and self.processing_active[camera_id]:
            print(f"[INFO] Processing already active for camera {camera_id}")
            return
        
        # Initialize camera-specific data structures
        if camera_id not in self.processing_locks:
            self.processing_locks[camera_id] = threading.Lock()
        
        if camera_id not in self.frame_queues:
            self.frame_queues[camera_id] = deque(maxlen=self.max_queue_size)
        
        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0
        
        if camera_id not in self.last_processed_frame:
            self.last_processed_frame[camera_id] = 0
        
        # Start background processing thread
        self.processing_active[camera_id] = True
        thread = threading.Thread(
            target=self._background_processor,
            args=(camera_id,),
            daemon=True,
            name=f"EnhancedFrameProcessor-{camera_id}"
        )
        thread.start()
        self.processing_threads[camera_id] = thread
        
        print(f"[INFO] Enhanced frame processing started for camera {camera_id}")
    
    def stop_processing_for_camera(self, camera_id: str):
        """Stop background processing for a camera"""
        if camera_id in self.processing_active:
            self.processing_active[camera_id] = False
            print(f"[INFO] Enhanced frame processing stopped for camera {camera_id}")
        
        # Cleanup tracking data
        person_tracking_service.cleanup_camera(camera_id)
    
    def submit_frame_for_processing(self, frame: np.ndarray, camera_id: str) -> int:
        """Submit a frame for background processing"""
        if camera_id not in self.processing_active or not self.processing_active[camera_id]:
            self.start_processing_for_camera(camera_id)
        
        # Increment frame counter
        self.frame_counters[camera_id] += 1
        frame_id = self.frame_counters[camera_id]
        
        # Add frame to queue with thread safety
        with self.processing_locks[camera_id]:
            # Clear queue if it's full to prevent lag
            if len(self.frame_queues[camera_id]) >= self.max_queue_size:
                self.frame_queues[camera_id].clear()
            
            self.frame_queues[camera_id].append((frame.copy(), frame_id))
        
        return frame_id
    
    def get_latest_processing_result(self, camera_id: str) -> Optional[EnhancedFrameProcessingResult]:
        """Get the latest processing result for a camera"""
        current_time = time.time()
        
        # Check if we have a recent cached result
        if (camera_id in self.processing_cache and 
            camera_id in self.cache_timestamps and
            current_time - self.cache_timestamps[camera_id] < self.cache_duration):
            return self.processing_cache[camera_id]
        
        return None
    
    def process_frame_immediately(self, frame: np.ndarray, camera_id: str) -> EnhancedFrameProcessingResult:
        """Process a frame immediately without queuing"""
        frame_id = self.frame_counters.get(camera_id, 0) + 1
        self.frame_counters[camera_id] = frame_id
        
        return self._process_frame_internal(frame, camera_id, frame_id)
    
    def _background_processor(self, camera_id: str):
        """Background processing loop for a camera"""
        print(f"[INFO] Enhanced background processor started for camera {camera_id}")
        
        while self.processing_active.get(camera_id, False):
            try:
                frame_data = None
                
                # Get frame from queue with thread safety
                with self.processing_locks[camera_id]:
                    if self.frame_queues[camera_id]:
                        frame_data = self.frame_queues[camera_id].popleft()
                
                if frame_data:
                    frame, frame_id = frame_data
                    current_time = time.time()
                    
                    # Process the frame
                    result = self._process_frame_internal(frame, camera_id, frame_id)
                    
                    # Cache the result with synchronization
                    with self.processing_locks[camera_id]:
                        self.processing_cache[camera_id] = result
                        self.cache_timestamps[camera_id] = current_time
                        self.last_processed_frame[camera_id] = frame_id
                else:
                    # No frames to process, sleep briefly
                    time.sleep(0.01)
            
            except Exception as e:
                print(f"[ERROR] Error in enhanced background processor for {camera_id}: {e}")
                time.sleep(0.1)
        
        print(f"[INFO] Enhanced background processor stopped for camera {camera_id}")
    
    def _process_frame_internal(self, frame: np.ndarray, camera_id: str, frame_id: int = 0) -> EnhancedFrameProcessingResult:
        """Internal frame processing logic with enhanced tracking"""
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
        
        # Enhanced tracking processing
        tracking_results = []
        if self.tracking_mode == "enhanced" and self.enable_tracking:
            try:
                tracking_results = person_tracking_service.update_tracks(
                    camera_id, person_boxes, face_results
                )
            except Exception as e:
                print(f"[ERROR] Error in tracking for {camera_id}: {e}")
        
        # Create processed frame with detections
        if self.tracking_mode == "enhanced" and tracking_results:
            processed_frame = self._create_enhanced_processed_frame(
                frame, person_boxes, face_results, tracking_results, camera_id, timestamp, roi_info, frame_id
            )
        else:
            processed_frame = self._create_basic_processed_frame(
                frame, person_boxes, face_results, camera_id, timestamp, roi_info, frame_id
            )
        
        return EnhancedFrameProcessingResult(
            camera_id=camera_id,
            original_frame=frame,
            processed_frame=processed_frame,
            person_boxes=person_boxes,
            person_count=person_count,
            face_results=face_results,
            detected_names=detected_names,
            timestamp=timestamp,
            frame_id=frame_id,
            tracking_results=tracking_results
        )
    
    def _create_enhanced_processed_frame(self, frame: np.ndarray, person_boxes: List, 
                                       face_results: List[Dict], tracking_results: List[Dict],
                                       camera_id: str, timestamp: datetime, 
                                       roi_info: Dict = None, frame_id: int = 0) -> np.ndarray:
        """Create processed frame with enhanced tracking visualization"""
        processed_frame = frame.copy()
        
        # Draw enhanced tracking detections
        processed_frame = person_tracking_service.draw_enhanced_detections(
            processed_frame, tracking_results
        )
        
        # Add enhanced statistics
        tracking_stats = person_tracking_service.get_tracking_stats(camera_id)
        roi_status = "ROI: ON" if roi_info else "ROI: OFF"
        
        stats_text = f"Persons: {len(person_boxes)} | Faces: {len(face_results)} | Tracks: {tracking_stats['active_tracks']} | {roi_status} | Frame: {frame_id}"
        cv2.putText(processed_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp_str, (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add tracking mode indicator
        cv2.putText(processed_frame, "ENHANCED TRACKING", (10, processed_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw ROI
        processed_frame = stream_recognizer._draw_roi_on_frame(processed_frame, camera_id)
        
        return processed_frame
    
    def _create_basic_processed_frame(self, frame: np.ndarray, person_boxes: List, 
                                    face_results: List[Dict], camera_id: str, 
                                    timestamp: datetime, roi_info: Dict = None, frame_id: int = 0) -> np.ndarray:
        """Create processed frame with basic detections (fallback mode)"""
        processed_frame = frame.copy()
        
        # Draw person detections
        processed_frame = person_detection_service.draw_person_detections(processed_frame, person_boxes)
        
        # Draw face detections
        for face_info in face_results:
            processed_frame = face_detection_service.draw_face_detection(processed_frame, face_info)
        
        # Add statistics
        roi_status = "ROI: ON" if roi_info else "ROI: OFF"
        stats_text = f"Persons: {len(person_boxes)} | Faces: {len(face_results)} | {roi_status} | Frame: {frame_id}"
        cv2.putText(processed_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp_str, (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add mode indicator
        cv2.putText(processed_frame, "BASIC MODE", (10, processed_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Draw ROI
        processed_frame = stream_recognizer._draw_roi_on_frame(processed_frame, camera_id)
        
        return processed_frame
    
    def _adjust_person_boxes_for_roi(self, person_boxes: List, roi_info: Dict) -> List:
        """Adjust person box coordinates from ROI to original image"""
        if not roi_info:
            return person_boxes
        
        x_offset = roi_info.get('x_offset', 0)
        y_offset = roi_info.get('y_offset', 0)
        
        adjusted_boxes = []
        for box in person_boxes:
            if len(box) == 5:  # (x1, y1, x2, y2, confidence)
                x1, y1, x2, y2, conf = box
                adjusted_boxes.append((
                    x1 + x_offset, y1 + y_offset,
                    x2 + x_offset, y2 + y_offset, conf
                ))
            else:  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = box
                adjusted_boxes.append((
                    x1 + x_offset, y1 + y_offset,
                    x2 + x_offset, y2 + y_offset
                ))
        
        return adjusted_boxes
    
    def _adjust_faces_for_roi(self, face_results: List[Dict], roi_info: Dict) -> List[Dict]:
        """Adjust face coordinates from ROI to original image"""
        if not roi_info:
            return face_results
        
        x_offset = roi_info.get('x_offset', 0)
        y_offset = roi_info.get('y_offset', 0)
        
        adjusted_faces = []
        for face in face_results:
            adjusted_face = face.copy()
            bbox = face['bbox']
            adjusted_face['bbox'] = [
                bbox[0] + x_offset, bbox[1] + y_offset,
                bbox[2] + x_offset, bbox[3] + y_offset
            ]
            adjusted_faces.append(adjusted_face)
        
        return adjusted_faces
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics"""
        basic_stats = {
            "active_cameras": list(self.processing_active.keys()),
            "tracking_mode": self.tracking_mode,
            "tracking_enabled": self.enable_tracking,
            "cache_status": {
                camera_id: {
                    "has_cache": camera_id in self.processing_cache,
                    "cache_age": time.time() - self.cache_timestamps.get(camera_id, 0),
                    "last_processed_frame": self.last_processed_frame.get(camera_id, 0),
                    "current_frame_counter": self.frame_counters.get(camera_id, 0)
                }
                for camera_id in self.processing_active.keys()
            },
            "queue_sizes": {
                camera_id: len(queue) 
                for camera_id, queue in self.frame_queues.items()
            },
            "frame_synchronization": {
                camera_id: {
                    "frames_submitted": self.frame_counters.get(camera_id, 0),
                    "frames_processed": self.last_processed_frame.get(camera_id, 0),
                    "frames_pending": self.frame_counters.get(camera_id, 0) - self.last_processed_frame.get(camera_id, 0)
                }
                for camera_id in self.processing_active.keys()
            }
        }
        
        # Add tracking statistics
        if self.tracking_mode == "enhanced":
            basic_stats["tracking_stats"] = {
                camera_id: person_tracking_service.get_tracking_stats(camera_id)
                for camera_id in self.processing_active.keys()
            }
        
        return basic_stats

# Global instance
enhanced_frame_processing_service = EnhancedFrameProcessingService()