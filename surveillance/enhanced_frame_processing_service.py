import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import threading
import math
from scipy.spatial import distance as dist

# Importar servicios existentes
from .person_detection_service import person_detection_service
from .face_detection_service import face_detection_service
from services.camera_stream import stream_recognizer
from config.camera_config import CAMERAS
import pytz

# Configure Bogotá, Colombia timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class TrackedPerson:
    """Represents a tracked person with identity and tracking information"""
    def __init__(self, person_id: int, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], 
                 name: str = "Unknown", confidence: float = 0.0):
        self.person_id = person_id
        self.centroid = centroid
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.name = name
        self.confidence = confidence
        self.last_seen = time.time()
        self.frames_without_face = 0
        self.total_frames_tracked = 1
        self.tracker = None  # OpenCV tracker instance
        self.tracker_initialized = False
        self.identity_confirmed = name != "Unknown"
        self.identity_persistence_frames = 0  # Frames since identity was confirmed
        
    def update_position(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]):
        """Update position and tracking info"""
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = time.time()
        self.total_frames_tracked += 1
        
    def update_identity(self, name: str, confidence: float):
        """Update identity information when face is detected"""
        # Actualizando identidad de persona
        if name != "Unknown":
            old_name = self.name
            old_confidence = self.confidence
            self.name = name
            # Actualizar confianza solo si es mayor, pero mantener la identidad
            if confidence > self.confidence:
                self.confidence = confidence
            self.identity_confirmed = True
            self.identity_persistence_frames = 0
            self.frames_without_face = 0
            # Identidad actualizada
        
    def increment_frames_without_face(self):
        """Increment counter for frames without face detection"""
        self.frames_without_face += 1
        if self.identity_confirmed:
            self.identity_persistence_frames += 1
            
    def should_maintain_identity(self, max_frames_without_face: int = 30) -> bool:
        """Check if identity should be maintained despite no face detection"""
        return (self.identity_confirmed and 
                self.identity_persistence_frames < max_frames_without_face)

class EnhancedFrameProcessingResult:
    """Container for enhanced frame processing results with tracking"""
    def __init__(self, camera_id: str, original_frame: np.ndarray, processed_frame: np.ndarray,
                 person_boxes: List, person_count: int, face_results: List[Dict],
                 detected_names: List[str], timestamp: datetime, frame_id: int = 0,
                 tracked_persons: List[TrackedPerson] = None):
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
        self.tracked_persons = tracked_persons or []
        
        # Include tracked identities in detected names
        self.all_detected_names = self._get_all_detected_names()
        
    def _get_all_detected_names(self) -> List[str]:
        """Get all detected names including tracked identities"""
        all_names = list(self.detected_names)  # Direct face detections
        # Nombres de detección directa
        
        # Add tracked identities that are not already in face detections
        for tracked_person in self.tracked_persons:
            if (tracked_person.name != "Unknown" and 
                tracked_person.name not in all_names and
                tracked_person.should_maintain_identity()):
                all_names.append(tracked_person.name)
                # Agregando nombre de tracking
        
        # Todos los nombres detectados
        return all_names

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
        
        # Names detection reporting
        self.names_reporting_active = {}  # camera_id -> bool
        self.names_reporting_threads = {}  # camera_id -> threading.Thread
        self.last_reported_names = {}  # camera_id -> set of names
        
        # Camera-specific configurations (loaded from camera_config.py)
        self.camera_configs = {}  # camera_id -> detection_config
        self._load_camera_configs()
        
        # Person tracking configuration
        self.tracked_persons = {}  # camera_id -> Dict[person_id, TrackedPerson]
        self.next_person_id = {}  # camera_id -> int
        self.max_disappeared_frames = 30  # Frames before removing a tracked person
        self.identity_persistence_frames = 30  # Frames to maintain identity without face
        
        # OpenCV tracker configuration
        self.use_opencv_trackers = True
        self.tracker_type = "KCF"  # KCF is more stable than CSRT in OpenCV 4.x
        
        # Enhanced Frame Processing Service with Tracking initialized
        self._verify_tracker_availability()
        
    def _verify_tracker_availability(self):
        """Verify if OpenCV trackers are available"""
        try:
            # Test tracker creation
            test_tracker = cv2.legacy.TrackerKCF_create()
            # OpenCV legacy trackers available
        except Exception as e:
            # OpenCV trackers not available, disabling OpenCV tracking
            self.use_opencv_trackers = False
    
    def _start_names_reporting(self, camera_id: str):
        """Start periodic names reporting for a camera"""
        if camera_id in self.names_reporting_active and self.names_reporting_active[camera_id]:
            return
        
        self.names_reporting_active[camera_id] = True
        self.last_reported_names[camera_id] = set()
        
        def names_reporting_loop():
            while self.names_reporting_active.get(camera_id, False):
                try:
                    # Get current tracked persons
                    tracked_persons = self.tracked_persons.get(camera_id, {})
                    current_names = set()
                    
                    for person in tracked_persons.values():
                        if person.identity_confirmed and person.name != "Unknown":
                            current_names.add(person.name)
                    
                    # Only report if names have changed
                    if current_names != self.last_reported_names[camera_id]:
                        if current_names:
                            names_list = list(current_names)
                            print(f"[DETECCIÓN] Cámara {camera_id} - Nombres detectados: {', '.join(names_list)}")
                        else:
                            print(f"[DETECCIÓN] Cámara {camera_id} - No hay personas identificadas")
                        
                        self.last_reported_names[camera_id] = current_names.copy()
                    
                    time.sleep(1.0)  # Report every second
                    
                except Exception as e:
                    print(f"[ERROR] Error en reporte de nombres para cámara {camera_id}: {e}")
                    time.sleep(1.0)
        
        thread = threading.Thread(target=names_reporting_loop, daemon=True)
        thread.start()
        self.names_reporting_threads[camera_id] = thread
    
    def _stop_names_reporting(self, camera_id: str):
        """Stop periodic names reporting for a camera"""
        if camera_id in self.names_reporting_active:
            self.names_reporting_active[camera_id] = False
        
        if camera_id in self.names_reporting_threads:
            thread = self.names_reporting_threads[camera_id]
            if thread.is_alive():
                thread.join(timeout=2.0)
            del self.names_reporting_threads[camera_id]
        
        if camera_id in self.last_reported_names:
            del self.last_reported_names[camera_id]
    
    def _load_camera_configs(self):
        """Load camera-specific detection configurations"""
        for camera_id, camera_config in CAMERAS.items():
            detection_config = camera_config.get('detection_config', {})
            
            # Set default values if not specified
            self.camera_configs[camera_id] = {
                'min_face_size': detection_config.get('min_face_size', 30),
                'max_distance_threshold': detection_config.get('max_distance_threshold', 250),
                'person_confidence': detection_config.get('person_confidence', 0.5),
                'face_confidence': detection_config.get('face_confidence', 0.6)
            }
            
            # Camera configuration loaded
    
    def get_camera_config(self, camera_id: str, config_key: str, default_value=None):
        """Get specific configuration value for a camera"""
        return self.camera_configs.get(camera_id, {}).get(config_key, default_value)
    
    def update_camera_config(self, camera_id: str, config_key: str, value):
        """Update specific configuration value for a camera"""
        if camera_id not in self.camera_configs:
            self.camera_configs[camera_id] = {}
        
        self.camera_configs[camera_id][config_key] = value
        # Updated camera config
    
    def set_min_face_size(self, min_size: int, camera_id: str = None):
        """Set the minimum face size for detection
        
        Args:
            min_size: Minimum size in pixels (width or height) for valid face detection
            camera_id: Specific camera ID, if None applies to all cameras
        """
        if min_size < 10:
            # Minimum face size too small, setting to 10 pixels
            min_size = 10
        elif min_size > 200:
            # Minimum face size too large, setting to 200 pixels
            min_size = 200
        
        if camera_id:
            self.update_camera_config(camera_id, 'min_face_size', min_size)
        else:
            # Apply to all cameras
            for cam_id in self.camera_configs.keys():
                self.update_camera_config(cam_id, 'min_face_size', min_size)
    
    def get_min_face_size(self, camera_id: str = None) -> int:
        """Get the current minimum face size for a specific camera"""
        if camera_id:
            return self.get_camera_config(camera_id, 'min_face_size', 30)
        else:
            # Return the first camera's config as default
            first_camera = next(iter(self.camera_configs.keys()), None)
            return self.get_camera_config(first_camera, 'min_face_size', 30) if first_camera else 30
    
    def set_max_distance_threshold(self, max_distance: int, camera_id: str = None):
        """Set the maximum distance threshold for face-person association
        
        Args:
            max_distance: Maximum distance in pixels
            camera_id: Specific camera ID, if None applies to all cameras
        """
        if camera_id:
            self.update_camera_config(camera_id, 'max_distance_threshold', max_distance)
        else:
            # Apply to all cameras
            for cam_id in self.camera_configs.keys():
                self.update_camera_config(cam_id, 'max_distance_threshold', max_distance)
    
    def get_max_distance_threshold(self, camera_id: str = None) -> int:
        """Get the current maximum distance threshold for a specific camera"""
        if camera_id:
            return self.get_camera_config(camera_id, 'max_distance_threshold', 250)
        else:
            # Return the first camera's config as default
            first_camera = next(iter(self.camera_configs.keys()), None)
            return self.get_camera_config(first_camera, 'max_distance_threshold', 250) if first_camera else 250

    def start_processing_for_camera(self, camera_id: str):
        """Start background processing for a camera"""
        if camera_id in self.processing_active and self.processing_active[camera_id]:
            # Processing already active for camera
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
        
        # Initialize tracking data structures
        if camera_id not in self.tracked_persons:
            self.tracked_persons[camera_id] = {}
        
        if camera_id not in self.next_person_id:
            self.next_person_id[camera_id] = 1
        
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
        
        # Start names reporting for this camera
        self._start_names_reporting(camera_id)
        
        # Enhanced frame processing with tracking started
    
    def stop_processing_for_camera(self, camera_id: str):
        """Stop background processing for a camera"""
        if camera_id in self.processing_active:
            self.processing_active[camera_id] = False
            # Enhanced frame processing stopped
        
        # Stop names reporting
        self._stop_names_reporting(camera_id)
        
        # Cleanup tracking data
        if camera_id in self.tracked_persons:
            # Release OpenCV trackers
            for person in self.tracked_persons[camera_id].values():
                if person.tracker is not None:
                    del person.tracker
            self.tracked_persons[camera_id].clear()
            
        # Tracking data cleaned
    
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
        # Enhanced background processor started
        
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
                import traceback
                # Error in enhanced background processor
                time.sleep(0.1)
        
        # Enhanced background processor stopped
    
    def _process_frame_internal(self, frame: np.ndarray, camera_id: str, frame_id: int = 0) -> EnhancedFrameProcessingResult:
        """Internal frame processing logic with person tracking"""
        timestamp = datetime.now(BOGOTA_TZ)
        
        # Get camera-specific configurations
        min_face_size = self.get_camera_config(camera_id, 'min_face_size', 30)
        person_confidence = self.get_camera_config(camera_id, 'person_confidence', 0.5)
        face_confidence = self.get_camera_config(camera_id, 'face_confidence', 0.6)
        
        # Apply ROI using stream_recognizer
        roi_img, roi_info = stream_recognizer._apply_roi(frame, camera_id)
        
        # Detect persons in ROI image with camera-specific confidence
        person_detection_service.confidence_threshold = person_confidence
        person_boxes, person_count = person_detection_service.detect_persons(roi_img)
        if roi_info and person_boxes:
            person_boxes = self._adjust_person_boxes_for_roi(person_boxes, roi_info)
        
        # Detect and recognize faces using modular service in ROI with camera-specific settings
        face_results = face_detection_service.process_faces_in_image(
            roi_img, 
            min_face_size=min_face_size
        )
        
        # Adjust face coordinates for original image if ROI was applied
        if roi_info and face_results:
            face_results = self._adjust_faces_for_roi(face_results, roi_info)
        
        # Extract detected names
        detected_names = [face['name'] for face in face_results]
        # Nombres detectados directamente
        
        # Update person tracking
        tracked_persons_list = self._update_person_tracking(
            camera_id, frame, person_boxes, face_results
        )
        
        # Debug tracking results
        # Personas trackeadas después de actualización
        
        # Create processed frame with tracking information
        processed_frame = self._create_enhanced_processed_frame(
            frame, person_boxes, face_results, tracked_persons_list, 
            camera_id, timestamp, roi_info, frame_id
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
            tracked_persons=tracked_persons_list
        )
    

    
    def _update_person_tracking(self, camera_id: str, frame: np.ndarray, 
                                person_boxes: List, face_results: List[Dict]) -> List[TrackedPerson]:
        """Update person tracking with new detections"""
        if camera_id not in self.tracked_persons:
            self.tracked_persons[camera_id] = {}
            
        current_tracked = self.tracked_persons[camera_id]
        
        # Convert person boxes to centroids
        input_centroids = []
        input_boxes = []
        
        for box in person_boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                input_centroids.append((cx, cy))
                input_boxes.append((x1, y1, x2, y2))
        
        # If no existing tracked persons, create new ones
        if len(current_tracked) == 0:
            for i, (centroid, bbox) in enumerate(zip(input_centroids, input_boxes)):
                person_id = self.next_person_id[camera_id]
                self.next_person_id[camera_id] += 1
                
                tracked_person = TrackedPerson(person_id, centroid, bbox)
                self._initialize_opencv_tracker(tracked_person, frame)
                current_tracked[person_id] = tracked_person
        else:
            # Update existing trackers with OpenCV tracking
            self._update_opencv_trackers(current_tracked, frame)
            
            # Match detections with existing tracked persons using centroid distance
            self._match_detections_to_tracked_persons(
                camera_id, current_tracked, input_centroids, input_boxes, frame
            )
        
        # Associate faces with tracked persons
        self._associate_faces_with_tracked_persons(camera_id, current_tracked, face_results)
        
        # Remove disappeared persons
        self._remove_disappeared_persons(current_tracked)
        
        return list(current_tracked.values())

    def _draw_face_with_color(self, frame: np.ndarray, face_info: Dict[str, Any], color: Tuple[int, int, int]) -> np.ndarray:
        """Draw face detection with specified color
        
        Args:
            frame: Image frame to draw on
            face_info: Face information dictionary
            color: BGR color tuple for drawing
            
        Returns:
            Frame with detection drawn
        """
        bbox = face_info['bbox']
        name = face_info['name']
        confidence = face_info['confidence']
        
        # Draw rectangle around face
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Determine label to show
        label = f"{name}"
        if confidence > 0:
            label += f" ({confidence:.2f})"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10), 
                     (bbox[0] + text_width, bbox[1]), color, -1)
        
        # Text
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def _initialize_opencv_tracker(self, person: TrackedPerson, frame: np.ndarray):
        """Initialize OpenCV tracker for a person"""
        if not self.use_opencv_trackers:
            return
            
        try:
            # Create tracker based on type using new OpenCV 4.x API
            if self.tracker_type == "CSRT":
                tracker = cv2.legacy.TrackerCSRT_create()
            elif self.tracker_type == "KCF":
                tracker = cv2.legacy.TrackerKCF_create()
            elif self.tracker_type == "MOSSE":
                tracker = cv2.legacy.TrackerMOSSE_create()
            else:
                # Fallback to KCF if CSRT is not available
                try:
                    tracker = cv2.legacy.TrackerCSRT_create()
                except:
                    tracker = cv2.legacy.TrackerKCF_create()
            
            # Initialize tracker with bounding box
            x1, y1, x2, y2 = tracked_person.bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
            
            success = tracker.init(frame, bbox)
            if success:
                tracked_person.tracker = tracker
                tracked_person.tracker_initialized = True
                # Tracker initialized for person
            else:
                # Failed to initialize tracker for person
                 pass
            
        except Exception as e:
            # Failed to initialize tracker
            # Disable OpenCV tracking if initialization fails
            tracked_person.tracker_initialized = False
    
    def _update_opencv_trackers(self, tracked_persons: Dict[int, TrackedPerson], frame: np.ndarray):
        """Update all OpenCV trackers"""
        for person in tracked_persons.values():
            if person.tracker is not None and person.tracker_initialized:
                try:
                    success, bbox = person.tracker.update(frame)
                    if success:
                        x, y, w, h = bbox
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        centroid = (int(x + w/2), int(y + h/2))
                        person.update_position(centroid, (x1, y1, x2, y2))
                    else:
                        # Tracker failed, will rely on detection matching
                        person.tracker_initialized = False
                except Exception as e:
                    # Tracker update failed
                    person.tracker_initialized = False
    
    def _match_detections_to_tracked_persons(self, camera_id: str, tracked_persons: Dict[int, TrackedPerson], 
                                           input_centroids: List, input_boxes: List, frame: np.ndarray):
        """Match new detections to existing tracked persons"""
        if len(input_centroids) == 0:
            # No detections, increment frames without detection for all
            for person in tracked_persons.values():
                person.increment_frames_without_face()
            return
        
        # Get existing centroids
        existing_centroids = [person.centroid for person in tracked_persons.values()]
        person_ids = list(tracked_persons.keys())
        
        if len(existing_centroids) == 0:
             # No existing persons, create new ones
             for centroid, bbox in zip(input_centroids, input_boxes):
                 person_id = self.next_person_id[camera_id]
                 self.next_person_id[camera_id] += 1
                 tracked_person = TrackedPerson(person_id, centroid, bbox)
                 self._initialize_opencv_tracker(tracked_person, frame)
                 tracked_persons[person_id] = tracked_person
             return
        
        # Compute distance matrix
        D = dist.cdist(np.array(existing_centroids), np.array(input_centroids))
        
        # Find minimum values and sort by distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_row_indices = set()
        used_col_indices = set()
        
        # Update matched persons
        for (row, col) in zip(rows, cols):
            if row in used_row_indices or col in used_col_indices:
                continue
                
            max_distance_threshold = self.get_camera_config(camera_id, 'max_distance_threshold', 250)
            if D[row, col] <= max_distance_threshold:
                person_id = person_ids[row]
                person = tracked_persons[person_id]
                person.update_position(input_centroids[col], input_boxes[col])
                
                # Reinitialize tracker if needed
                if not person.tracker_initialized:
                    self._initialize_opencv_tracker(person, frame)
                
                used_row_indices.add(row)
                used_col_indices.add(col)
        
        # Create new tracked persons for unmatched detections
        unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
        for col in unused_col_indices:
            person_id = self.next_person_id[camera_id]
            self.next_person_id[camera_id] += 1
            tracked_person = TrackedPerson(person_id, input_centroids[col], input_boxes[col])
            self._initialize_opencv_tracker(tracked_person, frame)
            tracked_persons[person_id] = tracked_person
        
        # Mark unmatched existing persons as not detected
        unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
        for row in unused_row_indices:
            person_id = person_ids[row]
            tracked_persons[person_id].increment_frames_without_face()
    
    def _associate_faces_with_tracked_persons(self, camera_id: str, tracked_persons: Dict[int, TrackedPerson], 
                                            face_results: List[Dict]):
        """Associate detected faces with tracked persons"""
        
        # Iniciando asociación de rostros con personas trackeadas
        
        for i, face in enumerate(face_results):
            face_bbox = face['bbox']  # [x1, y1, x2, y2]
            face_centroid = (
                int((face_bbox[0] + face_bbox[2]) / 2),
                int((face_bbox[1] + face_bbox[3]) / 2)
            )
            
            # Procesando rostro
            
            # Find closest tracked person
            min_distance = float('inf')
            closest_person = None
            
            max_distance_threshold = self.get_camera_config(camera_id, 'max_distance_threshold', 250)
            
            for person in tracked_persons.values():
                distance = math.sqrt(
                    (person.centroid[0] - face_centroid[0]) ** 2 + 
                    (person.centroid[1] - face_centroid[1]) ** 2
                )
                
                # Calculando distancia a persona
                
                if distance < min_distance and distance < max_distance_threshold:
                    min_distance = distance
                    closest_person = person
            
            # Update person identity if face is associated
            if closest_person and face['name'] != "Unknown":
                confidence = face.get('confidence', 0.0)
                # Asociando rostro con persona
                closest_person.update_identity(face['name'], confidence)
            elif closest_person:
                # Rostro desconocido asociado con persona
                pass
            else:
                # Rostro no pudo ser asociado con ninguna persona trackeada
                pass
    
    def _remove_disappeared_persons(self, tracked_persons: Dict[int, TrackedPerson]):
        """Remove persons that have disappeared for too long"""
        to_remove = []
        
        for person_id, person in tracked_persons.items():
            if person.frames_without_face > self.max_disappeared_frames:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            person = tracked_persons[person_id]
            if person.tracker is not None:
                del person.tracker
            del tracked_persons[person_id]
    
    def _create_enhanced_processed_frame(self, frame: np.ndarray, person_boxes: List, 
                                       face_results: List[Dict], tracked_persons: List[TrackedPerson],
                                       camera_id: str, timestamp: datetime, 
                                       roi_info: Dict = None, frame_id: int = 0) -> np.ndarray:
        """Create processed frame with tracking information"""
        processed_frame = frame.copy()
        
        # Draw tracked persons with enhanced information
        for person in tracked_persons:
            x1, y1, x2, y2 = person.bbox
            
            # Choose color based on identity status
            if person.identity_confirmed:
                color = (0, 255, 0)  # Green for identified persons
                label = f"ID:{person.person_id} - {person.name}"
            else:
                color = (0, 165, 255)  # Orange for unknown persons
                label = f"ID:{person.person_id} - Unknown"
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid
            cv2.circle(processed_frame, person.centroid, 5, color, -1)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(processed_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add tracking info
            info_text = f"Frames: {person.total_frames_tracked}"
            if person.identity_confirmed:
                info_text += f" | Conf: {person.confidence:.2f}"
            cv2.putText(processed_frame, info_text, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw face detections with enhanced logic for person association
        for face_info in face_results:
            # Check if this face is within a tracked person's bounding box
            face_bbox = face_info['bbox']
            face_centroid = (
                int((face_bbox[0] + face_bbox[2]) / 2),
                int((face_bbox[1] + face_bbox[3]) / 2)
            )
            
            # Find if face is within any identified person's bounding box
            within_identified_person = False
            for person in tracked_persons:
                if person.identity_confirmed:
                    px1, py1, px2, py2 = person.bbox
                    # Check if face centroid is within person's bounding box
                    if px1 <= face_centroid[0] <= px2 and py1 <= face_centroid[1] <= py2:
                        within_identified_person = True
                        break
            
            # Draw face with appropriate color logic
            if within_identified_person:
                # Face is within an identified person - show as green and change status
                # Create a modified face_info with updated name if it was Unknown
                modified_face_info = face_info.copy()
                if face_info['name'] == "Unknown":
                    modified_face_info['name'] = "Identificado por contexto"
                    modified_face_info['confidence'] = 1.0  # Set high confidence for context identification
                
                self._draw_face_with_color(processed_frame, modified_face_info, (0, 255, 0))
            else:
                # Use normal face detection drawing
                processed_frame = face_detection_service.draw_face_detection(processed_frame, face_info)
        
        # Add statistics
        roi_status = "ROI: ON" if roi_info else "ROI: OFF"
        tracked_count = len(tracked_persons)
        identified_count = len([p for p in tracked_persons if p.identity_confirmed and p.should_maintain_identity(self.identity_persistence_frames)])
        
        stats_text = f"Tracked: {tracked_count} | Identified: {identified_count} | Faces: {len(face_results)} | {roi_status}"
        cv2.putText(processed_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp_str, (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add mode indicator
        cv2.putText(processed_frame, f"TRACKING MODE ({self.tracker_type})", (10, processed_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw ROI
        processed_frame = stream_recognizer._draw_roi_on_frame(processed_frame, camera_id)
        
        return processed_frame
    
    def _adjust_person_boxes_for_roi(self, person_boxes: List, roi_info: Dict) -> List:
        """Adjust person box coordinates from ROI to original image"""
        if not roi_info:
            return person_boxes
        
        x_offset = roi_info.get('x', 0)
        y_offset = roi_info.get('y', 0)
        
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
        
        x_offset = roi_info.get('x', 0)
        y_offset = roi_info.get('y', 0)
        
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
        """Get enhanced processing statistics including tracking"""
        enhanced_stats = {
            "active_cameras": list(self.processing_active.keys()),
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
            },
            "tracking_statistics": {
                camera_id: {
                    "total_tracked_persons": len(self.tracked_persons.get(camera_id, {})),
                    "identified_persons": len([
                        p for p in self.tracked_persons.get(camera_id, {}).values() 
                        if p.identity_confirmed and p.should_maintain_identity(self.identity_persistence_frames)
                    ]),
                    "active_trackers": len([
                        p for p in self.tracked_persons.get(camera_id, {}).values() 
                        if p.tracker_initialized
                    ]),
                    "next_person_id": self.next_person_id.get(camera_id, 1)
                }
                for camera_id in self.processing_active.keys()
            },
            "tracking_configuration": {
                "tracker_type": self.tracker_type,
                "use_opencv_trackers": self.use_opencv_trackers,
                "max_disappeared_frames": self.max_disappeared_frames,
                "identity_persistence_frames": self.identity_persistence_frames
            },
            "camera_specific_configs": {
                camera_id: {
                    "min_face_size": self.get_camera_config(camera_id, 'min_face_size', 30),
                    "max_distance_threshold": self.get_camera_config(camera_id, 'max_distance_threshold', 250),
                    "person_confidence": self.get_camera_config(camera_id, 'person_confidence', 0.5),
                    "face_confidence": self.get_camera_config(camera_id, 'face_confidence', 0.6)
                }
                for camera_id in self.camera_configs.keys()
            }
        }
        
        return enhanced_stats

# Global instance
enhanced_frame_processing_service = EnhancedFrameProcessingService()