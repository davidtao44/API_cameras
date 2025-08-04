"""Simplified Surveillance Service

Focuses only on person detection and recognition.
Detects known and unknown persons without access control or alarms.
"""

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import time
from typing import Tuple, List, Dict, Any
from datetime import datetime
import pytz
from collections import deque

# Import modular services
from .face_detection_service import face_detection_service
from .person_detection_service import person_detection_service
from .enhanced_frame_processing_service import enhanced_frame_processing_service
from .intrusion_detection_service import intrusion_detection_service

# Import existing services
from services.camera_stream import stream_recognizer
from config.settings import CAMERAS

# Configure Bogotá, Colombia timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class SurveillanceService:
    """Simplified surveillance service for person detection and recognition"""
    
    def __init__(self):
        # Core monitoring state
        self.monitoring_active = False
        self.monitoring_threads = {}
        
        # Frame processing configuration
        self.frame_skip_interval = 5
        self.frame_counters = {}
        
        # Detection tracking for statistics
        self.last_detection_time = {}
        
        print("[INFO] Surveillance service initialized for person detection")
    
    # ==================== MONITORING CONTROL ====================
    
    def start_monitoring(self, camera_id: str = "cam1"):
        """Start automatic monitoring of a specific camera"""
        if camera_id in self.monitoring_threads and self.monitoring_threads[camera_id].is_alive():
            print(f"[WARNING] Monitoring for {camera_id} is already active")
            return
        
        if camera_id not in CAMERAS:
            print(f"[ERROR] Camera {camera_id} not found in configuration")
            return
        
        print(f"[INFO] Starting automatic monitoring for {camera_id}")
        
        # Start enhanced frame processing service for this camera
        enhanced_frame_processing_service.start_processing_for_camera(camera_id)
        
        thread = threading.Thread(
            target=self._monitor_camera,
            args=(camera_id,),
            daemon=True,
            name=f"surveillance_{camera_id}"
        )
        thread.start()
        self.monitoring_threads[camera_id] = thread
        self.monitoring_active = True
    
    def stop_monitoring(self, camera_id: str = None):
        """Stop monitoring for a specific camera or all cameras"""
        if camera_id:
            if camera_id in self.monitoring_threads:
                print(f"[INFO] Stopping monitoring for {camera_id}")
                # Stop frame processing service for this camera
                enhanced_frame_processing_service.stop_processing_for_camera(camera_id)
        else:
            print("[INFO] Stopping all monitoring")
            self.monitoring_active = False
            # Stop frame processing service for all cameras
            for cam_id in CAMERAS.keys():
                enhanced_frame_processing_service.stop_processing_for_camera(cam_id)
    
    def _monitor_camera(self, camera_id: str):
        """Monitor a specific camera for surveillance events"""
        config = CAMERAS[camera_id]
        url = config["url"]
        
        # Initialize tracking for this camera
        self.frame_counters[camera_id] = 0
        
        try:
            if config.get("auth_required", True):
                user = config["username"]
                pwd = config["password"]
                frame_generator = self._stream_frames_with_digest(url, user, pwd)
            else:
                frame_generator = self._stream_frames_without_auth(camera_id)
            
            for frame in frame_generator:
                if not self.monitoring_active:
                    break
                
                try:
                    self.frame_counters[camera_id] += 1
                    
                    # Process every N frames
                    if self.frame_counters[camera_id] % self.frame_skip_interval == 0:
                        self._process_surveillance_frame(frame, camera_id)
                    
                    time.sleep(0.001)
                    
                except Exception as e:
                    print(f"[ERROR] Error processing frame for {camera_id}: {e}")
                    time.sleep(1)
        
        except Exception as e:
            print(f"[ERROR] Error in camera monitoring for {camera_id}: {e}")
        finally:
            self._cleanup_camera_resources(camera_id)
    
    def _process_surveillance_frame(self, frame: np.ndarray, camera_id: str):
        """Process a single frame for surveillance events"""
        # IMPORTANT: Don't submit additional frames to avoid queue conflicts
        # The stream service (camera_service.py) is already submitting frames to the processing queue
        # Submitting frames from both services causes intermediate frames and synchronization issues
        processing_result = enhanced_frame_processing_service.get_latest_processing_result(camera_id)
        
        if not processing_result:
            # If no recent result available, process immediately without adding to queue
            processing_result = enhanced_frame_processing_service.process_frame_immediately(frame, camera_id)
        
        person_boxes = processing_result.person_boxes
        face_results = processing_result.face_results
        processed_frame = processing_result.processed_frame
        person_count = processing_result.person_count
        face_count = processing_result.face_count
        detected_names = processing_result.detected_names
        
        # Update frame counter
        self.frame_counters[camera_id] = self.frame_counters.get(camera_id, 0) + 1
        
        # Analyze frame for intrusion patterns
        try:
            intrusion_events = intrusion_detection_service.analyze_detection_frame(
                camera_id=camera_id,
                person_count=person_count,
                face_count=face_count,
                detected_names=detected_names,
                timestamp=processing_result.timestamp
            )
            
            # Log intrusion events if any detected
            if intrusion_events:
                for event in intrusion_events:
                    print(f"[INTRUSION] {camera_id}: {event.event_type} - {event.description}")
                    
        except Exception as e:
            print(f"[ERROR] Error in intrusion detection for {camera_id}: {e}")
    
    # Removed _create_processed_frame and _adjust_faces_for_roi methods
    # These are now handled by the centralized frame_processing_service
    
    # ==================== UTILITY METHODS ====================
    
    def _cleanup_camera_resources(self, camera_id: str):
        """Clean up resources for a camera"""
        if camera_id in self.frame_counters:
            del self.frame_counters[camera_id]
        
        print(f"[INFO] Monitoring for {camera_id} stopped")

    def update_intrusion_config(self, new_config: Dict[str, Any]) -> bool:
        """Update intrusion detection configuration"""
        try:
            intrusion_detection_service.update_config(new_config)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to update intrusion config: {e}")
            return False
    
    def _stream_frames_without_auth(self, camera_id: str):
        """Basic frame streaming for surveillance monitoring"""
        try:
            if camera_id not in CAMERAS:
                print(f"[ERROR] No configuration found for camera {camera_id}")
                return
            
            camera_config = CAMERAS[camera_id]
            url = camera_config.get('url')
            if not url:
                print(f"[ERROR] No URL found for camera {camera_id}")
                return
            
            # Use requests for MJPEG streaming
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            response = session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Parse MJPEG stream
            boundary = None
            content_type = response.headers.get('content-type', '')
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].strip()
            
            if boundary:
                boundary = boundary.encode('utf-8')
                buffer = b''
                
                for chunk in response.iter_content(chunk_size=1024):
                    if not self.monitoring_active:
                        break
                    
                    buffer += chunk
                    
                    while boundary in buffer:
                        frame_end = buffer.find(boundary)
                        frame_data = buffer[:frame_end]
                        buffer = buffer[frame_end + len(boundary):]
                        
                        # Find JPEG data
                        jpeg_start = frame_data.find(b'\xff\xd8')
                        jpeg_end = frame_data.rfind(b'\xff\xd9')
                        
                        if jpeg_start != -1 and jpeg_end != -1:
                            jpeg_data = frame_data[jpeg_start:jpeg_end + 2]
                            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                yield frame
            else:
                # Fallback to cv2.VideoCapture for non-MJPEG streams
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print(f"[ERROR] Failed to open camera stream for {camera_id}")
                    return
                
                while self.monitoring_active:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[ERROR] Failed to read frame from {camera_id}")
                        break
                    
                    yield frame
                
                cap.release()
                
        except Exception as e:
            print(f"[ERROR] Error in frame streaming for {camera_id}: {e}")
        finally:
            if 'session' in locals():
                session.close()
    
    def _stream_frames_with_digest(self, url: str, username: str, password: str):
        """Stream frames with HTTP Digest authentication for MJPEG cameras"""
        try:
            session = requests.Session()
            session.auth = HTTPDigestAuth(username, password)
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            response = session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Parse MJPEG stream
            boundary = None
            content_type = response.headers.get('content-type', '')
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].strip()
            
            if not boundary:
                print("[ERROR] No boundary found in MJPEG stream")
                return
            
            boundary = boundary.encode('utf-8')
            buffer = b''
            
            for chunk in response.iter_content(chunk_size=1024):
                if not self.monitoring_active:
                    break
                
                buffer += chunk
                
                while boundary in buffer:
                    frame_end = buffer.find(boundary)
                    frame_data = buffer[:frame_end]
                    buffer = buffer[frame_end + len(boundary):]
                    
                    # Find JPEG data
                    jpeg_start = frame_data.find(b'\xff\xd8')
                    jpeg_end = frame_data.rfind(b'\xff\xd9')
                    
                    if jpeg_start != -1 and jpeg_end != -1:
                        jpeg_data = frame_data[jpeg_start:jpeg_end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            yield frame
                            
        except Exception as e:
            print(f"[ERROR] Error in authenticated frame streaming: {e}")
        finally:
            if 'session' in locals():
                session.close()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.monitoring_active = False

# Global instance
surveillance_service = SurveillanceService()

