"""
Intrusion Detection Service

Detects suspicious behavior patterns such as:
- More persons than faces detected (potential intruders hiding faces)
- Unknown persons staying too long in camera range
- Unusual activity patterns
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pytz

# Import WhatsApp service
try:
    from services.whatsapp_service import whatsapp_service
    WHATSAPP_AVAILABLE = True
except ImportError:
    WHATSAPP_AVAILABLE = False
    print("[WARNING] WhatsApp service not available")

# Configure timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')


@dataclass
class IntrusionEvent:
    """Represents a detected intrusion event"""
    camera_id: str
    event_type: str  # 'face_mismatch', 'unknown_prolonged', 'suspicious_activity'
    severity: str    # 'low', 'medium', 'high'
    description: str
    timestamp: datetime
    person_count: int
    face_count: int
    unknown_duration: float = 0.0
    confidence: float = 0.0

@dataclass
class AlarmTimer:
    """Represents an active alarm countdown timer"""
    camera_id: str
    condition_type: str
    start_time: datetime
    countdown_duration: float  # seconds
    event: IntrusionEvent
    is_active: bool = True
    
    @property
    def remaining_time(self) -> float:
        """Get remaining countdown time in seconds"""
        elapsed = (datetime.now(BOGOTA_TZ) - self.start_time).total_seconds()
        return max(0, self.countdown_duration - elapsed)
    
    @property
    def is_expired(self) -> bool:
        """Check if countdown timer has expired"""
        return self.remaining_time <= 0


# UnknownPersonTracker removed - simplified logic


class IntrusionDetectionService:
    """Service for detecting intrusion patterns and suspicious behavior"""
    
    def __init__(self, intrusion_timer_duration=30.0, alarm_repeat_interval=120.0):
        self.active_timers: Dict[str, AlarmTimer] = {}  # timer_key -> AlarmTimer
        self.lock = threading.Lock()
        
        # Frame confirmation tracking for timer cancellation
        self.no_intrusion_frames: Dict[str, int] = {}  # camera_id -> consecutive frames without intrusion
        self.frames_required_for_cancel = 2  # Require 2 consecutive frames to cancel timer
        
        # Alarm tracking for repeat functionality
        self.last_alarm_time: Dict[str, datetime] = {}  # camera_id -> last alarm timestamp
        
        # Tracking for all persons recognized functionality
        self.all_recognized_frames: Dict[str, int] = {}  # camera_id -> consecutive frames with all persons recognized
        self.all_recognized_start_time: Dict[str, datetime] = {}  # camera_id -> timestamp when all recognition started
        self.frames_required_for_all_recognized = 90  # ~3 seconds at 30 FPS (3 * 30 = 90 frames)
        self.last_all_recognized_notification: Dict[str, datetime] = {}  # camera_id -> last notification timestamp
        
        # Configuration
        self.config = {
            'intrusion_timer_duration': intrusion_timer_duration,  # tiempo del timer para cualquier condición de intrusión
            'alarm_repeat_interval': alarm_repeat_interval,  # intervalo para reactivar el timer (segundos)
        }
        
        # Start timer monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_timers,
            daemon=True,
            name="alarm_timer_monitor"
        )
        self.monitor_thread.start()
        
        # Intrusion Detection Service initialized with countdown timers
    
    def analyze_detection_frame(self, camera_id: str, person_count: int,
                              face_count: int, detected_names: List[str],
                              timestamp: datetime = None, tracked_persons: List = None) -> List[IntrusionEvent]:
        """Analyze a detection frame for intrusion patterns with tracking support"""
        if timestamp is None:
            timestamp = datetime.now(BOGOTA_TZ)
        
        with self.lock:
            
            # Get all known identities from face detection only
            face_detected_names = set(detected_names) - {'Unknown'}
            
            # Count tracked persons with and without confirmed identities
            tracked_known_count = 0
            tracked_unknown_count = 0
            tracked_known_names = set()
            
            if tracked_persons:
                for person in tracked_persons:
                    if person.identity_confirmed and person.name and person.name != 'Unknown':
                        tracked_known_names.add(person.name)
                        tracked_known_count += 1
                    else:
                        # Person is tracked but unknown (orange box)
                        tracked_unknown_count += 1
                    
            # Calculate total known persons (face detected + tracked known)
            all_known_names = face_detected_names.union(tracked_known_names)
            total_known_persons = len(all_known_names)
            
           
            # Calculate unknown count: only count truly unknown persons
            # Don't count tracked persons with confirmed identity as unknown, even without visible face
            face_detected_unknown = detected_names.count('Unknown')
            
            # Si hay detección de personas, usar la lógica original
            # Si no hay detección de personas pero sí rostros, considerar rostros desconocidos
            if person_count > 0:
                unknown_count = max(0, person_count - total_known_persons)
            else:
                # Solo detección facial: contar rostros desconocidos
                unknown_count = face_detected_unknown
            
            # Check conditions for intrusion
            # Optimized logic: Only trigger intrusion if there are truly unknown persons or faces
            # Don't require both body and face detection - tracked persons with confirmed identity are valid
            # Face mismatch only matters if there are actually unknown persons without any form of identification
            face_mismatch = person_count > face_count and unknown_count > 0 and tracked_known_count == 0
            unknown_persons = unknown_count > 0
            
            # Determine if any intrusion condition is met
            # Intrusion is only detected if there are unknown persons/faces that cannot be identified through tracking
            # Permitir detección de intrusos solo con rostros desconocidos
            intrusion_detected = unknown_persons and (face_mismatch or tracked_known_count == 0 or (person_count == 0 and face_count > 0))
            
            if intrusion_detected:
                # Reset no-intrusion frame counter when intrusion is detected
                self.no_intrusion_frames[camera_id] = 0
                
                # Determine the primary condition for messaging
                if face_mismatch and unknown_persons:
                    condition_type = 'face_mismatch_and_unknown'
                    condition_name = 'Face Mismatch + Unknown Person'
                elif face_mismatch:
                    condition_type = 'face_mismatch'
                    condition_name = 'Face Mismatch'
                else:
                    condition_type = 'unknown_person'
                    condition_name = 'Unknown Person'
                
                # Start unified timer for intrusion
                self._start_countdown_timer_unified(camera_id, condition_type, condition_name, person_count, face_count, timestamp)
            else:
                # No intrusion detected, increment consecutive no-intrusion frames
                self.no_intrusion_frames[camera_id] = self.no_intrusion_frames.get(camera_id, 0) + 1
                
                # Only cancel timer after required consecutive frames without intrusion
                if self.no_intrusion_frames[camera_id] >= self.frames_required_for_cancel:
                    self._cancel_timer(camera_id, 'intrusion')
                    # Reset counter after cancellation
                    self.no_intrusion_frames[camera_id] = 0
            
            # Check for all persons recognized for 3+ seconds
            self._check_all_persons_recognized(camera_id, person_count, unknown_count, all_known_names, timestamp)
        
        return []
    
    def _start_countdown_timer_unified(self, camera_id: str, condition_type: str, condition_name: str,
                                     person_count: int, face_count: int, timestamp: datetime):
        """Start a unified countdown timer for intrusion conditions"""
        timer_key = f"{camera_id}_intrusion"
        
        # Check if timer already exists and is active
        if timer_key in self.active_timers and self.active_timers[timer_key].is_active:
            return  # Timer already running, no need to restart
        
        # Create event for the unified timer
        event = IntrusionEvent(
            camera_id=camera_id,
            event_type=condition_type,
            severity='medium',
            description=f"Intrusion detected: {condition_name}",
            timestamp=timestamp,
            person_count=person_count,
            face_count=face_count
        )
        
        # Use unified timer duration for all intrusion conditions
        countdown_duration = self.config['intrusion_timer_duration']
        
        # Create new unified timer
        timer = AlarmTimer(
            camera_id=camera_id,
            condition_type='intrusion',  # Unified condition type
            start_time=datetime.now(BOGOTA_TZ),
            countdown_duration=countdown_duration,
            event=event
        )
        
        self.active_timers[timer_key] = timer
        
        # Print alarm start message - commented out for debug clarity
        # print(f"[ALARM] {camera_id}: {condition_name} detected - Countdown started ({countdown_duration}s) | Persons: {person_count}, Faces: {face_count}")
    
    def _check_all_persons_recognized(self, camera_id: str, person_count: int, unknown_count: int, 
                                    all_known_names: set, timestamp: datetime):
        """Check if all persons have been recognized for more than 3 seconds"""
        # Only proceed if there are persons OR faces detected and no unknowns
        # Permitir funcionamiento solo con detección facial
        if (person_count > 0 or len(all_known_names) > 0) and unknown_count == 0 and len(all_known_names) > 0:
            # Increment consecutive frames with all persons recognized
            self.all_recognized_frames[camera_id] = self.all_recognized_frames.get(camera_id, 0) + 1
            
            # Set start time if this is the first frame with all recognized
            if self.all_recognized_frames[camera_id] == 1:
                self.all_recognized_start_time[camera_id] = timestamp
            
            # Check if we've reached the required duration (3 seconds)
            if self.all_recognized_frames[camera_id] >= self.frames_required_for_all_recognized:
                # Check if enough time has passed since last notification (avoid spam)
                print("Acceso concedido en la camara: ", camera_id)
                last_notification = self.last_all_recognized_notification.get(camera_id)
                if (not last_notification or 
                    (timestamp - last_notification).total_seconds() > 30):  # 30 seconds cooldown
                    
                    self._trigger_all_recognized_notification(camera_id, all_known_names, timestamp)
                    self.last_all_recognized_notification[camera_id] = timestamp
                
                # Reset counter after successful notification
                self.all_recognized_frames[camera_id] = 0
                if camera_id in self.all_recognized_start_time:
                    del self.all_recognized_start_time[camera_id]
        else:
            # Reset counter if condition is not met
            self.all_recognized_frames[camera_id] = 0
            if camera_id in self.all_recognized_start_time:
                 del self.all_recognized_start_time[camera_id]
    
    def _trigger_all_recognized_notification(self, camera_id: str, all_known_names: set, timestamp: datetime):
        """Trigger notification when all persons have been recognized for 3+ seconds"""
        # All persons recognized for 3+ seconds
        
        # Check if WhatsApp notifications are enabled for this camera
        from config.camera_config import CAMERAS
        camera_config = CAMERAS.get(camera_id, {})
        
        if not camera_config.get('whatsapp_notifications_enabled', False):
            # WhatsApp notifications disabled
            return
        
        # Get phone numbers from camera config
        phone_numbers = camera_config.get('notification_phones', [])
        if not phone_numbers:
            print(f"[WARNING] No phone numbers configured for {camera_id}")
            return
        
        # Send WhatsApp notification with video if available
        if WHATSAPP_AVAILABLE:
            try:
                # Create description of recognized persons
                persons_list = ", ".join(sorted(all_known_names))
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                description = f"✅ *RECONOCIMIENTO COMPLETO* ✅\n\n"
                description += f"📹 Cámara: {camera_id.upper()}\n"
                description += f"👥 Todas las personas han sido reconocidas por más de 3 segundos\n"
                description += f"🆔 Personas identificadas: {persons_list}\n"
                description += f"🕐 Hora: {timestamp_str}\n\n"
                description += f"📹 Video de confirmación adjunto"
                
                # Send video with recognition confirmation
                success = whatsapp_service.send_recognition_confirmation_video(
                    camera_id=camera_id,
                    phone_numbers=phone_numbers,
                    recognized_persons=list(all_known_names),
                    description=description
                )
                
                if success:
                    # Recognition confirmation video sent
                    pass
                else:
                    print(f"[WARNING] Failed to send recognition confirmation video for {camera_id}")
                    
            except Exception as e:
                 print(f"[ERROR] Error sending recognition confirmation for {camera_id}: {e}")
    
    def _cancel_timer(self, camera_id: str, condition_type: str) -> bool:
        """Cancel active countdown timer"""
        timer_key = f"{camera_id}_{condition_type}"
        if timer_key in self.active_timers:
            self.active_timers[timer_key].is_active = False
            del self.active_timers[timer_key]
            # print(f"[ALARM] {camera_id}: Intrusion condition resolved - Timer cancelled")
            return True
        return False
    
    def _monitor_timers(self):
        """Background thread to monitor countdown timers"""
        while self.monitoring_active:
            try:
                with self.lock:
                    expired_timers = []
                    
                    for timer_key, timer in self.active_timers.items():
                        if timer.is_active and timer.is_expired:
                            self._trigger_alarm(timer)
                            expired_timers.append(timer_key)
                    
                    # Remove expired timers
                    for timer_key in expired_timers:
                        del self.active_timers[timer_key]
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"[ERROR] Error in alarm timer monitor: {e}")
                time.sleep(1)
    
    def _trigger_alarm(self, timer: AlarmTimer):
        """Trigger an alarm when countdown expires"""
        event = timer.event
        current_time = datetime.now(BOGOTA_TZ)
        
        # Check if enough time has passed since last alarm for this camera
        last_alarm = self.last_alarm_time.get(event.camera_id)
        if last_alarm:
            time_since_last = (current_time - last_alarm).total_seconds()
            if time_since_last < self.config['alarm_repeat_interval']:
                # Too soon for another alarm, restart timer for next check
                remaining_wait = self.config['alarm_repeat_interval'] - time_since_last
                # print(f"[INFO] {event.camera_id}: Alarm cooldown active, next alarm in {remaining_wait:.1f}s")
                # Restart timer with remaining cooldown time
                timer.start_time = current_time
                timer.countdown_duration = remaining_wait
                return
        
        # Update last alarm time
        self.last_alarm_time[event.camera_id] = current_time
        
        # Unified alarm trigger message
        condition_name = "Face Mismatch" if event.event_type == "face_mismatch" else "Unknown Person"
        timestamp_str = event.timestamp.strftime('%H:%M:%S')
        
        # Mantener solo el mensaje de confirmación de intrusión
        print(f"[INTRUSION_CONFIRMED] {event.camera_id.upper()}: {condition_name} - Alarm triggered at {timestamp_str}")
        
        # Alarm history removed - simplified logic
        
        # Send WhatsApp notification if available
        if WHATSAPP_AVAILABLE:
            try:
                whatsapp_success = whatsapp_service.send_intrusion_notification(
                    camera_id=event.camera_id,
                    event_type=event.event_type,
                    person_count=event.person_count,
                    face_count=event.face_count
                )
                # WhatsApp notification status - commented for debug clarity
                # if whatsapp_success:
                #     print(f"[INFO] WhatsApp notification sent for {event.camera_id}")
                # else:
                #     print(f"[WARNING] WhatsApp notification failed for {event.camera_id}")
            except Exception as e:
                print(f"[ERROR] WhatsApp notification error for {event.camera_id}: {e}")
        
        # Save to Firebase if available
        try:
            alert_data = {
                'camera_id': event.camera_id,
                'event_type': event.event_type,
                'severity': event.severity,
                'description': event.description,
                'timestamp': event.timestamp.isoformat(),
                'person_count': event.person_count,
                'face_count': event.face_count,
                'confidence': event.confidence,
                'unknown_duration': event.unknown_duration,
                'trigger_time': datetime.now(BOGOTA_TZ).isoformat(),
                'countdown_duration': timer.countdown_duration
            }
            
            # print(f"[INFO] Alarm saved to Firebase: {event.event_type}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save alarm to Firebase: {e}")
    
    def get_camera_status(self, camera_id: str) -> Dict[str, Any]:
        """Get intrusion detection status for a camera"""
        with self.lock:
            # Get active timers for this camera
            active_timers = {
                key: {
                    'condition_type': timer.condition_type,
                    'remaining_time': timer.remaining_time,
                    'countdown_duration': timer.countdown_duration,
                    'severity': timer.event.severity,
                    'person_count': timer.event.person_count,
                    'face_count': timer.event.face_count
                }
                for key, timer in self.active_timers.items()
                if timer.camera_id == camera_id and timer.is_active
            }
            
            return {
                'camera_id': camera_id,
                'active_alarm_timers': active_timers,
                'total_active_timers': len(active_timers)
            }
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get overall intrusion detection status"""
        with self.lock:
            return {
                'active_alarm_timers': len(self.active_timers),
                'monitored_cameras_with_timers': list(set(timer.camera_id for timer in self.active_timers.values())),
                'monitoring_active': self.monitoring_active,
                'config': self.config.copy()
            }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters"""
        with self.lock:
            for key, value in new_config.items():
                if key in self.config:
                    self.config[key] = value
                    # Updated intrusion detection config
                else:
                    print(f"[WARNING] Unknown config key: {key}")
    
    def cancel_all_timers(self, camera_id: str = None) -> int:
        """Cancel all active timers for a camera or all cameras"""
        with self.lock:
            cancelled_count = 0
            timers_to_remove = []
            
            for timer_key, timer in self.active_timers.items():
                if camera_id is None or timer.camera_id == camera_id:
                    timer.is_active = False
                    timers_to_remove.append(timer_key)
                    cancelled_count += 1
            
            for timer_key in timers_to_remove:
                del self.active_timers[timer_key]
            
            if cancelled_count > 0:
                target = camera_id if camera_id else "all cameras"
                # Cancelled alarm timers
            
            return cancelled_count
    
    def shutdown(self):
        """Shutdown the intrusion detection service"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Cancel all active timers
        self.cancel_all_timers()
        
        # Intrusion Detection Service shutdown complete


# Global instance
intrusion_detection_service = IntrusionDetectionService(
    intrusion_timer_duration=15,      # Tiempo del timer para cualquier condición de intrusión
    alarm_repeat_interval=60         # Intervalo entre alarmas repetitivas (segundos)
)