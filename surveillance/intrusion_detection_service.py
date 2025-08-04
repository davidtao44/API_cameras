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
        
        print("[INFO] Intrusion Detection Service initialized with countdown timers")
    
    def analyze_detection_frame(self, camera_id: str, person_count: int, 
                              face_count: int, detected_names: List[str],
                              timestamp: datetime = None) -> List[IntrusionEvent]:
        """Analyze a detection frame for intrusion patterns"""
        if timestamp is None:
            timestamp = datetime.now(BOGOTA_TZ)
        
        with self.lock:
            # Check conditions for intrusion
            unknown_count = detected_names.count('Unknown')
            face_mismatch = person_count > face_count
            unknown_persons = unknown_count > 0
            
            # Determine if any intrusion condition is met
            intrusion_detected = face_mismatch or unknown_persons
            
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
        
        # Print alarm start message
        print(f"[ALARM] {camera_id}: {condition_name} detected - Countdown started ({countdown_duration}s) | Persons: {person_count}, Faces: {face_count}")
    
    def _cancel_timer(self, camera_id: str, condition_type: str) -> bool:
        """Cancel active countdown timer"""
        timer_key = f"{camera_id}_{condition_type}"
        if timer_key in self.active_timers:
            self.active_timers[timer_key].is_active = False
            del self.active_timers[timer_key]
            print(f"[ALARM] {camera_id}: Intrusion condition resolved - Timer cancelled")
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
                print(f"[INFO] {event.camera_id}: Alarm cooldown active, next alarm in {remaining_wait:.1f}s")
                # Restart timer with remaining cooldown time
                timer.start_time = current_time
                timer.countdown_duration = remaining_wait
                return
        
        # Update last alarm time
        self.last_alarm_time[event.camera_id] = current_time
        
        # Unified alarm trigger message
        condition_name = "Face Mismatch" if event.event_type == "face_mismatch" else "Unknown Person"
        timestamp_str = event.timestamp.strftime('%H:%M:%S')
        
        print(f"\n🚨 ALARM TRIGGERED 🚨")
        print(f"Camera: {event.camera_id.upper()} | Condition: {condition_name} | Time: {timestamp_str}")
        print(f"Details: {event.person_count} persons, {event.face_count} faces | Severity: {event.severity.upper()}")
        if event.event_type == 'face_mismatch':
            print(f"Issue: {event.person_count - event.face_count} more persons than faces detected")
        elif event.event_type == 'unknown_person':
            print(f"Issue: Unknown persons present for {event.unknown_duration:.1f} seconds")
        print(f"Next alarm possible in: {self.config['alarm_repeat_interval']} seconds")
        print("")
        
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
                if whatsapp_success:
                    print(f"[INFO] WhatsApp notification sent for {event.camera_id}")
                else:
                    print(f"[WARNING] WhatsApp notification failed for {event.camera_id}")
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
            
            print(f"[INFO] Alarm saved to Firebase: {event.event_type}")
            
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
                    print(f"[INFO] Updated intrusion detection config: {key} = {value}")
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
                print(f"[INFO] Cancelled {cancelled_count} alarm timers for {target}")
            
            return cancelled_count
    
    def shutdown(self):
        """Shutdown the intrusion detection service"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Cancel all active timers
        self.cancel_all_timers()
        
        print("[INFO] Intrusion Detection Service shutdown complete")


# Global instance
intrusion_detection_service = IntrusionDetectionService(
    intrusion_timer_duration=20,      # Tiempo del timer para cualquier condición de intrusión
    alarm_repeat_interval=60         # Intervalo entre alarmas repetitivas (segundos)
)