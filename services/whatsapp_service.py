"""WhatsApp Notification Service using EvoAPI

Service for sending video notifications via WhatsApp when intrusion events are detected.
Creates videos from processed frame buffers and sends them to configured phone numbers.
"""

import cv2
import numpy as np
import requests
import tempfile
import os
import threading
import time
import base64
from typing import List, Dict, Any, Optional
from threading import Timer
from datetime import datetime, timedelta
from collections import deque
import pytz
from dotenv import load_dotenv

# Import configuration
from config.camera_config import CAMERAS
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service
from utils.schedule_checker import schedule_checker

# Configure timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class VideoBuffer:
    """Buffer for storing frames to create videos"""
    def __init__(self, camera_id: str, max_duration: int = 10):
        self.camera_id = camera_id
        self.max_duration = max_duration  # seconds
        self.frames = deque()  # (frame, timestamp)
        self.lock = threading.Lock()
        self.fps = 6  # Target FPS for video creation (mejorado para reproducción más fluida)
        
    def add_frame(self, frame: np.ndarray, timestamp: datetime):
        """Add a frame to the buffer"""
        with self.lock:
            self.frames.append((frame.copy(), timestamp))
            
            # Remove old frames beyond max_duration
            cutoff_time = timestamp - timedelta(seconds=self.max_duration)
            while self.frames and self.frames[0][1] < cutoff_time:
                self.frames.popleft()
    
    def get_frames_for_video(self, duration: int = None) -> List[np.ndarray]:
        """Get frames for video creation"""
        if duration is None:
            duration = self.max_duration
            
        with self.lock:
            if not self.frames:
                return []
            
            # Get frames from the last 'duration' seconds
            end_time = self.frames[-1][1]
            start_time = end_time - timedelta(seconds=duration)
            
            video_frames = []
            for frame, timestamp in self.frames:
                if timestamp >= start_time:
                    video_frames.append(frame)
            
            return video_frames

class WhatsAppService:
    """Service for WhatsApp notifications using EvoAPI"""
    
    def __init__(self, evoapi_url: str = None, api_key: str = None, instance_name: str = None):
        # Load environment variables from .env file
        load_dotenv()
        
        # EvoAPI configuration - these should be set via environment variables or config
        self.evoapi_url = evoapi_url or os.environ.get('EVOAPI_URL')
        self.api_key = api_key or os.environ.get('EVOAPI_KEY')
        self.instance_name = instance_name or os.environ.get('EVOAPI_INSTANCE')
        
        # Video buffers for each camera (increased to 30 seconds to capture before/after)
        self.video_buffers: Dict[str, VideoBuffer] = {}
        
        # Notification cooldown to prevent spam
        self.last_notification_time: Dict[str, datetime] = {}
        self.notification_cooldown = 60  # 5 minutes between notifications per camera
        
        # Alarm event timestamps for before/after video creation
        self.alarm_event_times: Dict[str, datetime] = {}
        
        # Initialize video buffers for all cameras (30 seconds to capture before/after)
        for camera_id, config in CAMERAS.items():
            # Increase buffer to 30 seconds to capture 10 seconds before and after
            self.video_buffers[camera_id] = VideoBuffer(camera_id, 30)
        
        # Start frame collection thread
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collect_frames,
            daemon=True,
            name="whatsapp_frame_collector"
        )
        self.collection_thread.start()
        
        # WhatsApp Service initialized with EvoAPI
    
    def _collect_frames(self):
        """Background thread to collect processed frames for video creation"""
        last_frame_timestamps = {}  # Track last frame timestamp per camera to avoid duplicates
        
        while self.collection_active:
            try:
                # Import surveillance service here to avoid circular imports
                from surveillance.surveillance_service import surveillance_service
                
                # Only collect frames when surveillance is active
                if surveillance_service.monitoring_active:
                    for camera_id in CAMERAS.keys():
                        # Get latest processed frame
                        processing_result = enhanced_frame_processing_service.get_latest_processing_result(camera_id)
                        
                        if processing_result:
                            # Check if this is a new frame (avoid duplicates from cache)
                            last_timestamp = last_frame_timestamps.get(camera_id)
                            if last_timestamp is None or processing_result.timestamp > last_timestamp:
                                # Add processed frame to buffer only if it's new
                                self.video_buffers[camera_id].add_frame(
                                    processing_result.processed_frame,
                                    processing_result.timestamp
                                )
                                last_frame_timestamps[camera_id] = processing_result.timestamp
                else:
                    # Clear old frames when surveillance is not active
                    for camera_id in CAMERAS.keys():
                        if camera_id in self.video_buffers:
                            with self.video_buffers[camera_id].lock:
                                self.video_buffers[camera_id].frames.clear()
                    # Reset timestamps when surveillance stops
                    last_frame_timestamps.clear()
                
                time.sleep(0.1)  # Collect frames every 100ms
                
            except Exception as e:
                print(f"[ERROR] Error in frame collection: {e}")
                time.sleep(1)
    
    def _create_video_from_buffer(self, camera_id: str, duration_seconds: int = 10) -> str:
        """Crea un video MP4 compatible con WhatsApp a partir de los frames en el buffer"""
        try:
            # Wait for frames to be available (up to 3 seconds)
            max_wait_time = 3.0
            wait_interval = 0.2
            waited_time = 0.0
            
            while waited_time < max_wait_time:
                if camera_id in self.video_buffers and self.video_buffers[camera_id].frames:
                    break
                time.sleep(wait_interval)
                waited_time += wait_interval
            
            if camera_id not in self.video_buffers or not self.video_buffers[camera_id].frames:
                print(f"[ERROR] No hay frames disponibles en el buffer de video para {camera_id} después de esperar {max_wait_time}s")
                return ""
            
            frames_buffer = self.video_buffers[camera_id]
            
            # Calcular cuántos frames necesitamos para la duración deseada
            frames_needed = min(duration_seconds * frames_buffer.fps, len(frames_buffer.frames))
            
            # Mínimo de frames para un video válido (al menos 2 segundos)
            min_frames_required = frames_buffer.fps * 2
            if frames_needed < min_frames_required:
                print(f"[ERROR] Insuficientes frames para crear video ({frames_needed} frames disponibles, mínimo {min_frames_required})")
                return ""
                
            # Creating video with specified frames and FPS
            
            # Obtener frames del buffer de forma rápida para minimizar el tiempo de lock
            with frames_buffer.lock:
                all_frames = list(frames_buffer.frames)
            
            # Procesar frames fuera del lock para no bloquear la adición de nuevos frames
            # Filtrar frames duplicados basado en timestamp
            unique_frames = []
            seen_timestamps = set()
            
            for frame_data in reversed(all_frames):  # Empezar desde los más recientes
                frame, timestamp = frame_data
                timestamp_key = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
                if timestamp_key not in seen_timestamps:
                    unique_frames.append(frame_data)
                    seen_timestamps.add(timestamp_key)
                    if len(unique_frames) >= frames_needed:
                        break
            
            # Revertir para mantener orden cronológico
            recent_frames = list(reversed(unique_frames))
            
            if not recent_frames:
                return ""
            
           
            # Obtener dimensiones del primer frame
            first_frame = recent_frames[0][0]  # frame is the first element of tuple (frame, timestamp)
            original_height, original_width = first_frame.shape[:2]
            
            # Optimizar resolución para WhatsApp (máximo 1280x720 para mejor compatibilidad)
            max_width, max_height = 1280, 720
            
            # Calcular nueva resolución manteniendo aspect ratio
            aspect_ratio = original_width / original_height
            if original_width > max_width or original_height > max_height:
                if aspect_ratio > 1:  # landscape
                    width = max_width
                    height = int(max_width / aspect_ratio)
                else:  # portrait
                    height = max_height
                    width = int(max_height * aspect_ratio)
            else:
                width, height = original_width, original_height
            
            # Asegurar que las dimensiones sean pares (requerido por H.264)
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
                
            # Video resolution adjusted
            
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            try:
                # Crear video temporal inicial con OpenCV
                temp_raw_video = temp_video_path.replace('.mp4', '_raw.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Usar XVID para video temporal
                video_writer = cv2.VideoWriter(temp_raw_video, fourcc, frames_buffer.fps, (width, height))
                
                if not video_writer.isOpened():
                    print(f"[ERROR] Error al crear el video writer para {camera_id}")
                    return ""
                
                # Escribir frames al video
                for frame_data in recent_frames:
                    frame = frame_data[0]  # frame is the first element of tuple
                    # Redimensionar frame si es necesario
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video_writer.write(frame)
                
                video_writer.release()
                
                # Recodificar con FFmpeg para máxima compatibilidad con WhatsApp
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y',  # -y para sobrescribir sin preguntar
                    '-i', temp_raw_video,  # archivo de entrada
                    '-c:v', 'libx264',  # codec de video H.264
                    '-profile:v', 'baseline',  # perfil baseline para máxima compatibilidad
                    '-level', '3.0',  # nivel H.264 compatible con dispositivos móviles
                    '-preset', 'medium',  # preset balanceado para mejor calidad
                    '-crf', '20',  # mejor calidad (menor valor = mejor calidad)
                    '-pix_fmt', 'yuv420p',  # formato de pixel compatible
                    '-r', str(frames_buffer.fps),  # framerate explícito
                    '-g', str(frames_buffer.fps * 2),  # keyframe cada 2 segundos
                    '-sc_threshold', '0',  # deshabilitar detección de cambio de escena
                    '-movflags', '+faststart',  # optimización para streaming
                    '-avoid_negative_ts', 'make_zero',  # evitar timestamps negativos
                    '-fflags', '+genpts',  # generar timestamps de presentación
                    temp_video_path  # archivo de salida
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                # Limpiar archivo temporal
                if os.path.exists(temp_raw_video):
                    os.unlink(temp_raw_video)
                
                if result.returncode != 0:
                    print(f"[ERROR] Error en FFmpeg: {result.stderr}")
                    return ""
                
                # Video created successfully with H.264
                
                # Save a copy of the video in the root folder for verification
                import shutil
                root_videos_dir = "/home/tecon/Documentos/Camara/API_Camaras_Server/videos"
                os.makedirs(root_videos_dir, exist_ok=True)
                timestamp = int(time.time())
                saved_video_path = os.path.join(root_videos_dir, f"video_{camera_id}_{timestamp}.mp4")
                shutil.copy2(temp_video_path, saved_video_path)
                # Video saved for verification
                
                return temp_video_path
                
            except Exception as e:
                print(f"[ERROR] Error creando video: {e}")
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                return ""
                
        except Exception as e:
            print(f"[ERROR] Error en creación de video: {e}")
            return ""
    
    def send_whatsapp_video(self, phone_numbers: List[str], video_path: str, 
                           message: str, camera_id: str) -> bool:
        """Send video via WhatsApp using EvoAPI with base64 encoding"""
        try:
            if not self.api_key:
                print("[ERROR] EvoAPI key not configured")
                return False
            
            if not self.instance_name:
                print("[ERROR] EvoAPI instance name not configured")
                return False
            
            # Read and encode video file to base64
            try:
                with open(video_path, 'rb') as video_file:
                    video_data = video_file.read()
                    video_size_mb = len(video_data) / (1024 * 1024)
                    
                    if video_size_mb > 16:  # WhatsApp limit is usually 16MB
                        print(f"[ERROR] Video file too large: {video_size_mb:.2f} MB (max 16MB)")
                        return False
                    
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
            except Exception as e:
                print(f"[ERROR] Error reading video file: {e}")
                return False
            
            success_count = 0
            
            for phone_number in phone_numbers:
                try:
                    # Prepare headers for JSON request
                    headers = {
                        'apikey': self.api_key,
                        'Content-Type': 'application/json'
                    }
                    
                    # Prepare JSON payload with base64 encoded video
                    # Send raw base64 without prefix as Evolution API expects
                    
                    payload = {
                        'number': phone_number,
                        'mediatype': 'video',
                        'mimetype': 'video/mp4',
                        'media': video_base64,
                        'caption': message,
                        'fileName': f'video_{camera_id}_{int(time.time())}.mp4'
                    }
                    
                    
                    # Send request to Evolution API
                    api_url = f"{self.evoapi_url}/message/sendMedia/{self.instance_name}"
                    
                    
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    
                    
                    if response.status_code == 201:
                        success_count += 1
                        # WhatsApp video sent successfully
                        try:
                            response_json = response.json()
                            
                        except:
                            # Response received as text
                            pass
                    else:
                        print(f"[ERROR] Failed to send WhatsApp video to {phone_number}: {response.status_code}")
                        print(f"[ERROR] Request failed with status code: {response.status_code}")
                        try:
                            error_json = response.json()
                            print(f"[ERROR] Error response received")
                        except:
                            pass
                
                except Exception as e:
                    print(f"[ERROR] Error sending WhatsApp video to {phone_number}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"[ERROR] Error in WhatsApp video sending: {e}")
            return False
    
    def send_intrusion_notification(self, camera_id: str, event_type: str, 
                                  person_count: int, face_count: int, identified_names: List[str] = None) -> bool:
        """Send intrusion notification with video - schedules a single video with 10s before and after event"""
        try:
            # Check if notifications are enabled for this camera
            camera_config = CAMERAS.get(camera_id, {})
            notifications_enabled = camera_config.get('whatsapp_notifications_enabled', False)
           
            if not notifications_enabled:
                # WhatsApp notifications disabled
                return False
            
            # Check if current time is within notification schedule
            schedule_allowed = schedule_checker.is_notification_time_allowed(camera_config)
           
            
            if not schedule_allowed:
                schedule_info = schedule_checker.get_schedule_info(camera_config)
                # WhatsApp notifications outside of scheduled hours
                return False
            
            # Check cooldown period
            current_time = datetime.now(BOGOTA_TZ)
            last_notification = self.last_notification_time.get(camera_id)
            
            
            if last_notification:
                time_since_last = (current_time - last_notification).total_seconds()
                if time_since_last < self.notification_cooldown:
                    remaining_time = self.notification_cooldown - time_since_last
                    # WhatsApp notification cooldown active
                    return False
            
            
            # Get phone numbers for this camera
            phone_numbers = camera_config.get('notification_phones', [])
            
            if not phone_numbers:
                # No phone numbers configured
                return False
            
            # Store alarm event data for later video creation
            event_time = time.time()
            alarm_data = {
                'camera_id': camera_id,
                'event_type': event_type,
                'person_count': person_count,
                'face_count': face_count,
                'identified_names': identified_names,
                'phone_numbers': phone_numbers,
                'event_time': event_time,
                'timestamp_str': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Schedule video creation and sending for 10 seconds later
            # This allows capturing 10 seconds before (already in buffer) and 10 seconds after the event
            timer = Timer(10.0, self._send_alarm_video, args=[alarm_data])
            timer.start()
            
            # Update last notification time immediately to prevent spam
            self.last_notification_time[camera_id] = current_time
            
            print(f"[INFO] Alarm video scheduled for {camera_id} - will be sent in 10 seconds")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error sending intrusion notification for {camera_id}: {e}")
            return False
    
    def _send_alarm_video(self, alarm_data: Dict[str, Any]):
        """Send a single alarm video that includes 10 seconds before and after the event"""
        try:
            camera_id = alarm_data['camera_id']
            event_type = alarm_data['event_type']
            person_count = alarm_data['person_count']
            face_count = alarm_data['face_count']
            identified_names = alarm_data['identified_names']
            phone_numbers = alarm_data['phone_numbers']
            event_time = alarm_data['event_time']
            timestamp_str = alarm_data['timestamp_str']
            
            # Create video with 20 seconds total (10 before + 10 after the event)
            video_path = self._create_alarm_video_with_context(camera_id, event_time)
            
            if not video_path:
                print(f"[ERROR] Failed to create alarm video for {camera_id}")
                return
            
            # Save a copy of the video in the root folder for verification
            import shutil
            root_videos_dir = "/home/tecon/Documentos/Camara/API_Camaras_Server/videos"
            os.makedirs(root_videos_dir, exist_ok=True)
            timestamp = int(time.time())
            saved_video_path = os.path.join(root_videos_dir, f"video_{camera_id}_{timestamp}.mp4")
            shutil.copy2(video_path, saved_video_path)
            print(f"[INFO] Alarm video saved: {saved_video_path}")
            
            try:
                # Create notification message
                if event_type == 'access_granted':
                    message = f"🔓 *ACCESO AUTOMÁTICO CONCEDIDO* 🔓\n\n"
                    message += f"📹 Cámara: {camera_id.upper()}\n"
                    message += f"✅ Estado: Acceso Autorizado\n"
                    
                    if identified_names and len(identified_names) > 0:
                        names_text = ", ".join(identified_names)
                        message += f"👥 Personas identificadas: {names_text}\n"
                    
                    message += f"🕐 Hora: {timestamp_str}\n"
                    message += f"📹 Video: 10s antes y 10s después del evento\n\n"
                   
                elif event_type == 'face_mismatch':
                    condition_name = "Discrepancia de Rostros"
                    details = f"Se detectaron {person_count} personas pero solo {face_count} rostros"
                    message = f"🚨 *ALERTA DE SEGURIDAD* 🚨\n\n"
                    message += f"📹 Cámara: {camera_id.upper()}\n"
                    message += f"⚠️ Tipo: {condition_name}\n"
                    message += f"📊 Detalles: {details}\n"
                    message += f"🕐 Hora: {timestamp_str}\n"
                    message += f"📹 Video: 10s antes y 10s después del evento\n\n"
                   
                elif event_type == 'unknown_person':
                    condition_name = "Persona Desconocida"
                    details = f"Se detectó una persona no autorizada"
                    message = f"🚨 *ALERTA DE SEGURIDAD* 🚨\n\n"
                    message += f"📹 Cámara: {camera_id.upper()}\n"
                    message += f"⚠️ Tipo: {condition_name}\n"
                    message += f"📊 Detalles: {details}\n"
                    message += f"🕐 Hora: {timestamp_str}\n"
                    message += f"📹 Video: 10s antes y 10s después del evento\n\n"
                   
                else:
                    condition_name = "Intrusión Detectada"
                    details = f"Personas: {person_count}, Rostros: {face_count}"
                    message = f"🚨 *ALERTA DE SEGURIDAD* 🚨\n\n"
                    message += f"📹 Cámara: {camera_id.upper()}\n"
                    message += f"⚠️ Tipo: {condition_name}\n"
                    message += f"📊 Detalles: {details}\n"
                    message += f"🕐 Hora: {timestamp_str}\n"
                    message += f"📹 Video: 10s antes y 10s después del evento\n\n"
                
                # Send WhatsApp video
                success = self.send_whatsapp_video(
                    phone_numbers=phone_numbers,
                    video_path=video_path,
                    message=message,
                    camera_id=camera_id
                )
                
                if success:
                    print(f"[INFO] Alarm video sent successfully for {camera_id}")
                else:
                    print(f"[ERROR] Failed to send alarm video for {camera_id}")
                
            finally:
                # Clean up temporary video file
                try:
                    os.unlink(video_path)
                except Exception as e:
                    print(f"[WARNING] Failed to delete temporary video file: {e}")
                    
        except Exception as e:
            print(f"[ERROR] Error in _send_alarm_video for {alarm_data.get('camera_id', 'unknown')}: {e}")
    
    def _create_alarm_video_with_context(self, camera_id: str, event_time: float) -> str:
        """Create a video with 10 seconds before and 10 seconds after the alarm event"""
        try:
            if camera_id not in self.video_buffers:
                print(f"[ERROR] No video buffer available for {camera_id}")
                return ""
            
            buffer = self.video_buffers[camera_id]
            current_time = time.time()
            
            # Calculate time range: 10 seconds before event to 10 seconds after event
            start_time = event_time - 10.0
            end_time = event_time + 10.0
            
            # Collect frames within the time range
            frames_to_include = []
            
            with buffer.lock:
                for frame_data in buffer.frames:
                    # frame_data is a tuple (frame, timestamp) where timestamp is datetime
                    frame, timestamp = frame_data
                    # Convert datetime timestamp to float for comparison
                    frame_timestamp = timestamp.timestamp()
                    if start_time <= frame_timestamp <= end_time:
                        frames_to_include.append(frame)
            
            if not frames_to_include:
                print(f"[ERROR] No frames found in time range for {camera_id}")
                return ""
            
            # Create video from collected frames
            video_path = self._create_video_from_frames(frames_to_include, camera_id, "alarm_context")
            
            if video_path:
                print(f"[INFO] Created alarm context video for {camera_id} with {len(frames_to_include)} frames")
            
            return video_path
            
        except Exception as e:
            print(f"[ERROR] Error creating alarm context video for {camera_id}: {e}")
            return ""
    
    def _create_video_from_frames(self, frames: List[np.ndarray], camera_id: str, video_type: str = "custom") -> str:
        """Create a video from a list of frames"""
        try:
            if not frames:
                print(f"[ERROR] No frames provided for video creation")
                return ""
            
            # Get dimensions from first frame
            first_frame = frames[0]
            original_height, original_width = first_frame.shape[:2]
            
            # Optimize resolution for WhatsApp (max 1280x720)
            max_width, max_height = 1280, 720
            
            # Calculate new resolution maintaining aspect ratio
            aspect_ratio = original_width / original_height
            if original_width > max_width or original_height > max_height:
                if aspect_ratio > 1:  # landscape
                    width = max_width
                    height = int(max_width / aspect_ratio)
                else:  # portrait
                    height = max_height
                    width = int(max_height * aspect_ratio)
            else:
                width, height = original_width, original_height
            
            # Ensure dimensions are even (required by H.264)
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            try:
                # Create temporary raw video with OpenCV
                temp_raw_video = temp_video_path.replace('.mp4', '_raw.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 6  # Default FPS for alarm videos
                video_writer = cv2.VideoWriter(temp_raw_video, fourcc, fps, (width, height))
                
                if not video_writer.isOpened():
                    print(f"[ERROR] Error creating video writer for {camera_id}")
                    return ""
                
                # Write frames to video
                for frame in frames:
                    # Resize frame if necessary
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video_writer.write(frame)
                
                video_writer.release()
                
                # Re-encode with FFmpeg for WhatsApp compatibility
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_raw_video,
                    '-c:v', 'libx264',
                    '-profile:v', 'baseline',
                    '-level', '3.0',
                    '-preset', 'medium',
                    '-crf', '20',
                    '-pix_fmt', 'yuv420p',
                    '-r', str(fps),
                    '-g', str(fps * 2),
                    '-sc_threshold', '0',
                    '-movflags', '+faststart',
                    '-avoid_negative_ts', 'make_zero',
                    '-fflags', '+genpts',
                    temp_video_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                # Clean up temporary raw video
                if os.path.exists(temp_raw_video):
                    os.unlink(temp_raw_video)
                
                if result.returncode != 0:
                    print(f"[ERROR] FFmpeg error: {result.stderr}")
                    return ""
                
                return temp_video_path
                
            except Exception as e:
                print(f"[ERROR] Error creating video from frames: {e}")
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                return ""
                
        except Exception as e:
            print(f"[ERROR] Error in _create_video_from_frames: {e}")
            return ""
    
    def update_config(self, evoapi_url: str = None, api_key: str = None, 
                     instance_name: str = None, notification_cooldown: int = None):
        """Update WhatsApp service configuration"""
        if evoapi_url:
            self.evoapi_url = evoapi_url
            print(f"[INFO] Updated EvoAPI URL: {evoapi_url}")
        
        if api_key:
            self.api_key = api_key
            print("[INFO] Updated EvoAPI key")
        
        if instance_name:
            self.instance_name = instance_name
            print(f"[INFO] Updated EvoAPI instance: {instance_name}")
        
        if notification_cooldown is not None:
            self.notification_cooldown = notification_cooldown
            print(f"[INFO] Updated notification cooldown: {notification_cooldown}s")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get WhatsApp service status"""
        return {
            "service_active": self.collection_active,
            "evoapi_configured": bool(self.api_key and self.instance_name),
            "evoapi_url": self.evoapi_url,
            "instance_name": self.instance_name,
            "notification_cooldown": self.notification_cooldown,
            "cameras_with_buffers": list(self.video_buffers.keys()),
            "buffer_status": {
                camera_id: {
                    "frame_count": len(buffer.frames),
                    "max_duration": buffer.max_duration
                }
                for camera_id, buffer in self.video_buffers.items()
            },
            "last_notifications": {
                camera_id: timestamp.isoformat() if timestamp else None
                for camera_id, timestamp in self.last_notification_time.items()
            }
        }
    
    def send_recognition_confirmation_video(self, camera_id: str, phone_numbers: List[str], 
                                          recognized_persons: List[str], description: str) -> bool:
        """Send a video confirmation when all persons have been recognized for 3+ seconds"""
        try:
            # Get camera configuration
            camera_config = CAMERAS.get(camera_id, {})
            video_duration = camera_config.get('video_duration', 10)
            
            # Get video buffer for the camera
            if camera_id not in self.video_buffers:
                print(f"[WARNING] No video buffer available for {camera_id}")
                return False
            
            # Create video from buffer
            video_path = self._create_video_from_buffer(camera_id, video_duration)
            if not video_path:
                print(f"[ERROR] Failed to create recognition confirmation video for {camera_id}")
                return False

            try:
                # Send WhatsApp notification with video
                success = self.send_whatsapp_video(
                    phone_numbers=phone_numbers,
                    video_path=video_path,
                    message=description,
                    camera_id=camera_id
                )
                
                if success:
                    print(f"[INFO] Recognition confirmation video sent for {camera_id}")
                
                return success
                
            finally:
                # Clean up temporary video file
                try:
                    os.unlink(video_path)
                except Exception as e:
                    # Failed to delete temporary recognition video file
                    pass
            
        except Exception as e:
            print(f"[ERROR] Error sending recognition confirmation video for {camera_id}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the WhatsApp service"""
        self.collection_active = False
        if self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        
        print("[INFO] WhatsApp Service shutdown complete")

# Global instance
whatsapp_service = WhatsAppService()