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
from datetime import datetime, timedelta
from collections import deque
import pytz
from dotenv import load_dotenv

# Import configuration
from config.camera_config import CAMERAS
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service

# Configure timezone
BOGOTA_TZ = pytz.timezone('America/Bogota')

class VideoBuffer:
    """Buffer for storing frames to create videos"""
    def __init__(self, camera_id: str, max_duration: int = 10):
        self.camera_id = camera_id
        self.max_duration = max_duration  # seconds
        self.frames = deque()  # (frame, timestamp)
        self.lock = threading.Lock()
        self.fps = 10  # Target FPS for video creation
        
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
        
        # Video buffers for each camera
        self.video_buffers: Dict[str, VideoBuffer] = {}
        
        # Notification cooldown to prevent spam
        self.last_notification_time: Dict[str, datetime] = {}
        self.notification_cooldown = 300  # 5 minutes between notifications per camera
        
        # Initialize video buffers for all cameras
        for camera_id, config in CAMERAS.items():
            video_duration = config.get('video_duration_seconds', 10)
            self.video_buffers[camera_id] = VideoBuffer(camera_id, video_duration)
        
        # Start frame collection thread
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collect_frames,
            daemon=True,
            name="whatsapp_frame_collector"
        )
        self.collection_thread.start()
        
        print("[INFO] WhatsApp Service initialized with EvoAPI")
    
    def _collect_frames(self):
        """Background thread to collect processed frames for video creation"""
        last_frame_timestamps = {}  # Track last frame timestamp per camera to avoid duplicates
        
        while self.collection_active:
            try:
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
                
                time.sleep(0.1)  # Collect frames every 100ms
                
            except Exception as e:
                print(f"[ERROR] Error in frame collection: {e}")
                time.sleep(1)
    
    def _create_video_from_buffer(self, camera_id: str, duration_seconds: int = 10) -> str:
        """Crea un video MP4 compatible con WhatsApp a partir de los frames en el buffer"""
        try:
            if camera_id not in self.video_buffers or not self.video_buffers[camera_id].frames:
                print(f"[ERROR] No hay frames disponibles en el buffer de video para {camera_id}")
                return ""
            
            frames_buffer = self.video_buffers[camera_id]
            
            # Calcular cuántos frames necesitamos para la duración deseada
            frames_needed = min(duration_seconds * frames_buffer.fps, len(frames_buffer.frames))
            
            if frames_needed < 10:  # Mínimo 10 frames para un video válido para WhatsApp
                print(f"[ERROR] Insuficientes frames para crear video ({frames_needed} frames disponibles)")
                return ""
            
            # Tomar los frames más recientes y filtrar duplicados por timestamp
            all_frames = list(frames_buffer.frames)
            
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
            
            print(f"[DEBUG] Using {len(recent_frames)} unique frames out of {len(all_frames)} total frames for {camera_id}")
            
            # Obtener dimensiones del primer frame
            first_frame = recent_frames[0][0]  # frame is the first element of tuple (frame, timestamp)
            height, width = first_frame.shape[:2]
            
            # Asegurar que las dimensiones sean pares (requerido por H.264)
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
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
                
                # Recodificar con FFmpeg para compatibilidad con WhatsApp (H.264)
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y',  # -y para sobrescribir sin preguntar
                    '-i', temp_raw_video,  # archivo de entrada
                    '-c:v', 'libx264',  # codec de video H.264
                    '-preset', 'fast',  # preset de codificación rápida
                    '-crf', '23',  # calidad (18-28, menor = mejor calidad)
                    '-pix_fmt', 'yuv420p',  # formato de pixel compatible
                    '-movflags', '+faststart',  # optimización para streaming
                    temp_video_path  # archivo de salida
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                # Limpiar archivo temporal
                if os.path.exists(temp_raw_video):
                    os.unlink(temp_raw_video)
                
                if result.returncode != 0:
                    print(f"[ERROR] Error en FFmpeg: {result.stderr}")
                    return ""
                
                print(f"[INFO] Video creado exitosamente con H.264: {temp_video_path}")
                
                # Save a copy of the video in the root folder for verification
                import shutil
                root_videos_dir = "/home/tecon/Documentos/Camara/API_Camaras_Server/videos"
                os.makedirs(root_videos_dir, exist_ok=True)
                timestamp = int(time.time())
                saved_video_path = os.path.join(root_videos_dir, f"video_{camera_id}_{timestamp}.mp4")
                shutil.copy2(temp_video_path, saved_video_path)
                print(f"[INFO] Video guardado para verificación: {saved_video_path}")
                
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
                    print(f"[DEBUG] Video file size: {video_size_mb:.2f} MB")
                    
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
                    print(f"[DEBUG] API URL: {api_url}")
                    print(f"[DEBUG] Headers: {headers}")
                    
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    print(f"[DEBUG] Response status: {response.status_code}")
                    print(f"[DEBUG] Response headers: {dict(response.headers)}")
                    
                    if response.status_code == 201:
                        success_count += 1
                        print(f"[INFO] WhatsApp video sent successfully to {phone_number}")
                        try:
                            response_json = response.json()
                            print(f"[DEBUG] Response received successfully")
                        except:
                            print(f"[DEBUG] Response received as text")
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
                                  person_count: int, face_count: int) -> bool:
        """Send intrusion notification with video"""
        try:
            # Check if notifications are enabled for this camera
            camera_config = CAMERAS.get(camera_id, {})
            if not camera_config.get('whatsapp_notifications_enabled', False):
                print(f"[INFO] WhatsApp notifications disabled for {camera_id}")
                return False
            
            # Check cooldown period
            current_time = datetime.now(BOGOTA_TZ)
            last_notification = self.last_notification_time.get(camera_id)
            
            if last_notification:
                time_since_last = (current_time - last_notification).total_seconds()
                if time_since_last < self.notification_cooldown:
                    remaining_time = self.notification_cooldown - time_since_last
                    print(f"[INFO] WhatsApp notification cooldown active for {camera_id}, {remaining_time:.0f}s remaining")
                    return False
            
            # Get phone numbers for this camera
            phone_numbers = camera_config.get('notification_phones', [])
            if not phone_numbers:
                print(f"[WARNING] No phone numbers configured for {camera_id}")
                return False
            
            # Create video from buffer
            video_duration = camera_config.get('video_duration_seconds', 10)
            video_path = self._create_video_from_buffer(camera_id, video_duration)
            
            if not video_path:
                print(f"[ERROR] Failed to create video for {camera_id}")
                return False
            
            # Save a copy of the video in the root folder for verification
            import shutil
            import os
            root_videos_dir = "/home/tecon/Documentos/Camara/API_Camaras_Server/videos"
            os.makedirs(root_videos_dir, exist_ok=True)
            timestamp = int(time.time())
            saved_video_path = os.path.join(root_videos_dir, f"video_{camera_id}_{timestamp}.mp4")
            shutil.copy2(video_path, saved_video_path)
            print(f"[INFO] Video guardado para verificación: {saved_video_path}")
            
            try:
                # Create notification message
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                
                if event_type == 'face_mismatch':
                    condition_name = "Discrepancia de Rostros"
                    details = f"Se detectaron {person_count} personas pero solo {face_count} rostros"
                elif event_type == 'unknown_person':
                    condition_name = "Persona Desconocida"
                    details = f"Se detectó una persona no autorizada"
                else:
                    condition_name = "Intrusión Detectada"
                    details = f"Personas: {person_count}, Rostros: {face_count}"
                
                message = f"🚨 *ALERTA DE SEGURIDAD* 🚨\n\n"
                message += f"📹 Cámara: {camera_id.upper()}\n"
                message += f"⚠️ Tipo: {condition_name}\n"
                message += f"📊 Detalles: {details}\n"
                message += f"🕐 Hora: {timestamp_str}\n\n"
                message += f"📹 Video adjunto de los últimos {video_duration} segundos"
                
                # Send WhatsApp notification
                success = self.send_whatsapp_video(
                    phone_numbers=phone_numbers,
                    video_path=video_path,
                    message=message,
                    camera_id=camera_id
                )
                
                if success:
                    # Update last notification time
                    self.last_notification_time[camera_id] = current_time
                    print(f"[INFO] WhatsApp intrusion notification sent for {camera_id}")
                
                return success
                
            finally:
                # Clean up temporary video file
                try:
                    os.unlink(video_path)
                except Exception as e:
                    print(f"[WARNING] Failed to delete temporary video file: {e}")
            
        except Exception as e:
            print(f"[ERROR] Error sending intrusion notification for {camera_id}: {e}")
            return False
    
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
    
    def shutdown(self):
        """Shutdown the WhatsApp service"""
        self.collection_active = False
        if self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        
        print("[INFO] WhatsApp Service shutdown complete")

# Global instance
whatsapp_service = WhatsAppService()