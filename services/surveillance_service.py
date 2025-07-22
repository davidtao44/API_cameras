import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
import json
import os
from requests.auth import HTTPDigestAuth
import threading
import time
from typing import Tuple
from datetime import datetime
import asyncio

from ultralytics.utils.torch_utils import F
from services.access_service import access_service
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config.settings import CAMERAS
import base64
# Agregar importación para YOLO
from ultralytics import YOLO

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.40

class SurveillanceService:
    def __init__(self):
        # Configuración de InsightFace
        self._setup_insightface()
        
        # Configuración de YOLO para detección de personas
        self._setup_yolo()
        
        # Cargar embeddings conocidos
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.last_detection_time = {}
        
        # Configuración de red y sesiones HTTP
        self._setup_network_session()
        
        # Configuración de monitoreo
        self._setup_monitoring_config()
        
        # Configuración de buffers de frames
        self._setup_frame_buffers()
        
        # Configuración de detección de desconocidos
        self._setup_unknown_detection()
        
        # Configuración de relé de alarma
        self._setup_alarm_relay()
        
        # Configuración de notificaciones
        self._setup_notifications()

    def _setup_yolo(self):
        """Configura YOLO para detección de personas"""
        try:
            # Forzar YOLO a usar CPU para evitar conflictos con InsightFace GPU
            self.yolo_model = YOLO('yolov8n.pt')
            # Configurar explícitamente para usar CPU
            self.yolo_model.to('cpu')
            print("✅ Modelo YOLO cargado para detección de personas en vigilancia (CPU)")
        except Exception as e:
            print(f"❌ Error cargando modelo YOLO en vigilancia: {str(e)}")
            self.yolo_model = None

    def detect_persons(self, img):
        """Detecta personas en la imagen usando YOLO"""
        if self.yolo_model is None:
            return [], 0
        
        try:
            # Ejecutar detección YOLO en CPU
            results = self.yolo_model(img, device='cpu', verbose=False)
            
            person_boxes = []
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Clase 0 en COCO dataset es 'person'
                        if int(box.cls[0]) == 0:  # person class
                            confidence = float(box.conf[0])
                            if confidence > 0.5:  # Umbral de confianza
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                person_boxes.append((x1, y1, x2, y2, confidence))
                                person_count += 1
            
            return person_boxes, person_count
            
        except Exception as e:
            print(f"❌ Error en detección de personas: {e}")
            return [], 0

    def _setup_insightface(self):
        """Configura InsightFace con GPU/CPU"""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"🔍 Proveedores ONNX disponibles: {available_providers}")
            
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                print("✅ Configurando CUDA para GPU")
            
            if 'CPUExecutionProvider' in available_providers:
                providers.append('CPUExecutionProvider')
                print("✅ CPU como respaldo disponible")
            
            if not providers:
                providers = ['CPUExecutionProvider']
                print("⚠️ Solo CPU disponible")
            
            self.app = FaceAnalysis(providers=providers)
            
            if 'CUDAExecutionProvider' in [p[0] if isinstance(p, tuple) else p for p in providers]:
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                print("🚀 InsightFace configurado con GPU para vigilancia")
            else:
                self.app.prepare(ctx_id=-1, det_size=(320, 320))
                print("💻 InsightFace configurado con CPU para vigilancia")
                
        except Exception as e:
            print(f"❌ Error configurando InsightFace: {str(e)}")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("🔄 Usando CPU como respaldo para vigilancia")

    def _setup_network_session(self):
        """Configura sesión HTTP con reintentos"""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,
            pool_maxsize=1
        )
        self.session.mount("http://", adapter)
        self.network_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="surveillance")

    def _setup_monitoring_config(self):
        """Configura parámetros de monitoreo"""
        self.last_access_control_call = {}
        self.access_control_cooldown = 3
        self.monitoring_active = False
        self.monitoring_threads = {}
        self.frame_skip_interval = 5
        self.frame_counters = {}

    def _setup_frame_buffers(self):
        """Configura buffers de frames"""
        self.frame_buffers = {}
        self.processed_frame_buffers = {}
        self.max_buffer_size = 1

    def _setup_unknown_detection(self):
        """Configura detección de desconocidos persistentes"""
        self.unknown_detection_start = {}
        self.unknown_alert_sent = {}
        self.base_unknown_threshold = 15# Threshold base para detección de desconocidos (20 segundos)
        self.unknown_alert_cooldown = 3
        
        # Configuración para detección de discrepancias (más personas que rostros)
        self.discrepancy_detection_start = {}
        self.discrepancy_alert_sent = {}
        self.discrepancy_persistence_threshold = 10  # 10 segundos para activar alarma
        self.discrepancy_alert_cooldown = 30  # 30 segundos entre alertas
        
        # Configuración para sistema de alarmas progresivas
        self.alarm_activation_count = {}  # Contador de activaciones por cámara
        self.alarm_reset_time = {}  # Tiempo para resetear contador
        self.alarm_reset_threshold = 300  # 5 minutos para resetear contador
        self.base_cooldown_period = 20  # Cooldown base del relé (20 segundos)
        self.progressive_cooldown_reduction = 0.7  # Factor de reducción progresiva
        self.progressive_threshold_reduction = 0.7  # Factor de reducción progresiva para threshold de desconocidos

    def _setup_alarm_relay(self):
        """Configura relé de alarma"""
        self.alarm_relay_config = {
            "ip": "172.16.2.43",
            "relay_id": 0,
            "activation_duration": 2,
            "cooldown_period": 5
        }
        self.alarm_relay_active = False
        self.alarm_relay_timer = None
        self.last_alarm_activation = None

    def _setup_notifications(self):
        """Configura endpoints de notificaciones"""
        self.image_notification_config = {
            "url": "https://evoapi.tecon.com.co/message/sendMedia/Agente_OTTIS",
            "api_key": "bfbe51b2cadbb7948cbcc82d5ce694a3a389ca3e5bc3c4bb7e1398301e885edf"
        }
        self.access_notification_config = {
            "url": "https://evoapi.tecon.com.co/message/sendText/Agente_OTTIS",
            "api_key": "bfbe51b2cadbb7948cbcc82d5ce694a3a389ca3e5bc3c4bb7e1398301e885edf"
        }

    # ==================== MÉTODOS DE MONITOREO ====================
    
    def start_monitoring(self, camera_id: str = "cam1"):
        """Inicia el monitoreo automático de una cámara específica"""
        if camera_id in self.monitoring_threads and self.monitoring_threads[camera_id].is_alive():
            print(f"⚠️ El monitoreo de {camera_id} ya está activo")
            return
        
        if camera_id not in CAMERAS:
            print(f"❌ Cámara {camera_id} no encontrada en configuración")
            return
        
        print(f"🎥 Iniciando monitoreo automático de {camera_id}")
        
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
        """Detiene el monitoreo de una cámara específica o todas"""
        if camera_id:
            if camera_id in self.monitoring_threads:
                print(f"🛑 Deteniendo monitoreo de {camera_id}")
        else:
            print("🛑 Deteniendo todo el monitoreo")
            self.monitoring_active = False

    def _monitor_camera(self, camera_id: str):
        """Monitorea una cámara específica para control de acceso"""
        config = CAMERAS[camera_id]
        url = config["url"]
        
        # Inicializar buffers y contadores
        self.frame_buffers[camera_id] = []
        self.processed_frame_buffers[camera_id] = []
        self.frame_counters[camera_id] = 0
        
        try:
            if config.get("auth_required", True):
                user = config["username"]
                pwd = config["password"]
                frame_generator = self._stream_frames_with_digest(url, user, pwd)
            else:
                frame_generator = self._stream_frames_without_auth(url)
            
            for frame in frame_generator:
                if not self.monitoring_active:
                    break
                    
                try:
                    self._add_frame_to_buffer(camera_id, frame)
                    self.frame_counters[camera_id] += 1
                    
                    # Procesar cada N frames
                    if self.frame_counters[camera_id] % self.frame_skip_interval == 0:
                        latest_frame = self._get_latest_frame(camera_id)
                        if latest_frame is not None:
                            detected_names, processed_frame, detection_stats = self._process_frame_for_surveillance(latest_frame, camera_id)
                            self._add_processed_frame_to_buffer(camera_id, processed_frame)
                            
                            # Verificar discrepancias de detección (más personas que rostros)
                            self._check_discrepancy_persistence(camera_id, detection_stats)
                            
                            # Verificar desconocidos persistentes
                            self._check_unknown_persistence(camera_id, detected_names)
                            
                            # Notificar control de acceso SOLO si está habilitado para esta cámara
                            if (detected_names and 
                                any(name != "Desconocido" for name in detected_names) and
                                config.get("access_control_enabled", False)):
                                
                                current_time = time.time()
                                
                                if (camera_id not in self.last_access_control_call or 
                                    current_time - self.last_access_control_call[camera_id] >= self.access_control_cooldown):
                                    
                                    self.last_access_control_call[camera_id] = current_time
                                    self._notify_access_control(camera_id, detected_names, detection_stats)

                    time.sleep(0.001)
                    
                except Exception as e:
                    print(f"❌ Error procesando frame de {camera_id}: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            print(f"❌ Error en monitoreo de {camera_id}: {e}")
        finally:
            self._cleanup_camera_resources(camera_id)

    def _cleanup_camera_resources(self, camera_id: str):
        """Limpia recursos de una cámara al finalizar monitoreo"""
        resources_to_clean = [
            self.frame_buffers,
            self.processed_frame_buffers,
            self.frame_counters,
            self.unknown_detection_start,
            self.unknown_alert_sent,
            self.alarm_activation_count,
            self.alarm_reset_time
        ]
        
        for resource_dict in resources_to_clean:
            if camera_id in resource_dict:
                del resource_dict[camera_id]
        
        print(f"🛑 Monitoreo de {camera_id} detenido")

    # ==================== MÉTODOS DE PROCESAMIENTO DE FRAMES ====================
    
    def _process_frame_for_surveillance(self, img, camera_id: str) -> tuple:
        """Procesa un frame para vigilancia y devuelve nombres detectados, frame procesado y estadísticas de detección"""
        processed_frame = img.copy()
        
        # 1. Detección de personas con YOLO
        person_boxes, person_count = self.detect_persons(img)
        
        # 2. Reconocimiento facial con InsightFace
        faces = self.app.get(img)
        detected_names = []
        face_count = len(faces)
        
        for face in faces:
            match_name = "Desconocido"
            max_sim = -1
            
            for name, embeddings in self.known_faces.items():
                for known_embedding in embeddings:
                    sim = self.calculate_similarity(face.embedding, known_embedding)
                    if sim > THRESHOLD and sim > max_sim:
                        match_name = name
                        max_sim = sim
    
            detected_names.append(match_name)
            
            # Dibujar detección facial en el frame
            self._draw_detection_on_frame(processed_frame, face, match_name, max_sim)
            
            # Logging para personas conocidas
            if match_name != "Desconocido":
                person_id = f"{match_name}_{int(face.bbox[0])}"
                self.log_access(
                    person_id=person_id,
                    name=match_name,
                    event_type="detección",
                    camera_id=camera_id,
                    confidence=max_sim
                )
        
        # Dibujar detecciones de personas
        for x1, y1, x2, y2, conf in person_boxes:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(processed_frame, f"Persona {conf:.2f}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Mostrar estadísticas de detección
        stats_text = f"Personas: {person_count} | Rostros: {face_count}"
        cv2.putText(processed_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Agregar timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp, (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detectar discrepancias (más personas que rostros detectados)
        if person_count > face_count and person_count > 0:
            discrepancy_count = person_count - face_count
            discrepancy_text = f"⚠️ ALERTA: {discrepancy_count} persona(s) sin rostro detectado"
            cv2.putText(processed_frame, discrepancy_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Agregar indicador visual de tiempo de discrepancia si está en progreso
            if camera_id in self.discrepancy_detection_start:
                elapsed_time = time.time() - self.discrepancy_detection_start[camera_id]
                remaining_time = max(0, self.discrepancy_persistence_threshold - elapsed_time)
                time_text = f"Alarma en: {remaining_time:.1f}s"
                cv2.putText(processed_frame, time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return detected_names, processed_frame, {"person_count": person_count, "face_count": face_count}

    def _draw_detection_on_frame(self, frame, face, match_name: str, confidence: float):
        """Dibuja la detección en el frame"""
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
        
        # Rectángulo alrededor del rostro
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Etiqueta con nombre y confianza
        label = f"{match_name}"
        if confidence > 0:
            label += f" ({confidence:.2f})"
        
        # Fondo para el texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10), 
                     (bbox[0] + text_width, bbox[1]), color, -1)
        
        # Texto
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ==================== MÉTODOS DE BUFFER DE FRAMES ====================
    
    def _add_frame_to_buffer(self, camera_id: str, frame):
        """Agrega un frame al buffer y mantiene solo los más recientes"""
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = []
        
        self.frame_buffers[camera_id].append(frame)
        
        if len(self.frame_buffers[camera_id]) > self.max_buffer_size:
            self.frame_buffers[camera_id].pop(0)
    
    def _add_processed_frame_to_buffer(self, camera_id: str, processed_frame):
        """Agrega un frame procesado al buffer y mantiene solo los más recientes"""
        if camera_id not in self.processed_frame_buffers:
            self.processed_frame_buffers[camera_id] = []
        
        self.processed_frame_buffers[camera_id].append(processed_frame)
        
        if len(self.processed_frame_buffers[camera_id]) > self.max_buffer_size:
            self.processed_frame_buffers[camera_id].pop(0)

    def _get_latest_frame(self, camera_id: str):
        """Obtiene el frame más reciente del buffer"""
        if camera_id in self.frame_buffers and self.frame_buffers[camera_id]:
            return self.frame_buffers[camera_id][-1]
        return None

    def _get_latest_processed_frame(self, camera_id: str):
        """Obtiene el frame procesado más reciente del buffer"""
        if camera_id in self.processed_frame_buffers and self.processed_frame_buffers[camera_id]:
            return self.processed_frame_buffers[camera_id][-1]
        return None

    def _clear_frame_buffer(self, camera_id: str):
        """Limpia el buffer de frames para una cámara específica"""
        if camera_id in self.frame_buffers:
            self.frame_buffers[camera_id].clear()
            print(f"🧹 Buffer de frames limpiado para {camera_id}")

    # ==================== MÉTODOS DE CONTROL DE ACCESO ====================
    
    def _notify_access_control(self, camera_id: str, detected_names: list, detection_stats: dict):
        """Notifica al servicio de control de acceso sobre las detecciones"""
        try:
            self.network_executor.submit(
                self._call_access_control,
                camera_id,
                detected_names,
                detection_stats
            )
        except Exception as e:
            print(f"❌ Error notificando control de acceso: {e}")
    
    def _call_access_control(self, camera_id: str, detected_names: list, detection_stats: dict):
        """Llama al servicio de control de acceso de forma síncrona"""
        try:
            import asyncio
            from services.access_control_service import access_control_service
            from datetime import datetime
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    access_control_service.evaluate_access(
                        camera_id, 
                        detected_names, 
                        datetime.now(),
                        detection_stats
                    )
                )
                
                if result.get("relay_activated"):
                    names_only = [name.split('_')[0] for name in detected_names if name != "Desconocido"]
                    print(f"🚨 Relé activado - {camera_id}: {names_only}")
                    self._send_access_message("573107834112", f"Se concedio el acceso a: {', '.join(names_only)}")
                    self._send_access_message("573132843357", f"Se concedio el acceso a: {', '.join(names_only)}")
                    
                elif result.get("access_granted"):
                    names_only = [name.split('_')[0] for name in detected_names if name != "Desconocido"]
                    print(f"✅ Acceso concedido - {camera_id}: {names_only}")
                else:
                    print(f"🚫 Acceso denegado - {camera_id}: {result.get('message', 'Sin mensaje')}")
                    
            finally:
                loop.close()
                
        except Exception as e:
            print(f"❌ Error en evaluación de acceso: {e}")

    # ==================== MÉTODOS DE DETECCIÓN DE DESCONOCIDOS ====================
    
    def _check_discrepancy_persistence(self, camera_id: str, detection_stats: dict):
        """Verifica si hay discrepancias persistentes (más personas que rostros) y activa alarma"""
        current_time = time.time()
        person_count = detection_stats["person_count"]
        face_count = detection_stats["face_count"]
        has_discrepancy = person_count > face_count and person_count > 0
        
        if has_discrepancy:
            discrepancy_count = person_count - face_count
            # Iniciar o continuar detección de discrepancia
            if camera_id not in self.discrepancy_detection_start:
                self.discrepancy_detection_start[camera_id] = current_time
                print(f"🔍 Iniciando detección de discrepancia persistente en {camera_id}: {discrepancy_count} persona(s) sin rostro")
            
            # Verificar si ha pasado el tiempo umbral
            elapsed_time = current_time - self.discrepancy_detection_start[camera_id]
            
            if (elapsed_time >= self.discrepancy_persistence_threshold and 
                (camera_id not in self.discrepancy_alert_sent or 
                 current_time - self.discrepancy_alert_sent[camera_id] >= self.discrepancy_alert_cooldown)):
                
                print(f"🚨 Discrepancia persistente detectada en {camera_id}: {discrepancy_count} persona(s) sin identificar por {elapsed_time:.1f} segundos")
                self._notify_discrepancy_with_alarm(camera_id, discrepancy_count)
                self.discrepancy_alert_sent[camera_id] = current_time
        else:
            # Limpiar detección si no hay discrepancias
            if camera_id in self.discrepancy_detection_start:
                del self.discrepancy_detection_start[camera_id]
                print(f"✅ Detección de discrepancia finalizada en {camera_id}")

    def _check_unknown_persistence(self, camera_id: str, detected_names: list):
        """Verifica si hay desconocidos que han permanecido demasiado tiempo"""
        current_time = time.time()
        has_unknown = "Desconocido" in detected_names
        
        if has_unknown:
            if camera_id not in self.unknown_detection_start:
                self.unknown_detection_start[camera_id] = current_time
                print(f"🔍 Iniciando seguimiento de desconocido en {camera_id}")
            
            time_elapsed = current_time - self.unknown_detection_start[camera_id]
            
            # Calcular threshold progresivo basado en activaciones de alarma
            activation_count = self.alarm_activation_count.get(camera_id, 0)
            progressive_threshold = max(5, self.base_unknown_threshold * (self.progressive_threshold_reduction ** activation_count))
            
            if time_elapsed >= progressive_threshold:
                can_alert = True
                if camera_id in self.unknown_alert_sent:
                    time_since_last_alert = current_time - self.unknown_alert_sent[camera_id]
                    if time_since_last_alert < self.unknown_alert_cooldown:
                        can_alert = False
                
                if can_alert:
                    self.unknown_alert_sent[camera_id] = current_time
                    print(f"⏰ Threshold progresivo para {camera_id}: {progressive_threshold:.1f}s (activación #{activation_count + 1})")
                    self._notify_unknown_persistence_with_alarm(camera_id, time_elapsed)
        else:
            if camera_id in self.unknown_detection_start:
                total_time = current_time - self.unknown_detection_start[camera_id]
                print(f"✅ Desconocido ya no detectado en {camera_id} después de {total_time:.1f}s")
                del self.unknown_detection_start[camera_id]
    
    def _notify_discrepancy_with_alarm(self, camera_id: str, discrepancy_count: int):
        """Notifica sobre discrepancias persistentes y activa relé de alarma"""
        config = CAMERAS[camera_id]
        
        # Verificar si las notificaciones HTTP están habilitadas para esta cámara
        if not config.get("http_notifications_enabled", False):
            print(f"� Notificaciones HTTP deshabilitadas para {camera_id}")
            return
        
        print(f"�🚨 ALERTA DE SEGURIDAD: {discrepancy_count} persona(s) sin identificar en cámara {camera_id}")
        
        self.log_access(
            person_id=f"discrepancy_persistent_{camera_id}",
            name=f"Discrepancia: {discrepancy_count} persona(s) sin rostro",
            event_type="alerta_discrepancia",
            camera_id=camera_id,
            confidence=1.0
        )
        
        # Reiniciar el tiempo de detección de desconocidos después de activar la alarma
        current_time = time.time()
        if camera_id in self.unknown_detection_start:
            self.unknown_detection_start[camera_id] = current_time
            print(f"� Tiempo de detección de desconocidos reiniciado para {camera_id} (alarma por discrepancia)")
        
        # Activar relé y enviar imagen
        self.network_executor.submit(self._activate_alarm_relay_sync, camera_id)
        self.network_executor.submit(self._send_discrepancy_alarm_image, camera_id, discrepancy_count)

    def _notify_unknown_persistence_with_alarm(self, camera_id: str, time_elapsed: float):
        """Notifica sobre desconocidos persistentes y activa relé de alarma"""
        config = CAMERAS[camera_id]
        
        # Verificar si las notificaciones HTTP están habilitadas para esta cámara
        if not config.get("http_notifications_enabled", False):
            print(f"📵 Notificaciones HTTP deshabilitadas para {camera_id}")
            return
        
        minutes = int(time_elapsed // 60)
        seconds = int(time_elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        print(f"🚨 ALERTA DE SEGURIDAD: Desconocido detectado por {time_str} en cámara {camera_id}")
        
        self.log_access(
            person_id=f"unknown_persistent_{camera_id}",
            name="Desconocido Persistente",
            event_type="alerta_seguridad",
            camera_id=camera_id,
            confidence=1.0
        )
        
        # Reiniciar el tiempo de detección de desconocidos después de activar la alarma
        current_time = time.time()
        self.unknown_detection_start[camera_id] = current_time
        print(f"🔄 Tiempo de detección de desconocidos reiniciado para {camera_id}")
        
        # Activar relé y enviar imagen
        self.network_executor.submit(self._activate_alarm_relay_sync, camera_id)
        self.network_executor.submit(self._send_alarm_image, camera_id, time_str)

    # ==================== MÉTODOS DE NOTIFICACIONES ====================
    
    def _send_access_message(self, phone_number: str, message: str) -> bool:
        """Envía mensaje de texto al endpoint configurado"""
        try:
            headers = {
                "Content-Type": "application/json",
                "apikey": self.access_notification_config["api_key"]
            }
            
            payload = {
                "number": phone_number,
                "text": message
            }
            
            response = self.session.post(
                self.access_notification_config["url"],
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Mensaje de acceso enviado exitosamente a {phone_number}")
                return True
            else:
                print(f"❌ Error enviando mensaje de acceso a {phone_number}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error en solicitud de mensaje de acceso a {phone_number}: {e}")
            return False
    
    def _send_discrepancy_alarm_image(self, camera_id: str, discrepancy_count: int):
        """Envía imagen en base64 cuando se dispara la alarma por discrepancia"""
        try:
            config = CAMERAS[camera_id]
            
            # Verificar si las notificaciones HTTP están habilitadas
            if not config.get("http_notifications_enabled", False):
                print(f"📵 Notificaciones HTTP deshabilitadas para {camera_id}")
                return
            
            latest_processed_frame = self._get_latest_processed_frame(camera_id)
            if latest_processed_frame is None:
                latest_processed_frame = self._get_latest_frame(camera_id)
                if latest_processed_frame is None:
                    print(f"❌ No hay frame disponible para enviar imagen de {camera_id}")
                    return
            
            image_base64 = self._frame_to_base64(latest_processed_frame)
            if not image_base64:
                print(f"❌ Error al convertir frame a base64 para {camera_id}")
                return
            
            ubication = "Puerta" if camera_id == "cam2" else "Test"
            caption = f"🚨 ALERTA DE SEGURIDAD\n" \
                     f"Cámara: {ubication}\n" \
                     f"{discrepancy_count} persona(s) detectada(s) sin rostro identificable\n" \
                     f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                     f"Posible intrusión o persona con rostro oculto\n" \
                     f"Si desea enrrolar al desconocido acceda al link: https://homeface.tecon.com.co/\n" \
                     f"Usuario: devps@tecon.com.co contraseña: admin123 "     
            
            # Obtener números de teléfono específicos de la configuración de la cámara
            notification_phones = config.get("notification_phones", ["573107834112",  "573132843357"])
            
            # Enviar a los números configurados para esta cámara
            for phone in notification_phones:
                success = self._send_image_notification(phone, image_base64, caption)
                if success:
                    print(f"📱 Imagen de alarma por discrepancia enviada exitosamente a {phone} para {camera_id}")
                else:
                    print(f"❌ Error al enviar imagen de alarma por discrepancia a {phone} para {camera_id}")
                    
        except Exception as e:
            print(f"❌ Error enviando imagen de alarma por discrepancia: {e}")

    def _send_alarm_image(self, camera_id: str, time_str: str):
        """Envía imagen en base64 cuando se dispara la alarma"""
        try:
            config = CAMERAS[camera_id]
            
            # Verificar si las notificaciones HTTP están habilitadas
            if not config.get("http_notifications_enabled", False):
                print(f"📵 Notificaciones HTTP deshabilitadas para {camera_id}")
                return
            
            latest_processed_frame = self._get_latest_processed_frame(camera_id)
            if latest_processed_frame is None:
                latest_processed_frame = self._get_latest_frame(camera_id)
                if latest_processed_frame is None:
                    print(f"❌ No hay frame disponible para enviar imagen de {camera_id}")
                    return
            
            image_base64 = self._frame_to_base64(latest_processed_frame)
            if not image_base64:
                print(f"❌ Error al convertir frame a base64 para {camera_id}")
                return
            
            ubication = "Puerta" if camera_id == "cam2" else "Test"
            caption = f"🚨 ALERTA DE SEGURIDAD\n" \
                     f"Cámara: {ubication}\n" \
                     f"Desconocido detectado por: {time_str}\n" \
                     f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                     f"Si desea inscribir a la persona debe acceder al link: https://homeface.tecon.com.co/"
            
            # Obtener números de teléfono específicos de la configuración de la cámara
            notification_phones = config.get("notification_phones", ["573107834112", "573132843357"])
            
            # Enviar a los números configurados para esta cámara
            for phone in notification_phones:
                success = self._send_image_notification(phone, image_base64, caption)
                if success:
                    print(f"📱 Imagen de alarma enviada exitosamente a {phone} para {camera_id}")
                else:
                    print(f"❌ Error al enviar imagen de alarma a {phone} para {camera_id}")
                    
        except Exception as e:
            print(f"❌ Error enviando imagen de alarma: {e}")
    
    def _send_image_notification(self, phone_number: str, image_base64: str, caption: str) -> bool:
        """Envía notificación con imagen en base64 al endpoint configurado"""
        try:
            headers = {
                "Content-Type": "application/json",
                "apikey": self.image_notification_config["api_key"]
            }
            
            payload = {
                "number": phone_number,
                "caption": caption,
                "mediatype": "image",
                "media": image_base64
            }
            
            response = self.session.post(
                self.image_notification_config["url"],
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                return True
            else:
                print(f"❌ Error al enviar imagen: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error en envío de imagen: {e}")
            return False

    def _frame_to_base64(self, frame) -> str:
        """Convierte un frame de OpenCV a string base64"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            print(f"❌ Error convirtiendo frame a base64: {e}")
            return ""

    # ==================== MÉTODOS DE RELÉ DE ALARMA ====================
    
    def _activate_alarm_relay_sync(self, camera_id: str):
        """Activa el relé de alarma de forma síncrona (para ThreadPoolExecutor)"""
        try:
            current_time = time.time()
            
            # Inicializar contadores si no existen
            if camera_id not in self.alarm_activation_count:
                self.alarm_activation_count[camera_id] = 0
                self.alarm_reset_time[camera_id] = current_time
            
            # Resetear contador si ha pasado el tiempo umbral
            if current_time - self.alarm_reset_time[camera_id] > self.alarm_reset_threshold:
                self.alarm_activation_count[camera_id] = 0
                self.alarm_reset_time[camera_id] = current_time
                print(f"🔄 Contador de alarmas reseteado para {camera_id}")
            
            # Calcular cooldown progresivo
            activation_count = self.alarm_activation_count[camera_id]
            progressive_cooldown = max(1, self.base_cooldown_period * (self.progressive_cooldown_reduction ** activation_count))
            
            # Verificar cooldown
            if self.last_alarm_activation:
                time_since_last = current_time - self.last_alarm_activation
                if time_since_last < progressive_cooldown:
                    print(f"⏳ Relé de alarma en cooldown ({progressive_cooldown - time_since_last:.1f}s restantes, activación #{activation_count + 1})")
                    return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success = loop.run_until_complete(self._activate_alarm_relay())
                if success:
                    self.last_alarm_activation = current_time
                    self.alarm_activation_count[camera_id] += 1
                    print(f"🚨 Relé de alarma activado en {self.alarm_relay_config['ip']} por {self.alarm_relay_config['activation_duration']}s")
                    print(f"📊 Activación #{self.alarm_activation_count[camera_id]} para {camera_id}, próximo cooldown: {progressive_cooldown * self.progressive_cooldown_reduction:.1f}s")
                else:
                    print(f"❌ Error al activar relé de alarma en {self.alarm_relay_config['ip']}")
            finally:
                loop.close()
                
        except Exception as e:
            print(f"❌ Error en activación de relé de alarma: {e}")
    
    async def _activate_alarm_relay(self) -> bool:
        """Activa el relé de alarma de forma asíncrona"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = "http://localhost:8000/relay/switch"
                data = {
                    "ip": self.alarm_relay_config["ip"],
                    "relay_id": self.alarm_relay_config["relay_id"],
                    "state": True,
                    "timeout": 5
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.alarm_relay_active = True
                        self._schedule_alarm_deactivation()
                        return True
                    else:
                        print(f"❌ Error HTTP {response.status} al activar relé de alarma")
                        return False
        except Exception as e:
            print(f"❌ Error de conexión al activar relé de alarma: {e}")
            return False
    
    def _schedule_alarm_deactivation(self):
        """Programa la desactivación automática del relé de alarma"""
        if self.alarm_relay_timer:
            self.alarm_relay_timer.cancel()
        
        self.alarm_relay_timer = threading.Timer(
            self.alarm_relay_config["activation_duration"],
            self._deactivate_alarm_relay_sync
        )
        self.alarm_relay_timer.start()

    def _deactivate_alarm_relay_sync(self):
        """Desactiva el relé de alarma de forma síncrona (para threading.Timer)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success = loop.run_until_complete(self._deactivate_alarm_relay())
                if success:
                    print(f"✅ Relé de alarma desactivado automáticamente")
                else:
                    print(f"❌ Error al desactivar relé de alarma")
            finally:
                loop.close()
        except Exception as e:
            print(f"❌ Error en desactivación de relé de alarma: {e}")
    
    async def _deactivate_alarm_relay(self) -> bool:
        """Desactiva el relé de alarma de forma asíncrona"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = "http://localhost:8000/relay/switch"
                data = {
                    "ip": self.alarm_relay_config["ip"],
                    "relay_id": self.alarm_relay_config["relay_id"],
                    "state": False,
                    "timeout": 5
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.alarm_relay_active = False
                        return True
                    return False
        except Exception as e:
            print(f"❌ Error al desactivar relé de alarma: {e}")
            return False

    async def manual_alarm_relay_control(self, activate: bool) -> dict:
        """Control manual del relé de alarma"""
        try:
            if activate:
                success = await self._activate_alarm_relay()
                message = "Relé de alarma activado manualmente" if success else "Error al activar relé de alarma"
            else:
                success = await self._deactivate_alarm_relay()
                if self.alarm_relay_timer:
                    self.alarm_relay_timer.cancel()
                message = "Relé de alarma desactivado manualmente" if success else "Error al desactivar relé de alarma"
            
            return {
                "success": success,
                "message": message,
                "alarm_relay_active": self.alarm_relay_active
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error en control manual: {e}",
                "alarm_relay_active": self.alarm_relay_active
            }
    
    def get_alarm_statistics(self, camera_id: str = None) -> dict:
        """Obtiene estadísticas del sistema de alarmas progresivas"""
        try:
            current_time = time.time()
            
            if camera_id:
                # Estadísticas para una cámara específica
                if camera_id not in self.alarm_activation_count:
                    return {
                    "camera_id": camera_id,
                    "activation_count": 0,
                    "time_since_reset": 0,
                    "current_cooldown": self.base_cooldown_period,
                    "next_cooldown": self.base_cooldown_period * self.progressive_cooldown_reduction,
                    "current_unknown_threshold": self.base_unknown_threshold,
                    "next_unknown_threshold": self.base_unknown_threshold * self.progressive_threshold_reduction
                }
                
                activation_count = self.alarm_activation_count[camera_id]
                time_since_reset = current_time - self.alarm_reset_time[camera_id]
                current_cooldown = max(1, self.base_cooldown_period * (self.progressive_cooldown_reduction ** activation_count))
                next_cooldown = max(1, self.base_cooldown_period * (self.progressive_cooldown_reduction ** (activation_count + 1)))
                current_unknown_threshold = max(5, self.base_unknown_threshold * (self.progressive_threshold_reduction ** activation_count))
                next_unknown_threshold = max(5, self.base_unknown_threshold * (self.progressive_threshold_reduction ** (activation_count + 1)))
                
                return {
                    "camera_id": camera_id,
                    "activation_count": activation_count,
                    "time_since_reset": round(time_since_reset, 1),
                    "current_cooldown": round(current_cooldown, 1),
                    "next_cooldown": round(next_cooldown, 1),
                    "current_unknown_threshold": round(current_unknown_threshold, 1),
                    "next_unknown_threshold": round(next_unknown_threshold, 1),
                    "reset_threshold": self.alarm_reset_threshold
                }
            else:
                # Estadísticas para todas las cámaras
                stats = {}
                for cam_id in self.alarm_activation_count.keys():
                    stats[cam_id] = self.get_alarm_statistics(cam_id)
                return stats
                
        except Exception as e:
            return {"error": f"Error obteniendo estadísticas: {e}"}

    # ==================== MÉTODOS DE STREAMING ====================
    
    def _stream_frames_with_digest(self, url, username, password):
        """Stream con autenticación digest"""
        auth = HTTPDigestAuth(username, password)
        with requests.get(url, auth=auth, stream=True) as r:
            if r.status_code != 200:
                print(f"⚠️ Error al acceder al stream: {r.status_code}")
                return
            bytes_data = b""
            for chunk in r.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame

    def _stream_frames_without_auth(self, url):
        """Stream sin autenticación"""
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                print(f"⚠️ Error al acceder al stream: {r.status_code}")
                return
            bytes_data = b""
            for chunk in r.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame

    # ==================== MÉTODOS UTILITARIOS ====================
    
    def load_embeddings(self, embeddings_file):
        """Carga los embeddings desde un archivo JSON"""
        if not os.path.exists(embeddings_file):
            print(f"Archivo de embeddings no encontrado: {embeddings_file}")
            return {}
        
        try:
            with open(embeddings_file, 'r') as f:
                file_content = f.read().strip()
                if not file_content:
                    print(f"Archivo de embeddings vacío: {embeddings_file}")
                    return {}
                data = json.loads(file_content)
                return data
        except json.JSONDecodeError:
            print(f"Error al decodificar el archivo JSON: {embeddings_file}")
            return {}
        except Exception as e:
            print(f"Error al cargar embeddings: {str(e)}")
            return {}

    def calculate_similarity(self, e1, e2):
        """Calcula la similitud coseno entre dos embeddings"""
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    def log_access(self, person_id: str, name: str, event_type: str, camera_id: str, confidence: float):
        """Registra un evento de acceso"""
        current_time = datetime.now()
        
        # Evitar registros duplicados
        if person_id in self.last_detection_time:
            time_diff = (current_time - self.last_detection_time[person_id]).total_seconds()
            if time_diff < 5:
                return

        self.last_detection_time[person_id] = current_time

        access_log = {
            "person_id": person_id,
            "name": name,
            "timestamp": current_time.isoformat(),
            "event_type": event_type,
            "camera_id": camera_id,
            "confidence": confidence
        }

    # ==================== MÉTODOS DE CONFIGURACIÓN ====================
    
    def set_buffer_size(self, size: int):
        """Configura el tamaño del buffer de frames"""
        if size < 1:
            size = 1
        elif size > 10:
            size = 10
        
        self.max_buffer_size = size
        print(f"📊 Tamaño de buffer configurado: {size} frames")

    def configure_alarm_relay(self, ip: str = None, relay_id: int = None, 
                             activation_duration: int = None, cooldown_period: int = None):
        """Configura los parámetros del relé de alarma"""
        if ip is not None:
            self.alarm_relay_config["ip"] = ip
        if relay_id is not None:
            self.alarm_relay_config["relay_id"] = relay_id
        if activation_duration is not None:
            self.alarm_relay_config["activation_duration"] = max(1, activation_duration)
        if cooldown_period is not None:
            self.alarm_relay_config["cooldown_period"] = max(5, cooldown_period)
        
        print(f"🔧 Configuración de relé de alarma actualizada: {self.alarm_relay_config}")
    
    def set_unknown_thresholds(self, persistence_seconds: int = 30, alert_cooldown_seconds: int = 60):
        """Configura los umbrales para detección de desconocidos persistentes"""
        self.unknown_persistence_threshold = max(5, persistence_seconds)
        self.unknown_alert_cooldown = max(10, alert_cooldown_seconds)
        print(f"⏱️ Umbrales configurados: {self.unknown_persistence_threshold}s persistencia, {self.unknown_alert_cooldown}s cooldown")

    def get_monitoring_stats(self) -> dict:
        """Obtiene estadísticas del monitoreo"""
        buffer_stats = {}
        for camera_id, buffer in self.frame_buffers.items():
            buffer_stats[camera_id] = len(buffer)
        
        return {
            "monitoring_active": self.monitoring_active,
            "frame_skip_interval": self.frame_skip_interval,
            "max_buffer_size": self.max_buffer_size,
            "active_cameras": list(self.frame_counters.keys()),
            "frame_counters": self.frame_counters.copy(),
            "buffer_sizes": buffer_stats,
            "access_control_cooldown": self.access_control_cooldown,
            "unknown_persistence_threshold": self.unknown_persistence_threshold,
            "unknown_alert_cooldown": self.unknown_alert_cooldown,
            "alarm_relay_config": self.alarm_relay_config.copy(),
            "alarm_relay_active": self.alarm_relay_active,
            "last_alarm_activation": self.last_alarm_activation,
            "active_unknown_detections": list(self.unknown_detection_start.keys())
        }

    def __del__(self):
        """Cleanup al destruir el objeto"""
        self.monitoring_active = False
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'network_executor'):
            self.network_executor.shutdown(wait=False)

    def configure_camera_access_control(self, camera_id: str, enabled: bool):
        """Configura si una cámara debe hacer control de acceso"""
        if camera_id in CAMERAS:
            CAMERAS[camera_id]["access_control_enabled"] = enabled
            print(f"✅ Control de acceso {'habilitado' if enabled else 'deshabilitado'} para {camera_id}")
        else:
            print(f"❌ Cámara {camera_id} no encontrada")

    def configure_camera_notifications(self, camera_id: str, enabled: bool, phones: list = None):
        """Configura las notificaciones HTTP para una cámara"""
        if camera_id in CAMERAS:
            CAMERAS[camera_id]["http_notifications_enabled"] = enabled
            if phones:
                CAMERAS[camera_id]["notification_phones"] = phones
            print(f"✅ Notificaciones HTTP {'habilitadas' if enabled else 'deshabilitadas'} para {camera_id}")
            if phones:
                print(f"📱 Números configurados: {phones}")
        else:
            print(f"❌ Cámara {camera_id} no encontrada")

    def get_camera_config(self, camera_id: str) -> dict:
        """Obtiene la configuración actual de una cámara"""
        if camera_id in CAMERAS:
            config = CAMERAS[camera_id].copy()
            # No mostrar credenciales por seguridad
            config.pop("password", None)
            return config
        return {}

    def get_all_cameras_config(self) -> dict:
        """Obtiene la configuración de todas las cámaras"""
        configs = {}
        for camera_id in CAMERAS:
            configs[camera_id] = self.get_camera_config(camera_id)
        return configs


# Instancia global del servicio de vigilancia
surveillance_service = SurveillanceService()

# Agregar estos métodos al final de la clase SurveillanceService

