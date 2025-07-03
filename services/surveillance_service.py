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
from services.access_service import access_service
import aiohttp
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config.settings import CAMERAS

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.45
MOVEMENT_THRESHOLD = 50

class SurveillanceService:
    def __init__(self):
        # Configuración de InsightFace (misma configuración que antes)
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"🔍 Proveedores ONNX disponibles: {available_providers}")
            
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
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
                print("🐌 InsightFace configurado con CPU para vigilancia")
                
        except Exception as e:
            print(f"❌ Error configurando InsightFace: {str(e)}")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("🔄 Usando CPU como respaldo para vigilancia")
        
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.last_detection_time = {}
        self._session = None
        # Eliminar: self.event_queue = Queue()
        # Eliminar: self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        # Eliminar: self.processing_thread.start()
        
        # Optimización: Pool de conexiones reutilizables
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
        
        # Pool de hilos para operaciones de red
        self.network_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="surveillance")
        
        # Control de throttling para access-control
        self.last_access_control_call = {}
        self.access_control_cooldown = 3
        
        # Control de monitoreo
        self.monitoring_active = False
        self.monitoring_threads = {}
        
        # Buffer para frames más recientes
        self.frame_buffers = {}  # Buffer por cámara
        self.max_buffer_size = 3  # Mantener solo los 3 frames más recientes
        
        # Configuración de muestreo de frames
        self.frame_skip_interval = 5
        self.frame_counters = {}

    def _notify_access_control(self, camera_id: str, detected_names: list):
        """Notifica al servicio de control de acceso sobre las detecciones"""
        try:
            # Usar el pool de hilos existente para no bloquear
            self.network_executor.submit(
                self._call_access_control,
                camera_id,
                detected_names
            )
        except Exception as e:
            print(f"❌ Error notificando control de acceso: {e}")
    
    def _call_access_control(self, camera_id: str, detected_names: list):
        """Llama al servicio de control de acceso de forma síncrona"""
        try:
            import asyncio
            from services.access_control_service import access_control_service
            from datetime import datetime
            
            # Crear loop para este hilo
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    access_control_service.evaluate_access(
                        camera_id, 
                        detected_names, 
                        datetime.now()
                    )
                )
                
                if result.get("relay_activated"):
                    print(f"🔓 Relé activado - {camera_id}: {detected_names}")
                elif result.get("access_granted"):
                    print(f"✅ Acceso concedido - {camera_id}: {detected_names}")
                else:
                    print(f"🚫 Acceso denegado - {camera_id}: {result.get('message', 'Sin mensaje')}")
                    
            finally:
                loop.close()
                
        except Exception as e:
            print(f"❌ Error en evaluación de acceso: {e}")


    def _monitor_camera(self, camera_id: str):
        """Monitorea una cámara específica para control de acceso"""
        config = CAMERAS[camera_id]
        url = config["url"]
        
        # Inicializar buffer y contador para esta cámara
        self.frame_buffers[camera_id] = []
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
                    # Agregar frame al buffer y mantener solo los más recientes
                    self._add_frame_to_buffer(camera_id, frame)
                    
                    # Incrementar contador de frames
                    self.frame_counters[camera_id] += 1
                    
                    # Solo procesar cada N frames (muestreo)
                    if self.frame_counters[camera_id] % self.frame_skip_interval == 0:
                        # Procesar el frame más reciente del buffer
                        latest_frame = self._get_latest_frame(camera_id)
                        if latest_frame is not None:
                            detected_names = self._process_frame_for_surveillance(latest_frame, camera_id)
                            
                            # Notificar al servicio de control de acceso
                            # Líneas 187-195
                            if detected_names and any(name != "Desconocido" for name in detected_names):
                                current_time = time.time()
                                
                                if (camera_id not in self.last_access_control_call or 
                                    current_time - self.last_access_control_call[camera_id] >= self.access_control_cooldown):
                                    
                                    self.last_access_control_call[camera_id] = current_time
                                    # Agregar esta línea:
                                    self._notify_access_control(camera_id, detected_names)

                    # Pausa mínima para no bloquear el stream
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ Error procesando frame de {camera_id}: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            print(f"❌ Error en monitoreo de {camera_id}: {e}")
        finally:
            # Limpiar buffer y contador al finalizar
            if camera_id in self.frame_buffers:
                del self.frame_buffers[camera_id]
            if camera_id in self.frame_counters:
                del self.frame_counters[camera_id]
            print(f"🛑 Monitoreo de {camera_id} detenido")

    def _add_frame_to_buffer(self, camera_id: str, frame):
        """Agrega un frame al buffer y mantiene solo los más recientes"""
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = []
        
        # Agregar el nuevo frame
        self.frame_buffers[camera_id].append(frame)
        
        # Mantener solo los frames más recientes
        if len(self.frame_buffers[camera_id]) > self.max_buffer_size:
            self.frame_buffers[camera_id].pop(0)  # Eliminar el más antiguo

    def _get_latest_frame(self, camera_id: str):
        """Obtiene el frame más reciente del buffer"""
        if camera_id in self.frame_buffers and self.frame_buffers[camera_id]:
            return self.frame_buffers[camera_id][-1]  # Último frame (más reciente)
        return None

    def _clear_frame_buffer(self, camera_id: str):
        """Limpia el buffer de frames para una cámara específica"""
        if camera_id in self.frame_buffers:
            self.frame_buffers[camera_id].clear()
            print(f"🧹 Buffer de frames limpiado para {camera_id}")

    def set_buffer_size(self, size: int):
        """Configura el tamaño del buffer de frames"""
        if size < 1:
            size = 1
        elif size > 10:
            size = 10
        
        self.max_buffer_size = size
        print(f"📊 Tamaño de buffer configurado: {size} frames")

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
            "access_control_cooldown": self.access_control_cooldown
        }

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

    def _process_frame_for_surveillance(self, img, camera_id: str) -> list:
        """Procesa un frame para vigilancia (sin elementos visuales)"""
        # Procesamiento de caras
        faces = self.app.get(img)
        detected_names = []
        
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
            
            # Solo logging básico sin determine_movement
            if match_name != "Desconocido":
                person_id = f"{match_name}_{int(face.bbox[0])}"
                self.log_access(
                    person_id=person_id,
                    name=match_name,
                    event_type="detección",  # Tipo fijo
                    camera_id=camera_id,
                    confidence=max_sim
                )
        
        return detected_names

    def stop_monitoring(self, camera_id: str = None):
        """Detiene el monitoreo de una cámara específica o todas"""
        if camera_id:
            if camera_id in self.monitoring_threads:
                print(f"🛑 Deteniendo monitoreo de {camera_id}")
                # El hilo se detendrá cuando monitoring_active sea False
        else:
            print("🛑 Deteniendo todo el monitoreo")
            self.monitoring_active = False

    # Métodos auxiliares (mantener los existentes)
    # Eliminar completamente estos métodos obsoletos (líneas ~230-270):
    # def _process_events(self):
    # async def _send_event(self, event):
        """Procesa eventos de la cola en un hilo separado"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while True:
                try:
                    event = self.event_queue.get(timeout=1)
                    if event:
                        loop.run_until_complete(self._send_event(event))
                except:
                    continue
        except Exception as e:
            print(f"Error procesando evento: {str(e)}")
        finally:
            loop.close()

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
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    def set_detection_region(self, frame_height: int):
        """Configura las líneas de entrada y salida basadas en la altura del frame"""
        if self.region_height is None:
            self.region_height = frame_height
            self.entry_line_y = int(frame_height * 0.7)
            self.exit_line_y = int(frame_height * 0.3)

    def log_access(self, person_id: str, name: str, event_type: str, camera_id: str, confidence: float):
        """Registra un evento de acceso directamente sin cola"""
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
        # Eliminar esta línea: self.event_queue.put(access_log)

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

    def __del__(self):
        """Cleanup al destruir el objeto"""
        self.monitoring_active = False
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'network_executor'):
            self.network_executor.shutdown(wait=False)
        # Eliminar: if hasattr(self, '_session') and self._session:
        #     asyncio.run(self._session.close())

# Instancia global del servicio de vigilancia
surveillance_service = SurveillanceService()

    