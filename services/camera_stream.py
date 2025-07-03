import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
import json
import os
import requests
from requests.auth import HTTPDigestAuth
import threading
import time
from typing import Tuple
from datetime import datetime
import asyncio
from services.access_service import access_service
import aiohttp
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # ✅ Agregar esta importación

EMBEDDINGS_FILE = "embeddings_arcface.json"
#EMBEDDINGS_FILE = r"C:/Users/jhona/Documents/Tecon/Camaras/embeddings_arcface.json"
THRESHOLD = 0.5
MOVEMENT_THRESHOLD = 50  # Umbral para detectar movimiento

class FaceRecognizer:
    def __init__(self):
        # Verificar proveedores disponibles y configurar GPU
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"🔍 Proveedores ONNX disponibles: {available_providers}")
            
            # Configurar proveedores en orden de preferencia
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
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
            
            # Configurar contexto GPU si está disponible
            if 'CUDAExecutionProvider' in [p[0] if isinstance(p, tuple) else p for p in providers]:
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                print("🚀 InsightFace configurado con GPU")
            else:
                self.app.prepare(ctx_id=-1, det_size=(320, 320))  # Tamaño menor para CPU
                print("🐌 InsightFace configurado con CPU (tamaño reducido)")
                
        except Exception as e:
            print(f"❌ Error configurando InsightFace: {str(e)}")
            # Fallback a CPU
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("🔄 Usando CPU como respaldo")
        
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.previous_positions = {}
        self.last_detection_time = {}
        self.entry_line_y = None
        self.exit_line_y = None
        self.region_height = None
        self._session = None
        self.event_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
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
        self.network_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="relay")
        
        # Cache de estado del relé para evitar peticiones innecesarias
        self.relay_state_cache = {
            'active': False,
            'last_update': None
        }
        
        # Control de acceso automático
        self.relay_config = {
            "ip": "172.16.2.47",
            "relay_id": 0,
            "timeout": 0
        }
        self.access_duration = 1
        self.last_relay_activation = None
        self.cooldown_period = 20
        self.relay_active = False
        self.relay_timer = None
        self.relay_deactivation_time = None
        
        # Control de throttling para access-control
        self.last_access_control_call = {}
        self.access_control_cooldown = 3  # 3 segundos entre llamadas por cámara

    def _process_events(self):
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

    async def _send_event(self, event):
        """Envía el evento al servidor"""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            
            async with self._session.post("http://localhost:8000/access/log", json=event) as response:
                if response.status == 200:
                    print(f"✅ Registro de {event['event_type']} exitoso para {event['name']}")
                else:
                    print(f"❌ Error al registrar {event['event_type']} para {event['name']}: {response.status}")
        except Exception as e:
            print(f"❌ Error al enviar evento: {str(e)}")

    def _is_in_cooldown(self) -> bool:
        """Verifica si el sistema está en período de enfriamiento"""
        if self.last_relay_activation is None:
            return False
        
        time_since_last = (datetime.now() - self.last_relay_activation).total_seconds()
        return time_since_last < self.cooldown_period

    def _activate_relay_optimized(self):
        """Versión optimizada de activación del relé - no bloquea el hilo principal"""
        # Verificar cache antes de hacer petición
        if self.relay_state_cache['active'] and self.relay_state_cache['last_update']:
            time_diff = time.time() - self.relay_state_cache['last_update']
            if time_diff < 1:  # Cache válido por 1 segundo
                return True
        
        def _make_request():
            try:
                url = "http://localhost:8000/relay/switch"
                data = {
                    "ip": self.relay_config["ip"],
                    "relay_id": self.relay_config["relay_id"],
                    "state": True,
                    "timeout": self.relay_config["timeout"]
                }
                
                # Usar session reutilizable con timeout corto
                response = self.session.post(url, json=data, timeout=2)
                
                if response.status_code == 200:
                    self.relay_active = True
                    self.relay_state_cache = {
                        'active': True,
                        'last_update': time.time()
                    }
                    print(f"🚪 Relé activado - Acceso concedido por {self.access_duration} segundos")
                    # Programar desactivación
                    self._schedule_relay_deactivation_optimized()
                    return True
                else:
                    print(f"❌ Error al activar relé: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Error al activar relé: {str(e)}")
                return False
        
        # Ejecutar en hilo separado para no bloquear GPU
        future = self.network_executor.submit(_make_request)
        return future

    def _schedule_relay_deactivation_optimized(self):
        """Versión optimizada de desactivación del relé"""
        def deactivate_after_delay():
            try:
                time.sleep(self.access_duration)
                
                if self.relay_active:
                    try:
                        url = "http://localhost:8000/relay/switch"
                        data = {
                            "ip": self.relay_config["ip"],
                            "relay_id": self.relay_config["relay_id"],
                            "state": False,
                            "timeout": self.relay_config["timeout"]
                        }
                        
                        # Usar session reutilizable
                        response = self.session.post(url, json=data, timeout=2)
                        
                        if response.status_code == 200:
                            self.relay_active = False
                            self.relay_state_cache = {
                                'active': False,
                                'last_update': time.time()
                            }
                            print(f"🔒 Relé desactivado - Acceso cerrado")
                        else:
                            print(f"❌ Error al desactivar relé: {response.status_code}")
                    except Exception as e:
                        print(f"❌ Error al desactivar relé: {str(e)}")
            except Exception as e:
                print(f"❌ Error en desactivación programada: {str(e)}")
        
        # Cancelar timer anterior si existe
        if hasattr(self, 'relay_timer') and self.relay_timer and self.relay_timer.is_alive():
            self.relay_timer.cancel()
        
        # Ejecutar desactivación en el pool de hilos
        self.relay_timer = threading.Timer(0, lambda: self.network_executor.submit(deactivate_after_delay))
        self.relay_timer.start()

    # En la clase FaceRecognizer, remover todos los métodos relacionados con relé:
    # - _activate_relay_optimized
    # - _schedule_relay_deactivation_optimized  
    # - _process_access_control_optimized
    # - relay_config, relay_active, etc.
    
    # Agregar método para comunicarse con el servicio de control de acceso:
    async def _notify_access_control(self, detected_names: list[str], camera_id: str):
        """Notifica al servicio de control de acceso sobre rostros detectados"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
                url = "http://localhost:8000/access-control/evaluate"
                data = {
                    "camera_id": camera_id,
                    "detected_names": detected_names,
                    "timestamp": datetime.now().isoformat()
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result["relay_activated"]:
                            print(f"🚪 {result['message']}")
        except Exception as e:
            print(f"Error notificando control de acceso: {e}")
    
    # Modificar el método recognize para usar el nuevo sistema:
    def recognize(self, img, camera_id: str) -> Tuple[np.ndarray, list]:
        """Versión optimizada del reconocimiento - prioriza GPU"""
        # Configurar región de detección si no está configurada
        if self.region_height is None:
            self.set_detection_region(img.shape[0])
    
        # Dibujar elementos UI (operaciones rápidas)
        cv2.line(img, (0, self.entry_line_y), (img.shape[1], self.entry_line_y), (0, 255, 0), 2)
        cv2.line(img, (0, self.exit_line_y), (img.shape[1], self.exit_line_y), (0, 0, 255), 2)
        cv2.putText(img, "Entrada", (10, self.entry_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "Salida", (10, self.exit_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        # Ya no mostramos el estado del relé aquí, ya que ahora es responsabilidad del servicio de control de acceso
        # cv2.putText(img, f"Rele: {relay_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
        # PROCESAMIENTO GPU - Esta es la parte crítica que no debe bloquearse
        faces = self.app.get(img)
        detected_names = []
        
        # Procesamiento de caras (mantener en hilo principal para GPU)
        for face in faces:
            match_name = "Desconocido"
            max_sim = -1
            
            # Optimización: usar numpy vectorizado cuando sea posible
            for name, embeddings in self.known_faces.items():
                for known_embedding in embeddings:
                    sim = self.calculate_similarity(face.embedding, known_embedding)
                    if sim > THRESHOLD and sim > max_sim:
                        match_name = name
                        max_sim = sim
    
            detected_names.append(match_name)
            
            # Logging de acceso (operación rápida)
            current_y = face.bbox[1]
            person_id = f"{match_name}_{face.bbox[0]}"
            event_type = self.determine_movement(person_id, current_y)
            
            if event_type and match_name != "Desconocido":
                self.log_access(
                    person_id=person_id,
                    name=match_name,
                    event_type=event_type,
                    camera_id=camera_id,
                    confidence=max_sim
                )
            
            # Dibujar resultados
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Notificar al servicio de control de acceso con throttling (no bloqueante)
        if detected_names and any(name != "Desconocido" for name in detected_names):
            current_time = time.time()
            
            # Verificar si ha pasado suficiente tiempo desde la última llamada para esta cámara
            if (camera_id not in self.last_access_control_call or 
                current_time - self.last_access_control_call[camera_id] >= self.access_control_cooldown):
                
                self.last_access_control_call[camera_id] = current_time
                
                # Ejecutar en hilo separado para no bloquear GPU
                threading.Thread(
                    target=lambda: asyncio.run(self._notify_access_control(detected_names, camera_id)),
                    daemon=True
                ).start()
        
        return img, detected_names

    def __del__(self):
        """Cleanup al destruir el objeto"""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'network_executor'):
            self.network_executor.shutdown(wait=False)

    def _all_faces_recognized(self, detected_names: list) -> bool:
        """Verifica si todos los rostros detectados son reconocidos"""
        if not detected_names:
            return False
            
        # Todos los rostros deben ser reconocidos (no "Desconocido")
        return all(name != "Desconocido" for name in detected_names)

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
            # Definir las líneas de entrada y salida
            self.entry_line_y = int(frame_height * 0.7)  # 70% de la altura
            self.exit_line_y = int(frame_height * 0.3)   # 30% de la altura

    def log_access(self, person_id: str, name: str, event_type: str, camera_id: str, confidence: float):
        """Agrega un evento de acceso a la cola"""
        current_time = datetime.now()
        
        # Evitar registros duplicados (mínimo 5 segundos entre registros)
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

        # Agregar evento a la cola
        self.event_queue.put(access_log)

    def determine_movement(self, person_id: str, current_y: int) -> str:
        """Determina si la persona está entrando o saliendo basado en su posición relativa a las líneas"""
        if person_id not in self.previous_positions:
            self.previous_positions[person_id] = current_y
            return None

        previous_y = self.previous_positions[person_id]
        
        # Actualizar posición
        self.previous_positions[person_id] = current_y
        
        # Determinar dirección del movimiento basado en la posición actual
        if current_y > self.entry_line_y:
            return "ENTRY"
        elif current_y < self.exit_line_y:
            return "EXIT"
        
        return None

recognizer = FaceRecognizer()

# Resto del código (stream_frames_with_digest y stream_frames_without_auth) permanece igual

def stream_frames_with_digest(url, username, password):
    auth = HTTPDigestAuth(username, password)

    with requests.get(url, auth=auth, stream=True) as r:
        if r.status_code != 200:
            print(f"⚠️ Error al acceder al stream: {r.status_code}")
            return

        bytes_data = b""
        for chunk in r.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

def stream_frames_without_auth(url):
    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            print(f"⚠️ Error al acceder al stream: {r.status_code}")
            return

        bytes_data = b""
        for chunk in r.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame