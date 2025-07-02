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
from app.services.access_service import access_service
import aiohttp
from queue import Queue
import threading

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.5
MOVEMENT_THRESHOLD = 50  # Umbral para detectar movimiento

class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.previous_positions = {}  # Almacena posiciones anteriores de personas
        self.last_detection_time = {}  # Almacena el último tiempo de detección por persona
        self.entry_line_y = None  # Línea de entrada
        self.exit_line_y = None   # Línea de salida
        self.region_height = None # Altura de la región de detección
        self._session = None
        self.event_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()

    def _process_events(self):
        """Procesa eventos de la cola en un hilo separado"""
        while True:
            try:
                event = self.event_queue.get()
                if event:
                    asyncio.run(self._send_event(event))
                time.sleep(0.1)  # Pequeña pausa para no sobrecargar la CPU
            except Exception as e:
                print(f"Error procesando evento: {str(e)}")

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

    def recognize(self, img, camera_id: str) -> Tuple[np.ndarray, list]:
        """Reconoce caras en la imagen y registra entradas/salidas"""
        # Configurar región de detección si no está configurada
        if self.region_height is None:
            self.set_detection_region(img.shape[0])

        # Dibujar líneas de entrada y salida
        cv2.line(img, (0, self.entry_line_y), (img.shape[1], self.entry_line_y), (0, 255, 0), 2)
        cv2.line(img, (0, self.exit_line_y), (img.shape[1], self.exit_line_y), (0, 0, 255), 2)
        cv2.putText(img, "Entrada", (10, self.entry_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "Salida", (10, self.exit_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
            
            # Determinar movimiento basado en la posición actual
            current_y = face.bbox[1]  # Coordenada Y del rostro
            person_id = f"{match_name}_{face.bbox[0]}"  # ID único basado en nombre y posición X
            event_type = self.determine_movement(person_id, current_y)
            
            # Registrar acceso si se detectó una posición válida
            if event_type and match_name != "Desconocido":
                self.log_access(
                    person_id=person_id,
                    name=match_name,
                    event_type=event_type,
                    camera_id=camera_id,
                    confidence=max_sim
                )
            
            # Draw box
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img, detected_names

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