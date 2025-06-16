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

    def determine_movement(self, person_id: str, current_y: int) -> str:
        """Determina si la persona está entrando o saliendo basado en su posición"""
        if person_id not in self.previous_positions:
            self.previous_positions[person_id] = current_y
            return None

        previous_y = self.previous_positions[person_id]
        movement = current_y - previous_y
        
        # Actualizar posición
        self.previous_positions[person_id] = current_y
        
        # Determinar dirección del movimiento
        if abs(movement) < MOVEMENT_THRESHOLD:
            return None
        return "EXIT" if movement > 0 else "ENTRY"

    async def log_access(self, person_id: str, name: str, event_type: str, camera_id: str, confidence: float):
        """Registra el acceso de una persona"""
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

        try:
            response = requests.post(
                "http://localhost:8000/access/log",
                json=access_log
            )
            if response.status_code == 200:
                print(f"✅ Registro de {event_type} exitoso para {name}")
            else:
                print(f"❌ Error al registrar {event_type} para {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error al registrar acceso: {str(e)}")

    def recognize(self, img, camera_id: str) -> Tuple[np.ndarray, list]:
        """Reconoce caras en la imagen y registra entradas/salidas"""
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
            
            # Determinar movimiento
            current_y = face.bbox[1]  # Coordenada Y del rostro
            person_id = f"{match_name}_{face.bbox[0]}"  # ID único basado en nombre y posición X
            event_type = self.determine_movement(person_id, current_y)
            
            # Registrar acceso si se detectó movimiento
            if event_type and match_name != "Desconocido":
                asyncio.create_task(self.log_access(
                    person_id=person_id,
                    name=match_name,
                    event_type=event_type,
                    camera_id=camera_id,
                    confidence=max_sim
                ))
            
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