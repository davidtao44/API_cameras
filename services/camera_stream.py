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

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.5

class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)

    def load_embeddings(self, embeddings_file):
        """Carga los embeddings desde un archivo JSON"""
        if not os.path.exists(embeddings_file):
            print(f"Archivo de embeddings no encontrado: {embeddings_file}")
            return {}
        
        try:
            with open(embeddings_file, 'r') as f:
                file_content = f.read().strip()
                if not file_content:  # Verificar si el archivo está vacío
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

    def recognize(self, img) -> Tuple[np.ndarray, list]:
        """Reconoce caras en la imagen y devuelve la imagen modificada y los nombres detectados"""
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