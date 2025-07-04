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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.45

class FaceRecognizer:
    def __init__(self):
        # Configuración de InsightFace (mantener la configuración GPU existente)
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
                print("🚀 InsightFace configurado con GPU")
            else:
                self.app.prepare(ctx_id=-1, det_size=(320, 320))
                print("🐌 InsightFace configurado con CPU (tamaño reducido)")
                
        except Exception as e:
            print(f"❌ Error configurando InsightFace: {str(e)}")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("🔄 Usando CPU como respaldo")
        
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        # Eliminar estas líneas:
        # self.region_height = None
        # self.entry_line_y = None
        # self.exit_line_y = None
        self.last_analysis_time = {}
        self.cached_results = {}  # Cache de resultados por cámara
        
    def recognize_for_stream_optimized(self, img, camera_id="default") -> Tuple[np.ndarray, list]:
        """Versión optimizada con cache para evitar procesamiento innecesario"""
        current_time = time.time()
        
        # Si tenemos resultados recientes (menos de 1 segundo), usar cache
        if (camera_id in self.last_analysis_time and 
            current_time - self.last_analysis_time[camera_id] < 1.0 and
            camera_id in self.cached_results):
            
            # Dibujar resultados cacheados
            cached_names, cached_boxes = self.cached_results[camera_id]
            for i, (name, box) in enumerate(zip(cached_names, cached_boxes)):
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return img, cached_names
        
        # Realizar análisis completo
        faces = self.app.get(img)
        detected_names = []
        detected_boxes = []
        
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
            box = face.bbox.astype(int)
            detected_boxes.append(box)
            
            # Dibujar resultados
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Actualizar cache
        self.cached_results[camera_id] = (detected_names, detected_boxes)
        self.last_analysis_time[camera_id] = current_time
        
        return img, detected_names

    def recognize_for_stream(self, img) -> Tuple[np.ndarray, list]:
        """Versión simplificada solo para streaming - sin líneas de detección"""
        # Eliminar todo el código de líneas:
        # if self.region_height is None:
        #     self.set_detection_region(img.shape[0])
        # cv2.line(img, (0, self.entry_line_y), (img.shape[1], self.entry_line_y), (0, 255, 0), 2)
        # cv2.line(img, (0, self.exit_line_y), (img.shape[1], self.exit_line_y), (0, 0, 255), 2)
        # cv2.putText(img, "Entrada", (10, self.entry_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.putText(img, "Salida", (10, self.exit_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        # Procesamiento de caras (mantener)
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
            
            # Dibujar resultados (mantener)
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img, detected_names

    # Eliminar completamente este método:
    # def set_detection_region(self, frame_height: int):
    #     """Configura las líneas de entrada y salida basadas en la altura del frame"""
    #     if self.region_height is None:
    #         self.region_height = frame_height
    #         self.entry_line_y = int(frame_height * 0.7)
    #         self.exit_line_y = int(frame_height * 0.3)

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

# Instancia global para streaming
stream_recognizer = FaceRecognizer()

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