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
# Agregar importación para YOLO
from ultralytics import YOLO

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.40

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
        
        # Configuración de YOLO para detección de personas
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Modelo ligero para detección
            # Configurar explícitamente para usar CPU
            self.yolo_model.to('cpu')
            print("✅ Modelo YOLO cargado para detección de personas (CPU)")
        except Exception as e:
            print(f"❌ Error cargando modelo YOLO: {str(e)}")
            self.yolo_model = None
        
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.last_analysis_time = {}
        self.cached_results = {}  # Cache de resultados por cámara
        self.person_detection_cache = {}  # Cache para detección de personas
        
    def detect_persons(self, img):
        """Detecta personas en la imagen usando YOLO"""
        if self.yolo_model is None:
            return [], 0
        
        try:
            # Ejecutar detección YOLO
            results = self.yolo_model(img, verbose=False)
            
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
    
    def recognize_for_stream_optimized(self, img, camera_id="default") -> Tuple[np.ndarray, list, int]:
        """Versión optimizada con cache para evitar procesamiento innecesario + detección de personas"""
        current_time = time.time()
        
        # Si tenemos resultados recientes (menos de 1 segundo), usar cache
        if (camera_id in self.last_analysis_time and 
            current_time - self.last_analysis_time[camera_id] < 1.0 and
            camera_id in self.cached_results and
            camera_id in self.person_detection_cache):
            
            # Dibujar resultados cacheados de reconocimiento facial
            cached_names, cached_boxes = self.cached_results[camera_id]
            for i, (name, box) in enumerate(zip(cached_names, cached_boxes)):
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dibujar resultados cacheados de detección de personas
            cached_person_boxes, cached_person_count = self.person_detection_cache[camera_id]
            for x1, y1, x2, y2, conf in cached_person_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"Persona {conf:.2f}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Mostrar contador de personas
            cv2.putText(img, f"Personas detectadas: {cached_person_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return img, cached_names, cached_person_count
        
        # Realizar análisis completo
        # 1. Detección de personas con YOLO
        person_boxes, person_count = self.detect_persons(img)
        
        # 2. Reconocimiento facial con InsightFace
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
            
            # Dibujar resultados de reconocimiento facial
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar detecciones de personas
        for x1, y1, x2, y2, conf in person_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Persona {conf:.2f}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Mostrar contador de personas
        cv2.putText(img, f"Personas detectadas: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Actualizar cache
        self.cached_results[camera_id] = (detected_names, detected_boxes)
        self.person_detection_cache[camera_id] = (person_boxes, person_count)
        self.last_analysis_time[camera_id] = current_time
        
        return img, detected_names, person_count

    def recognize_for_stream(self, img) -> Tuple[np.ndarray, list, int]:
        """Versión simplificada para streaming con detección de personas"""
        # Detección de personas con YOLO
        person_boxes, person_count = self.detect_persons(img)
        
        # Procesamiento de caras con InsightFace
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
            
            # Dibujar resultados de reconocimiento facial
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar detecciones de personas
        for x1, y1, x2, y2, conf in person_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Persona {conf:.2f}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Mostrar contador de personas
        cv2.putText(img, f"Personas detectadas: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img, detected_names, person_count

    def get_person_count(self, img) -> int:
        """Método específico para obtener solo el conteo de personas"""
        _, person_count = self.detect_persons(img)
        return person_count

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