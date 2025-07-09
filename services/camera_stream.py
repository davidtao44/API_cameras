import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
import json
import os
from requests.auth import HTTPDigestAuth
import threading
import time
from typing import Tuple, List, Dict
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Intentar importar YOLO para detección de personas
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO disponible para detección de personas")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO no disponible. Solo se usará detección facial")

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
        self.person_detector = None
        if YOLO_AVAILABLE:
            try:
                # Usar modelo YOLOv8n (nano) para mejor rendimiento
                self.person_detector = YOLO('yolov8n.pt')
                print("🚀 YOLO configurado para detección de personas")
            except Exception as e:
                print(f"❌ Error configurando YOLO: {str(e)}")
                self.person_detector = None
        
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.last_analysis_time = {}
        self.cached_results = {}  # Cache de resultados por cámara
        self.person_count_cache = {}  # Cache para conteo de personas
        
    def detect_persons(self, img) -> List[Dict]:
        """Detecta personas en la imagen usando YOLO"""
        if not self.person_detector:
            return []
        
        try:
            # Ejecutar detección YOLO
            results = self.person_detector(img, verbose=False)
            persons = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Clase 0 es 'person' en COCO dataset
                        if int(box.cls[0]) == 0:  # person class
                            confidence = float(box.conf[0])
                            if confidence > 0.5:  # Umbral de confianza
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                persons.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence
                                })
            
            return persons
        except Exception as e:
            print(f"❌ Error en detección de personas: {e}")
            return []
    
    def recognize_for_stream_with_person_detection(self, img, camera_id="default") -> Tuple[np.ndarray, list, int]:
        """Versión mejorada que incluye detección de personas y conteo"""
        current_time = time.time()
        
        # Detectar personas
        persons = self.detect_persons(img)
        person_count = len(persons)
        
        # Dibujar detecciones de personas
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            confidence = person['confidence']
            
            # Rectángulo azul para personas
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Persona ({confidence:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Cache para reconocimiento facial (optimización)
        if (camera_id in self.last_analysis_time and 
            current_time - self.last_analysis_time[camera_id] < 1.0 and
            camera_id in self.cached_results):
            
            # Usar resultados cacheados para rostros
            cached_names, cached_boxes = self.cached_results[camera_id]
            for i, (name, box) in enumerate(zip(cached_names, cached_boxes)):
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Actualizar cache de conteo
            self.person_count_cache[camera_id] = person_count
            
            return img, cached_names, person_count
        
        # Realizar análisis facial completo
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
            
            # Dibujar resultados faciales
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{match_name} ({max_sim:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Mostrar conteo de personas en la esquina superior izquierda
        cv2.putText(img, f"Personas: {person_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Rostros: {len(detected_names)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Actualizar caches
        self.cached_results[camera_id] = (detected_names, detected_boxes)
        self.person_count_cache[camera_id] = person_count
        self.last_analysis_time[camera_id] = current_time
        
        return img, detected_names, person_count
    
    def recognize_for_stream_optimized(self, img, camera_id="default") -> Tuple[np.ndarray, list]:
        """Versión optimizada con cache para evitar procesamiento innecesario"""
        # Llamar a la nueva función y devolver solo los primeros dos valores para compatibilidad
        processed_img, detected_names, _ = self.recognize_for_stream_with_person_detection(img, camera_id)
        return processed_img, detected_names

    def recognize_for_stream(self, img) -> Tuple[np.ndarray, list]:
        """Versión simplificada solo para streaming - ahora incluye detección de personas"""
        # Usar la nueva función con detección de personas
        processed_img, detected_names, person_count = self.recognize_for_stream_with_person_detection(img, "default")
        return processed_img, detected_names
    
    def get_person_count(self, camera_id="default") -> int:
        """Obtiene el último conteo de personas para una cámara"""
        return self.person_count_cache.get(camera_id, 0)
    
    def get_detection_stats(self, camera_id="default") -> Dict:
        """Obtiene estadísticas de detección para una cámara"""
        return {
            'person_count': self.person_count_cache.get(camera_id, 0),
            'face_count': len(self.cached_results.get(camera_id, [[], []])[0]),
            'last_update': self.last_analysis_time.get(camera_id, 0)
        }

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