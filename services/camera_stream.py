import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
import json
import os
from requests.auth import HTTPDigestAuth
import threading
import time
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math
# Agregar importación para YOLO
from ultralytics import YOLO
from config.camera_config import CAMERAS

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.5

class FaceRecognizer:
    def __init__(self):
        # Configuración de InsightFace (mantener la configuración GPU existente)
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"[INFO] Proveedores ONNX disponibles: {available_providers}")
            
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                print("[INFO] Configurando CUDA para GPU")
            
            if 'CPUExecutionProvider' in available_providers:
                providers.append('CPUExecutionProvider')
                print("[INFO] CPU como respaldo disponible")
            
            if not providers:
                providers = ['CPUExecutionProvider']
                print("[WARNING] Solo CPU disponible")
            
            self.app = FaceAnalysis(providers=providers)
            
            if 'CUDAExecutionProvider' in [p[0] if isinstance(p, tuple) else p for p in providers]:
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                print("[INFO] InsightFace configurado con GPU")
            else:
                self.app.prepare(ctx_id=-1, det_size=(320, 320))
                print("🐌 InsightFace configurado con CPU (tamaño reducido)")
                
        except Exception as e:
            print(f"[ERROR] Error configurando InsightFace: {str(e)}")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("[INFO] Usando CPU como respaldo")
        
        # Configuración de YOLO para detección de personas
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Modelo ligero para detección
            # Configurar explícitamente para usar CPU
            self.yolo_model.to('cpu')
            print("[INFO] Modelo YOLO cargado para detección de personas (CPU)")
        except Exception as e:
            print(f"[ERROR] Error cargando modelo YOLO: {str(e)}")
            self.yolo_model = None
        
        self.known_faces = self.load_embeddings(EMBEDDINGS_FILE)
        self.last_analysis_time = {}
        self.cached_results = {}  # Cache de resultados por cámara
        self.person_detection_cache = {}  # Cache para detección de personas
        
        # Cache para configuración ROI
        self.roi_config_cache = {}
        self.roi_cache_time = {}
        
        # Cache para ROI adaptativa
        self.adaptive_roi_history = {}
        
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
            print(f"[ERROR] Error en detección de personas: {e}")
            return [], 0
    
    def detect_persons_with_roi(self, img, camera_id="default"):
        """Detecta personas usando YOLO con soporte para ROI"""
        # Aplicar ROI si está habilitada
        roi_img, roi_info = self._apply_roi(img, camera_id)
        
        # Detectar personas en la imagen ROI
        person_boxes, person_count = self.detect_persons(roi_img)
        
        # Ajustar coordenadas para la imagen original
        if roi_info:
            person_boxes = self._adjust_detections_for_roi(person_boxes, roi_info)
        
        return person_boxes, person_count
    
    def recognize_for_stream_optimized(self, img, camera_id="default") -> Tuple[np.ndarray, list, int]:
        """Versión optimizada con cache para evitar procesamiento innecesario + detección de personas + ROI"""
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
            
            # Dibujar ROI si está habilitada
            img = self._draw_roi_on_frame(img, camera_id)
            
            return img, cached_names, cached_person_count
        
        # Realizar análisis completo con ROI
        # Aplicar ROI si está habilitada
        roi_img, roi_info = self._apply_roi(img, camera_id)
        
        # 1. Detección de personas con YOLO en la imagen ROI
        person_boxes, person_count = self.detect_persons(roi_img)
        
        # Ajustar coordenadas de personas para la imagen original
        if roi_info:
            person_boxes = self._adjust_detections_for_roi(person_boxes, roi_info)
        
        # 2. Reconocimiento facial con InsightFace en la imagen ROI
        faces = self.app.get(roi_img)
        
        # Ajustar coordenadas de caras para la imagen original
        if roi_info:
            faces = self._adjust_faces_for_roi(faces, roi_info)
        
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
        
        # Dibujar ROI en la imagen final
        img = self._draw_roi_on_frame(img, camera_id)
        
        # Actualizar historial para ROI adaptativa
        self._update_adaptive_roi_history(camera_id, person_boxes)
        
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
    
    # ==================== MÉTODOS ROI FLEXIBLES ====================
    
    def _get_roi_config(self, camera_id: str) -> Dict:
        """Obtiene configuración ROI con cache"""
        current_time = time.time()
        
        # Usar cache si es reciente (menos de 5 segundos)
        if (camera_id in self.roi_cache_time and 
            current_time - self.roi_cache_time[camera_id] < 5.0 and
            camera_id in self.roi_config_cache):
            return self.roi_config_cache[camera_id]
        
        # Obtener configuración desde settings
        if camera_id in CAMERAS:
            roi_config = CAMERAS[camera_id].get("roi", {})
            # Configuración por defecto si no existe
            default_config = {
                "enabled": False,
                "x": 0.0,
                "y": 0.0,
                "width": 1.0,
                "height": 1.0,
                "use_percentage": True,
                "rotation": 0,
                "shape": "rectangle",
                "polygon_points": [],
                "adaptive": False,
                "margin": 0.1
            }
            # Combinar con valores por defecto
            for key, value in default_config.items():
                if key not in roi_config:
                    roi_config[key] = value
            
            # Actualizar cache
            self.roi_config_cache[camera_id] = roi_config
            self.roi_cache_time[camera_id] = current_time
            
            return roi_config
        
        return {"enabled": False}
    
    def _apply_roi(self, frame: np.ndarray, camera_id: str) -> Tuple[np.ndarray, Dict]:
        """Aplica ROI al frame según la configuración"""
        roi_config = self._get_roi_config(camera_id)
        
        if not roi_config.get("enabled", False):
            return frame, {"x": 0, "y": 0, "width": frame.shape[1], "height": frame.shape[0]}
        
        h, w = frame.shape[:2]
        
        # Calcular coordenadas
        if roi_config.get("use_percentage", True):
            x = int(roi_config["x"] * w)
            y = int(roi_config["y"] * h)
            width = int(roi_config["width"] * w)
            height = int(roi_config["height"] * h)
        else:
            x = int(roi_config["x"])
            y = int(roi_config["y"])
            width = int(roi_config["width"])
            height = int(roi_config["height"])
        
        # Validar límites
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        width = max(1, min(width, w-x))
        height = max(1, min(height, h-y))
        
        # Aplicar ROI adaptativa si está habilitada
        if roi_config.get("adaptive", False):
            x, y, width, height = self._apply_adaptive_roi(camera_id, x, y, width, height, w, h)
        
        # Extraer región según la forma
        shape = roi_config.get("shape", "rectangle")
        
        if shape == "rectangle":
            roi_frame = frame[y:y+height, x:x+width]
        elif shape == "circle":
            roi_frame, x, y, width, height = self._apply_circular_roi(frame, x, y, width, height)
        elif shape == "polygon":
            roi_frame, x, y, width, height = self._apply_polygon_roi(frame, roi_config.get("polygon_points", []), w, h)
        else:
            roi_frame = frame[y:y+height, x:x+width]
        
        # Aplicar rotación si es necesaria
        rotation = roi_config.get("rotation", 0)
        if rotation != 0:
            roi_frame = self._rotate_frame(roi_frame, rotation)
        
        roi_offset = {"x": x, "y": y, "width": width, "height": height}
        
        return roi_frame, roi_offset
    
    def _apply_adaptive_roi(self, camera_id: str, x: int, y: int, width: int, height: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Aplica ROI adaptativa basada en detecciones previas"""
        if camera_id not in self.adaptive_roi_history:
            self.adaptive_roi_history[camera_id] = []
        
        history = self.adaptive_roi_history[camera_id]
        
        # Si tenemos historial de detecciones, ajustar ROI
        if len(history) > 0:
            # Calcular centro promedio de detecciones recientes
            recent_detections = history[-10:]  # Últimas 10 detecciones
            if recent_detections:
                avg_x = sum(det["center_x"] for det in recent_detections) / len(recent_detections)
                avg_y = sum(det["center_y"] for det in recent_detections) / len(recent_detections)
                
                # Obtener margen de la configuración
                roi_config = self._get_roi_config(camera_id)
                margin = roi_config.get("margin", 0.1)
                
                # Calcular nueva ROI centrada en las detecciones
                margin_w = int(width * margin)
                margin_h = int(height * margin)
                
                new_x = max(0, int(avg_x - width//2 - margin_w))
                new_y = max(0, int(avg_y - height//2 - margin_h))
                new_width = min(frame_w - new_x, width + 2*margin_w)
                new_height = min(frame_h - new_y, height + 2*margin_h)
                
                return new_x, new_y, new_width, new_height
        
        return x, y, width, height
    
    def _apply_circular_roi(self, frame: np.ndarray, x: int, y: int, width: int, height: int) -> Tuple[np.ndarray, int, int, int, int]:
        """Aplica ROI circular"""
        h, w = frame.shape[:2]
        
        # Calcular centro y radio
        center_x = x + width // 2
        center_y = y + height // 2
        radius = min(width, height) // 2
        
        # Crear máscara circular
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Aplicar máscara
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Extraer región rectangular que contiene el círculo
        x_start = max(0, center_x - radius)
        y_start = max(0, center_y - radius)
        x_end = min(w, center_x + radius)
        y_end = min(h, center_y + radius)
        
        roi_frame = masked_frame[y_start:y_end, x_start:x_end]
        
        return roi_frame, x_start, y_start, x_end - x_start, y_end - y_start
    
    def _apply_polygon_roi(self, frame: np.ndarray, polygon_points: List, frame_w: int, frame_h: int) -> Tuple[np.ndarray, int, int, int, int]:
        """Aplica ROI poligonal"""
        if not polygon_points or len(polygon_points) < 3:
            # Si no hay puntos válidos, usar frame completo
            return frame, 0, 0, frame.shape[1], frame.shape[0]
        
        h, w = frame.shape[:2]
        
        # Convertir puntos a coordenadas absolutas si son porcentajes
        abs_points = []
        for point in polygon_points:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                px, py = point
                if 0 <= px <= 1 and 0 <= py <= 1:  # Son porcentajes
                    abs_points.append((int(px * w), int(py * h)))
                else:  # Son píxeles absolutos
                    abs_points.append((int(px), int(py)))
        
        if len(abs_points) < 3:
            return frame, 0, 0, frame.shape[1], frame.shape[0]
        
        # Crear máscara poligonal
        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(abs_points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        
        # Aplicar máscara
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Encontrar bounding box del polígono
        x_coords = [p[0] for p in abs_points]
        y_coords = [p[1] for p in abs_points]
        
        x_min = max(0, min(x_coords))
        y_min = max(0, min(y_coords))
        x_max = min(w, max(x_coords))
        y_max = min(h, max(y_coords))
        
        roi_frame = masked_frame[y_min:y_max, x_min:x_max]
        
        return roi_frame, x_min, y_min, x_max - x_min, y_max - y_min
    
    def _rotate_frame(self, frame: np.ndarray, angle: int) -> np.ndarray:
        """Rota el frame según el ángulo especificado"""
        if angle == 0:
            return frame
        
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Crear matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcular nuevas dimensiones
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Ajustar matriz de rotación para centrar la imagen
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Aplicar rotación
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (new_w, new_h))
        
        return rotated_frame
    
    def _adjust_detections_for_roi(self, detections: List, roi_offset: Dict) -> List:
        """Ajusta coordenadas de detecciones de personas al frame completo"""
        if not detections:
            return detections
        
        adjusted_detections = []
        offset_x = roi_offset["x"]
        offset_y = roi_offset["y"]
        
        for detection in detections:
            if len(detection) >= 4:  # (x1, y1, x2, y2, confidence)
                x1, y1, x2, y2 = detection[:4]
                adjusted_detection = (
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y
                ) + detection[4:]  # Mantener confidence y otros datos
                adjusted_detections.append(adjusted_detection)
        
        return adjusted_detections
    
    def _adjust_faces_for_roi(self, faces: List, roi_offset: Dict) -> List:
        """Ajusta coordenadas de caras al frame completo"""
        if not faces:
            return faces
        
        offset_x = roi_offset["x"]
        offset_y = roi_offset["y"]
        
        for face in faces:
            if hasattr(face, 'bbox'):
                face.bbox[0] += offset_x  # x1
                face.bbox[1] += offset_y  # y1
                face.bbox[2] += offset_x  # x2
                face.bbox[3] += offset_y  # y2
        
        return faces
    
    def _draw_roi_on_frame(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Dibuja la ROI en el frame"""
        roi_config = self._get_roi_config(camera_id)
        
        if not roi_config.get("enabled", False):
            return frame
        
        h, w = frame.shape[:2]
        
        # Calcular coordenadas
        if roi_config.get("use_percentage", True):
            x = int(roi_config["x"] * w)
            y = int(roi_config["y"] * h)
            width = int(roi_config["width"] * w)
            height = int(roi_config["height"] * h)
        else:
            x = int(roi_config["x"])
            y = int(roi_config["y"])
            width = int(roi_config["width"])
            height = int(roi_config["height"])
        
        # Validar límites
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        width = max(1, min(width, w-x))
        height = max(1, min(height, h-y))
        
        shape = roi_config.get("shape", "rectangle")
        color = (0, 255, 255)  # Amarillo
        thickness = 2
        
        if shape == "rectangle":
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
            cv2.putText(frame, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        elif shape == "circle":
            center_x = x + width // 2
            center_y = y + height // 2
            radius = min(width, height) // 2
            cv2.circle(frame, (center_x, center_y), radius, color, thickness)
            cv2.putText(frame, "ROI", (center_x - 15, center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        elif shape == "polygon":
            polygon_points = roi_config.get("polygon_points", [])
            if polygon_points and len(polygon_points) >= 3:
                abs_points = []
                for point in polygon_points:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        px, py = point
                        if 0 <= px <= 1 and 0 <= py <= 1:  # Son porcentajes
                            abs_points.append((int(px * w), int(py * h)))
                        else:  # Son píxeles absolutos
                            abs_points.append((int(px), int(py)))
                
                if len(abs_points) >= 3:
                    points_array = np.array(abs_points, dtype=np.int32)
                    cv2.polylines(frame, [points_array], True, color, thickness)
                    # Dibujar etiqueta en el primer punto
                    cv2.putText(frame, "ROI", (abs_points[0][0], abs_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        # Mostrar información adicional
        info_text = f"Shape: {shape}"
        if roi_config.get("adaptive", False):
            info_text += " (Adaptive)"
        if roi_config.get("rotation", 0) != 0:
            info_text += f" Rot: {roi_config['rotation']}°"
        
        cv2.putText(frame, info_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _update_adaptive_roi_history(self, camera_id: str, detections: List):
        """Actualiza el historial para ROI adaptativa"""
        if camera_id not in self.adaptive_roi_history:
            self.adaptive_roi_history[camera_id] = []
        
        current_time = time.time()
        
        # Agregar detecciones actuales
        for detection in detections:
            if len(detection) >= 4:
                x1, y1, x2, y2 = detection[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                self.adaptive_roi_history[camera_id].append({
                    "center_x": center_x,
                    "center_y": center_y,
                    "timestamp": current_time
                })
        
        # Limpiar historial antiguo (más de 30 segundos)
        self.adaptive_roi_history[camera_id] = [
            det for det in self.adaptive_roi_history[camera_id]
            if current_time - det["timestamp"] < 30
        ]

# Instancia global para streaming
stream_recognizer = FaceRecognizer()

def stream_frames_with_digest(url, username, password):
    auth = HTTPDigestAuth(username, password)

    with requests.get(url, auth=auth, stream=True) as r:
        if r.status_code != 200:
            print(f"[WARNING] Error al acceder al stream: {r.status_code}")
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
            print(f"[WARNING] Error al acceder al stream: {r.status_code}")
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