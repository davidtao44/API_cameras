import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import math
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PersonTrack:
    """Representa un track de una persona detectada"""
    track_id: int
    last_face_name: str
    last_face_confidence: float
    last_face_time: datetime
    last_body_bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    last_face_bbox: Optional[Tuple[int, int, int, int]]
    last_seen: datetime
    face_recognition_history: List[Tuple[str, float, datetime]]  # name, confidence, time
    body_position_history: List[Tuple[Tuple[int, int, int, int], datetime]]  # bbox, time
    is_active: bool = True
    confidence_score: float = 0.0
    
    def update_face_recognition(self, name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        """Actualiza el reconocimiento facial del track"""
        current_time = datetime.now()
        self.last_face_name = name
        self.last_face_confidence = confidence
        self.last_face_time = current_time
        self.last_face_bbox = bbox
        self.last_seen = current_time
        
        # Agregar al historial
        self.face_recognition_history.append((name, confidence, current_time))
        
        # Mantener solo los últimos 10 reconocimientos
        if len(self.face_recognition_history) > 10:
            self.face_recognition_history = self.face_recognition_history[-10:]
    
    def update_body_position(self, bbox: Tuple[int, int, int, int]):
        """Actualiza la posición del cuerpo del track"""
        current_time = datetime.now()
        self.last_body_bbox = bbox
        self.last_seen = current_time
        
        # Agregar al historial de posiciones
        self.body_position_history.append((bbox, current_time))
        
        # Mantener solo las últimas 20 posiciones
        if len(self.body_position_history) > 20:
            self.body_position_history = self.body_position_history[-20:]
    
    def get_best_face_name(self, time_window_seconds: int = 10) -> Tuple[str, float]:
        """Obtiene el mejor nombre reconocido en una ventana de tiempo"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=time_window_seconds)
        
        # Filtrar reconocimientos recientes
        recent_recognitions = [
            (name, conf) for name, conf, timestamp in self.face_recognition_history
            if timestamp >= cutoff_time and name != "Unknown" and name != "Desconocido"
        ]
        
        if not recent_recognitions:
            return "Unknown", 0.0
        
        # Agrupar por nombre y calcular confianza promedio
        name_confidences = defaultdict(list)
        for name, conf in recent_recognitions:
            name_confidences[name].append(conf)
        
        # Encontrar el nombre con mayor confianza promedio
        best_name = "Unknown"
        best_confidence = 0.0
        
        for name, confidences in name_confidences.items():
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence > best_confidence:
                best_name = name
                best_confidence = avg_confidence
        
        return best_name, best_confidence
    
    def is_expired(self, max_age_seconds: int = 15) -> bool:
        """Verifica si el track ha expirado"""
        current_time = datetime.now()
        return (current_time - self.last_seen).total_seconds() > max_age_seconds

class PersonTrackingService:
    """Servicio de tracking de personas que vincula detección de cuerpo con reconocimiento facial"""
    
    def __init__(self):
        self.tracks: Dict[str, Dict[int, PersonTrack]] = {}  # camera_id -> {track_id: PersonTrack}
        self.next_track_id: Dict[str, int] = {}  # camera_id -> next_id
        self.face_persistence_time = 10  # segundos para mantener reconocimiento facial
        self.track_max_age = 15  # segundos antes de eliminar un track
        self.max_distance_threshold = 100  # píxeles máximos para asociar cuerpo con cara
        self.iou_threshold = 0.3  # IoU mínimo para asociar detecciones
        
        print("[INFO] Person Tracking Service initialized")
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calcula Intersection over Union entre dos cajas"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calcula la distancia entre los centros de dos cajas"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def face_overlaps_with_body(self, face_bbox: Tuple[int, int, int, int], 
                               body_bbox: Tuple[int, int, int, int]) -> bool:
        """Verifica si una cara está dentro o cerca de una detección de cuerpo"""
        fx1, fy1, fx2, fy2 = face_bbox
        bx1, by1, bx2, by2 = body_bbox
        
        # La cara debe estar en la parte superior del cuerpo
        face_center_y = (fy1 + fy2) / 2
        body_top_third = by1 + (by2 - by1) * 0.4  # 40% superior del cuerpo
        
        # Verificar si la cara está en la región esperada del cuerpo
        if face_center_y > body_top_third:
            return False
        
        # Verificar superposición horizontal
        face_center_x = (fx1 + fx2) / 2
        if face_center_x < bx1 or face_center_x > bx2:
            return False
        
        # Calcular IoU para verificación adicional
        iou = self.calculate_iou(face_bbox, body_bbox)
        return iou > 0.1  # Mínima superposición requerida
    
    def update_tracks(self, camera_id: str, person_boxes: List[Tuple[int, int, int, int, float]], 
                     face_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Actualiza los tracks con nuevas detecciones y devuelve resultados enriquecidos"""
        
        if camera_id not in self.tracks:
            self.tracks[camera_id] = {}
            self.next_track_id[camera_id] = 1
        
        current_tracks = self.tracks[camera_id]
        
        # Limpiar tracks expirados
        expired_tracks = [track_id for track_id, track in current_tracks.items() 
                         if track.is_expired(self.track_max_age)]
        for track_id in expired_tracks:
            del current_tracks[track_id]
        
        # Convertir person_boxes a formato sin confianza para procesamiento
        person_bboxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, conf in person_boxes]
        
        # Asociar detecciones de cuerpo con tracks existentes
        used_tracks = set()
        used_bodies = set()
        
        for i, body_bbox in enumerate(person_bboxes):
            best_track_id = None
            best_iou = 0
            
            for track_id, track in current_tracks.items():
                if track_id in used_tracks:
                    continue
                
                iou = self.calculate_iou(body_bbox, track.last_body_bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                current_tracks[best_track_id].update_body_position(body_bbox)
                used_tracks.add(best_track_id)
                used_bodies.add(i)
        
        # Crear nuevos tracks para cuerpos no asociados
        for i, body_bbox in enumerate(person_bboxes):
            if i not in used_bodies:
                track_id = self.next_track_id[camera_id]
                self.next_track_id[camera_id] += 1
                
                new_track = PersonTrack(
                    track_id=track_id,
                    last_face_name="Unknown",
                    last_face_confidence=0.0,
                    last_face_time=datetime.now(),
                    last_body_bbox=body_bbox,
                    last_face_bbox=None,
                    last_seen=datetime.now(),
                    face_recognition_history=[],
                    body_position_history=[(body_bbox, datetime.now())]
                )
                current_tracks[track_id] = new_track
        
        # Asociar reconocimientos faciales con tracks
        for face_result in face_results:
            face_bbox = tuple(face_result['bbox'])
            face_name = face_result['name']
            face_confidence = face_result['confidence']
            
            # Encontrar el track más cercano que contenga esta cara
            best_track_id = None
            best_score = 0
            
            for track_id, track in current_tracks.items():
                if self.face_overlaps_with_body(face_bbox, track.last_body_bbox):
                    # Calcular score basado en distancia y IoU
                    distance = self.calculate_distance(face_bbox, track.last_body_bbox)
                    iou = self.calculate_iou(face_bbox, track.last_body_bbox)
                    score = iou - (distance / 1000)  # Normalizar distancia
                    
                    if score > best_score:
                        best_score = score
                        best_track_id = track_id
            
            if best_track_id is not None:
                current_tracks[best_track_id].update_face_recognition(
                    face_name, face_confidence, face_bbox
                )
        
        # Generar resultados enriquecidos
        enriched_results = []
        
        for track_id, track in current_tracks.items():
            # Obtener el mejor nombre reconocido en la ventana de tiempo
            best_name, best_confidence = track.get_best_face_name(self.face_persistence_time)
            
            # Crear resultado enriquecido
            enriched_result = {
                'track_id': track_id,
                'body_bbox': track.last_body_bbox,
                'face_bbox': track.last_face_bbox,
                'name': best_name,
                'confidence': best_confidence,
                'last_face_time': track.last_face_time,
                'time_since_last_face': (datetime.now() - track.last_face_time).total_seconds(),
                'is_persistent': best_name != "Unknown" and best_confidence > 0,
                'recognition_history_count': len(track.face_recognition_history)
            }
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def draw_enhanced_detections(self, frame: np.ndarray, enriched_results: List[Dict[str, Any]]) -> np.ndarray:
        """Dibuja las detecciones mejoradas con tracking en el frame"""
        enhanced_frame = frame.copy()
        
        for result in enriched_results:
            track_id = result['track_id']
            body_bbox = result['body_bbox']
            face_bbox = result['face_bbox']
            name = result['name']
            confidence = result['confidence']
            time_since_face = result['time_since_last_face']
            is_persistent = result['is_persistent']
            
            # Color basado en el estado del reconocimiento
            if is_persistent:
                if time_since_face < 3:  # Reconocimiento reciente
                    color = (0, 255, 0)  # Verde
                elif time_since_face < 8:  # Reconocimiento medio
                    color = (0, 255, 255)  # Amarillo
                else:  # Reconocimiento persistente
                    color = (0, 165, 255)  # Naranja
            else:
                color = (0, 0, 255)  # Rojo para desconocido
            
            # Dibujar caja del cuerpo
            bx1, by1, bx2, by2 = body_bbox
            cv2.rectangle(enhanced_frame, (bx1, by1), (bx2, by2), color, 2)
            
            # Dibujar caja de la cara si existe
            if face_bbox:
                fx1, fy1, fx2, fy2 = face_bbox
                cv2.rectangle(enhanced_frame, (fx1, fy1), (fx2, fy2), color, 1)
                # Línea conectando cara con cuerpo
                face_center = (int((fx1 + fx2) / 2), int((fy1 + fy2) / 2))
                body_center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                cv2.line(enhanced_frame, face_center, body_center, color, 1)
            
            # Etiqueta con información del track
            label = f"ID:{track_id} {name}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            if time_since_face < self.face_persistence_time:
                label += f" [{time_since_face:.1f}s]"
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(enhanced_frame, (bx1, by1 - text_height - 10), 
                         (bx1 + text_width, by1), color, -1)
            
            # Texto
            cv2.putText(enhanced_frame, label, (bx1, by1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Información general del tracking
        active_tracks = len(enriched_results)
        persistent_tracks = sum(1 for r in enriched_results if r['is_persistent'])
        
        info_text = f"Tracks: {active_tracks} | Persistent: {persistent_tracks}"
        cv2.putText(enhanced_frame, info_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return enhanced_frame
    
    def get_tracking_stats(self, camera_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas del tracking para una cámara"""
        if camera_id not in self.tracks:
            return {"active_tracks": 0, "persistent_tracks": 0, "total_tracks_created": 0}
        
        current_tracks = self.tracks[camera_id]
        persistent_count = 0
        
        for track in current_tracks.values():
            best_name, _ = track.get_best_face_name(self.face_persistence_time)
            if best_name != "Unknown":
                persistent_count += 1
        
        return {
            "active_tracks": len(current_tracks),
            "persistent_tracks": persistent_count,
            "total_tracks_created": self.next_track_id.get(camera_id, 1) - 1
        }
    
    def cleanup_camera(self, camera_id: str):
        """Limpia todos los tracks de una cámara"""
        if camera_id in self.tracks:
            del self.tracks[camera_id]
        if camera_id in self.next_track_id:
            del self.next_track_id[camera_id]

# Instancia global del servicio
person_tracking_service = PersonTrackingService()