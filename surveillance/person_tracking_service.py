import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import math
from dataclasses import dataclass
from collections import defaultdict
from .physical_characteristics_service import physical_characteristics_service, PhysicalCharacteristics
from .physical_identification_service import physical_identification_service

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
    # Características físicas para identificación
    physical_characteristics: Optional[PhysicalCharacteristics] = None
    physical_id: Optional[str] = None  # ID basado en características físicas
    physical_confidence: float = 0.0
    physical_history: List[Tuple[PhysicalCharacteristics, datetime]] = None
    
    def __post_init__(self):
        if self.physical_history is None:
            self.physical_history = []
    
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
    
    def update_physical_characteristics(self, characteristics: PhysicalCharacteristics):
        """Actualiza las características físicas del track"""
        current_time = datetime.now()
        self.physical_characteristics = characteristics
        self.last_seen = current_time
        
        # Agregar al historial
        self.physical_history.append((characteristics, current_time))
        
        # Mantener solo las últimas 5 características
        if len(self.physical_history) > 5:
            self.physical_history = self.physical_history[-5:]
        
        # Generar ID físico basado en características (se actualiza externamente por physical_identification_service)
        if not self.physical_id:
            self.physical_id = self._generate_physical_id(characteristics)
        self.physical_confidence = characteristics.confidence
    
    def _generate_physical_id(self, characteristics: PhysicalCharacteristics) -> str:
        """Genera un ID único basado en características físicas (deprecated - usar physical_identification_service)"""
        # Este método se mantiene por compatibilidad pero se recomienda usar physical_identification_service
        components = []
        
        # Altura (redondeada a grupos de 10cm)
        if characteristics.estimated_height_cm:
            height_group = int(characteristics.estimated_height_cm // 10) * 10
            components.append(f"H{height_group}")
        
        # Colores dominantes (máximo 2)
        if characteristics.dominant_clothing_colors:
            colors = characteristics.dominant_clothing_colors[:2]
            color_code = "".join([c[:2].upper() for c in colors])
            components.append(f"C{color_code}")
        
        # Proporciones corporales
        if characteristics.body_proportions and 'aspect_ratio' in characteristics.body_proportions:
            ratio = characteristics.body_proportions['aspect_ratio']
            ratio_group = "S" if ratio < 2.0 else "M" if ratio < 2.5 else "T"  # Short, Medium, Tall
            components.append(f"R{ratio_group}")
        
        return "_".join(components) if components else "UNKNOWN"
    
    def get_physical_similarity(self, other_characteristics: PhysicalCharacteristics) -> float:
        """Calcula la similitud física con otras características"""
        if not self.physical_characteristics:
            return 0.0
        
        similarity_score = 0.0
        total_weight = 0.0
        
        # Comparar altura (peso: 0.3)
        if (self.physical_characteristics.estimated_height_cm and 
            other_characteristics.estimated_height_cm):
            height_diff = abs(self.physical_characteristics.estimated_height_cm - 
                            other_characteristics.estimated_height_cm)
            height_similarity = max(0, 1 - (height_diff / 30))  # 30cm tolerance
            similarity_score += height_similarity * 0.3
            total_weight += 0.3
        
        # Comparar colores de ropa (peso: 0.5)
        if (self.physical_characteristics.dominant_clothing_colors and 
            other_characteristics.dominant_clothing_colors):
            common_colors = set(self.physical_characteristics.dominant_clothing_colors) & \
                          set(other_characteristics.dominant_clothing_colors)
            max_colors = max(len(self.physical_characteristics.dominant_clothing_colors),
                           len(other_characteristics.dominant_clothing_colors))
            color_similarity = len(common_colors) / max_colors if max_colors > 0 else 0
            similarity_score += color_similarity * 0.5
            total_weight += 0.5
        
        # Comparar proporciones corporales (peso: 0.2)
        if (self.physical_characteristics.body_proportions and 
            other_characteristics.body_proportions and
            'aspect_ratio' in self.physical_characteristics.body_proportions and
            'aspect_ratio' in other_characteristics.body_proportions):
            ratio_diff = abs(self.physical_characteristics.body_proportions['aspect_ratio'] - 
                           other_characteristics.body_proportions['aspect_ratio'])
            ratio_similarity = max(0, 1 - (ratio_diff / 1.0))  # 1.0 tolerance
            similarity_score += ratio_similarity * 0.2
            total_weight += 0.2
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    def get_identification_info(self) -> Dict[str, Any]:
        """Obtiene información de identificación combinando facial y física"""
        # Priorizar reconocimiento facial si está disponible
        best_face_name, face_confidence = self.get_best_face_name()
        
        if best_face_name != "Unknown" and face_confidence > 0.7:
            return {
                'identification_type': 'facial',
                'name': best_face_name,
                'confidence': face_confidence,
                'physical_id': self.physical_id,
                'description': self.physical_characteristics.clothing_description if self.physical_characteristics else ""
            }
        elif self.physical_id and self.physical_confidence > 0.5:
            return {
                'identification_type': 'physical',
                'name': f"Persona_{self.physical_id}",
                'confidence': self.physical_confidence,
                'physical_id': self.physical_id,
                'description': self.physical_characteristics.clothing_description if self.physical_characteristics else ""
            }
        else:
            return {
                'identification_type': 'unknown',
                'name': "Desconocido",
                'confidence': 0.0,
                'physical_id': self.physical_id or "UNKNOWN",
                'description': self.physical_characteristics.clothing_description if self.physical_characteristics else ""
            }

class PersonTrackingService:
    """Servicio de tracking de personas que vincula detección de cuerpo con reconocimiento facial y características físicas"""
    
    def __init__(self, camera_id="unknown"):
        self.tracks: Dict[str, Dict[int, PersonTrack]] = {}  # camera_id -> {track_id: PersonTrack}
        self.next_track_id: Dict[str, int] = {}  # camera_id -> next_id
        self.face_persistence_time = 10  # segundos para mantener reconocimiento facial
        self.track_max_age = 15  # segundos antes de eliminar un track
        self.max_distance_threshold = 100  # píxeles máximos para asociar cuerpo con cara
        self.iou_threshold = 0.3  # IoU mínimo para asociar detecciones
        # Configuración para identificación física
        self.physical_similarity_threshold = 0.7  # Umbral para considerar similitud física
        self.physical_identification_enabled = True  # Habilitar identificación física
        self.physical_tracks_db: Dict[str, List[PersonTrack]] = {}  # Base de datos de tracks físicos por cámara
        self.camera_id = camera_id  # Identificador de la cámara para el registro de identificación física
        
        print(f"[INFO] Person Tracking Service initialized with physical identification support for camera: {camera_id}")
        
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
    
    def find_similar_physical_track(self, camera_id: str, characteristics: PhysicalCharacteristics) -> Optional[PersonTrack]:
        """Busca un track existente con características físicas similares"""
        if not self.physical_identification_enabled or camera_id not in self.physical_tracks_db:
            return None
        
        best_track = None
        best_similarity = 0.0
        
        for track in self.physical_tracks_db[camera_id]:
            if track.physical_characteristics:
                similarity = track.get_physical_similarity(characteristics)
                if similarity > self.physical_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_track = track
        
        return best_track
    
    def update_physical_tracks_db(self, camera_id: str, track: PersonTrack):
        """Actualiza la base de datos de tracks físicos"""
        if not self.physical_identification_enabled:
            return
        
        if camera_id not in self.physical_tracks_db:
            self.physical_tracks_db[camera_id] = []
        
        # Buscar si ya existe un track similar
        existing_track = None
        for existing in self.physical_tracks_db[camera_id]:
            if existing.track_id == track.track_id:
                existing_track = existing
                break
        
        if existing_track:
            # Actualizar track existente
            existing_track.physical_characteristics = track.physical_characteristics
            existing_track.physical_id = track.physical_id
            existing_track.physical_confidence = track.physical_confidence
        else:
            # Agregar nuevo track si tiene características físicas
            if track.physical_characteristics:
                self.physical_tracks_db[camera_id].append(track)
        
        # Limpiar tracks antiguos (mantener solo los últimos 50)
        if len(self.physical_tracks_db[camera_id]) > 50:
            self.physical_tracks_db[camera_id] = self.physical_tracks_db[camera_id][-50:]
    
    def update_tracks(self, camera_id: str, person_boxes: List[Tuple[int, int, int, int, float]], 
                     face_results: List[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
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
                
                # Analizar características físicas si el frame está disponible
                if frame is not None and self.physical_identification_enabled:
                    try:
                        characteristics = physical_characteristics_service.analyze_person_characteristics(
                            frame, body_bbox, camera_id
                        )
                        
                        # Registrar en el servicio de identificación física
                        physical_id = physical_identification_service.register_detection(
                            characteristics, 
                            camera_id=self.camera_id
                        )
                        
                        # Buscar track similar existente
                        similar_track = self.find_similar_physical_track(camera_id, characteristics)
                        
                        if similar_track:
                            # Usar información del track similar para identificación
                            new_track.physical_characteristics = characteristics
                            new_track.physical_id = physical_id
                            new_track.physical_confidence = min(characteristics.confidence, similar_track.physical_confidence)
                            print(f"[INFO] Track {track_id} asociado con persona física existente: {physical_id}")
                        else:
                            # Nuevo track con características únicas
                            new_track.update_physical_characteristics(characteristics)
                            new_track.physical_id = physical_id
                            print(f"[INFO] Nuevo track {track_id} con ID físico: {physical_id}")
                        
                        # Actualizar base de datos de tracks físicos
                        self.update_physical_tracks_db(camera_id, new_track)
                        
                    except Exception as e:
                        print(f"[ERROR] Error analizando características físicas: {e}")
                
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
        
        # Actualizar características físicas para tracks existentes
        if frame is not None and self.physical_identification_enabled:
            for track_id, track in current_tracks.items():
                try:
                    # Analizar características físicas periódicamente
                    time_since_last_analysis = (datetime.now() - track.last_seen).total_seconds()
                    if time_since_last_analysis > 2.0 or not track.physical_characteristics:  # Cada 2 segundos
                        characteristics = physical_characteristics_service.analyze_person_characteristics(
                            frame, track.last_body_bbox, camera_id
                        )
                        
                        # Registrar en el servicio de identificación física
                        physical_id = physical_identification_service.register_detection(
                            characteristics, 
                            camera_id=self.camera_id
                        )
                        
                        track.update_physical_characteristics(characteristics)
                        track.physical_id = physical_id
                        self.update_physical_tracks_db(camera_id, track)
                except Exception as e:
                    print(f"[ERROR] Error actualizando características físicas para track {track_id}: {e}")
        
        # Generar resultados enriquecidos
        enriched_results = []
        
        for track_id, track in current_tracks.items():
            # Obtener información de identificación combinada
            identification_info = track.get_identification_info()
            
            # Obtener el mejor nombre reconocido en la ventana de tiempo (para compatibilidad)
            best_name, best_confidence = track.get_best_face_name(self.face_persistence_time)
            
            # Crear resultado enriquecido con información física
            enriched_result = {
                'track_id': track_id,
                'body_bbox': track.last_body_bbox,
                'face_bbox': track.last_face_bbox,
                'name': identification_info['name'],
                'confidence': identification_info['confidence'],
                'identification_type': identification_info['identification_type'],
                'physical_id': identification_info['physical_id'],
                'description': identification_info['description'],
                'last_face_time': track.last_face_time,
                'time_since_last_face': (datetime.now() - track.last_face_time).total_seconds(),
                'is_persistent': identification_info['confidence'] > 0.5,
                'recognition_history_count': len(track.face_recognition_history),
                # Información adicional de características físicas
                'physical_characteristics': {
                    'estimated_height_cm': track.physical_characteristics.estimated_height_cm if track.physical_characteristics else None,
                    'dominant_clothing_colors': track.physical_characteristics.dominant_clothing_colors if track.physical_characteristics else [],
                    'clothing_description': track.physical_characteristics.clothing_description if track.physical_characteristics else "",
                    'body_proportions': track.physical_characteristics.body_proportions if track.physical_characteristics else {},
                    'confidence': track.physical_characteristics.confidence if track.physical_characteristics else 0.0
                }
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
            
            # Crear etiquetas con información de identificación
            identification_type = result.get('identification_type', 'unknown')
            physical_id = result.get('physical_id', '')
            description = result.get('description', '')
            
            # Etiqueta principal
            if identification_type == 'facial':
                label = f"ID:{track_id} {name} (Facial {confidence:.2f})"
            elif identification_type == 'physical':
                label = f"ID:{track_id} {name} (Físico {confidence:.2f})"
            else:
                label = f"ID:{track_id} {name}"
            
            # Información adicional
            info_lines = [label]
            
            # Agregar ID físico si está disponible
            if physical_id and physical_id != 'UNKNOWN':
                info_lines.append(f"Físico: {physical_id}")
            
            # Agregar descripción de ropa si está disponible
            if description and len(description) > 0:
                # Truncar descripción si es muy larga
                desc_short = description[:25] + "..." if len(description) > 25 else description
                info_lines.append(f"Ropa: {desc_short}")
            
            # Agregar altura si está disponible
            physical_chars = result.get('physical_characteristics', {})
            if physical_chars.get('estimated_height_cm'):
                height = physical_chars['estimated_height_cm']
                info_lines.append(f"Altura: ~{height}cm")
            
            # Calcular dimensiones del texto
            font_scale = 0.5
            font_thickness = 1
            line_height = 20
            max_width = 0
            
            for line in info_lines:
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                max_width = max(max_width, text_width)
            
            total_height = len(info_lines) * line_height + 10
            
            # Fondo para el texto
            cv2.rectangle(enhanced_frame, (bx1, by1 - total_height), 
                         (bx1 + max_width + 10, by1), color, -1)
            
            # Dibujar cada línea de texto
            for i, line in enumerate(info_lines):
                y_pos = by1 - total_height + (i + 1) * line_height
                cv2.putText(enhanced_frame, line, (bx1 + 5, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
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