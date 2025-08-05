"""Physical Identification Service

Servicio para gestionar la identificación de personas basada en características físicas.
Permite mantener un registro persistente de personas identificadas sin reconocimiento facial.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
from .physical_characteristics_service import PhysicalCharacteristics

@dataclass
class PhysicalPersonProfile:
    """Perfil de una persona basado en características físicas"""
    physical_id: str
    first_seen: datetime
    last_seen: datetime
    total_detections: int
    cameras_seen: List[str]
    characteristics_history: List[Dict[str, Any]]
    confidence_scores: List[float]
    average_confidence: float
    description: str
    estimated_height_cm: Optional[float] = None
    dominant_clothing_colors: List[str] = None
    
    def __post_init__(self):
        if self.dominant_clothing_colors is None:
            self.dominant_clothing_colors = []
    
    def update_detection(self, characteristics: PhysicalCharacteristics, camera_id: str):
        """Actualiza el perfil con una nueva detección"""
        current_time = datetime.now()
        self.last_seen = current_time
        self.total_detections += 1
        
        # Agregar cámara si no está en la lista
        if camera_id not in self.cameras_seen:
            self.cameras_seen.append(camera_id)
        
        # Agregar características al historial
        char_dict = {
            'timestamp': current_time.isoformat(),
            'camera_id': camera_id,
            'estimated_height_cm': characteristics.estimated_height_cm,
            'dominant_clothing_colors': characteristics.dominant_clothing_colors,
            'clothing_description': characteristics.clothing_description,
            'confidence': characteristics.confidence
        }
        self.characteristics_history.append(char_dict)
        
        # Mantener solo las últimas 20 detecciones
        if len(self.characteristics_history) > 20:
            self.characteristics_history = self.characteristics_history[-20:]
        
        # Actualizar confianza
        self.confidence_scores.append(characteristics.confidence)
        if len(self.confidence_scores) > 20:
            self.confidence_scores = self.confidence_scores[-20:]
        
        self.average_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        
        # Actualizar características promedio
        self._update_average_characteristics()
    
    def _update_average_characteristics(self):
        """Actualiza las características promedio basadas en el historial"""
        if not self.characteristics_history:
            return
        
        # Calcular altura promedio
        heights = [c['estimated_height_cm'] for c in self.characteristics_history 
                  if c['estimated_height_cm'] is not None]
        if heights:
            self.estimated_height_cm = sum(heights) / len(heights)
        
        # Encontrar colores más frecuentes
        color_counts = defaultdict(int)
        for char in self.characteristics_history:
            for color in char.get('dominant_clothing_colors', []):
                color_counts[color] += 1
        
        # Tomar los 3 colores más frecuentes
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        self.dominant_clothing_colors = [color for color, _ in sorted_colors[:3]]
        
        # Actualizar descripción
        recent_descriptions = [c['clothing_description'] for c in self.characteristics_history[-5:] 
                             if c['clothing_description']]
        if recent_descriptions:
            self.description = recent_descriptions[-1]  # Usar la más reciente

class PhysicalIdentificationService:
    """Servicio para gestionar identificación física de personas"""
    
    def __init__(self, data_file: str = "physical_persons_db.json"):
        self.data_file = os.path.join(os.path.dirname(__file__), '..', 'data', data_file)
        self.persons_db: Dict[str, PhysicalPersonProfile] = {}
        self.similarity_threshold = 0.75  # Umbral para considerar la misma persona
        self.min_detections_for_persistence = 3  # Mínimo de detecciones para persistir
        
        # Crear directorio de datos si no existe
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # Cargar base de datos existente
        self._load_database()
        
        print(f"[INFO] Physical Identification Service initialized with {len(self.persons_db)} persons")
    
    def _load_database(self):
        """Carga la base de datos desde archivo"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for physical_id, person_data in data.items():
                    # Convertir timestamps de string a datetime
                    person_data['first_seen'] = datetime.fromisoformat(person_data['first_seen'])
                    person_data['last_seen'] = datetime.fromisoformat(person_data['last_seen'])
                    
                    self.persons_db[physical_id] = PhysicalPersonProfile(**person_data)
                
                print(f"[INFO] Loaded {len(self.persons_db)} persons from database")
        except Exception as e:
            print(f"[ERROR] Error loading physical persons database: {e}")
            self.persons_db = {}
    
    def _save_database(self):
        """Guarda la base de datos en archivo"""
        try:
            # Convertir a formato serializable
            data = {}
            for physical_id, person in self.persons_db.items():
                person_dict = asdict(person)
                # Convertir datetime a string
                person_dict['first_seen'] = person.first_seen.isoformat()
                person_dict['last_seen'] = person.last_seen.isoformat()
                data[physical_id] = person_dict
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[ERROR] Error saving physical persons database: {e}")
    
    def find_matching_person(self, characteristics: PhysicalCharacteristics) -> Optional[PhysicalPersonProfile]:
        """Busca una persona existente que coincida con las características"""
        best_match = None
        best_similarity = 0.0
        
        for person in self.persons_db.values():
            similarity = self._calculate_similarity(characteristics, person)
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = person
        
        return best_match
    
    def _calculate_similarity(self, characteristics: PhysicalCharacteristics, person: PhysicalPersonProfile) -> float:
        """Calcula la similitud entre características y un perfil de persona"""
        similarity_score = 0.0
        total_weight = 0.0
        
        # Comparar altura (peso: 0.3)
        if characteristics.estimated_height_cm and person.estimated_height_cm:
            height_diff = abs(characteristics.estimated_height_cm - person.estimated_height_cm)
            height_similarity = max(0, 1 - (height_diff / 25))  # 25cm tolerance
            similarity_score += height_similarity * 0.3
            total_weight += 0.3
        
        # Comparar colores de ropa (peso: 0.5)
        if characteristics.dominant_clothing_colors and person.dominant_clothing_colors:
            common_colors = set(characteristics.dominant_clothing_colors) & set(person.dominant_clothing_colors)
            max_colors = max(len(characteristics.dominant_clothing_colors), len(person.dominant_clothing_colors))
            color_similarity = len(common_colors) / max_colors if max_colors > 0 else 0
            similarity_score += color_similarity * 0.5
            total_weight += 0.5
        
        # Comparar proporciones corporales (peso: 0.2)
        if (characteristics.body_proportions and 
            'aspect_ratio' in characteristics.body_proportions):
            # Usar el historial más reciente para comparar proporciones
            if person.characteristics_history:
                recent_chars = person.characteristics_history[-3:]  # Últimas 3 detecciones
                avg_ratio = 0
                ratio_count = 0
                
                for char_data in recent_chars:
                    if 'body_proportions' in char_data and char_data['body_proportions']:
                        if 'aspect_ratio' in char_data['body_proportions']:
                            avg_ratio += char_data['body_proportions']['aspect_ratio']
                            ratio_count += 1
                
                if ratio_count > 0:
                    avg_ratio /= ratio_count
                    ratio_diff = abs(characteristics.body_proportions['aspect_ratio'] - avg_ratio)
                    ratio_similarity = max(0, 1 - (ratio_diff / 1.0))  # 1.0 tolerance
                    similarity_score += ratio_similarity * 0.2
                    total_weight += 0.2
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    def register_detection(self, characteristics: PhysicalCharacteristics, camera_id: str, 
                          physical_id: Optional[str] = None) -> str:
        """Registra una nueva detección y devuelve el ID físico"""
        # Buscar persona existente
        matching_person = self.find_matching_person(characteristics)
        
        if matching_person:
            # Actualizar persona existente
            matching_person.update_detection(characteristics, camera_id)
            result_id = matching_person.physical_id
            print(f"[INFO] Updated existing person {result_id} (similarity found)")
        else:
            # Crear nueva persona
            if not physical_id:
                physical_id = self._generate_physical_id(characteristics)
            
            new_person = PhysicalPersonProfile(
                physical_id=physical_id,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                total_detections=1,
                cameras_seen=[camera_id],
                characteristics_history=[],
                confidence_scores=[characteristics.confidence],
                average_confidence=characteristics.confidence,
                description=characteristics.clothing_description,
                estimated_height_cm=characteristics.estimated_height_cm,
                dominant_clothing_colors=characteristics.dominant_clothing_colors.copy()
            )
            
            new_person.update_detection(characteristics, camera_id)
            self.persons_db[physical_id] = new_person
            result_id = physical_id
            print(f"[INFO] Created new person {result_id}")
        
        # Guardar base de datos periódicamente
        if len(self.persons_db) % 10 == 0:  # Cada 10 personas nuevas
            self._save_database()
        
        return result_id
    
    def _generate_physical_id(self, characteristics: PhysicalCharacteristics) -> str:
        """Genera un ID único para una nueva persona"""
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
        
        # Timestamp para unicidad
        timestamp = datetime.now().strftime("%m%d%H%M")
        components.append(timestamp)
        
        base_id = "_".join(components) if components else f"PERSON_{timestamp}"
        
        # Asegurar unicidad
        counter = 1
        final_id = base_id
        while final_id in self.persons_db:
            final_id = f"{base_id}_{counter}"
            counter += 1
        
        return final_id
    
    def get_person_info(self, physical_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de una persona"""
        if physical_id not in self.persons_db:
            return None
        
        person = self.persons_db[physical_id]
        return {
            'physical_id': person.physical_id,
            'first_seen': person.first_seen.isoformat(),
            'last_seen': person.last_seen.isoformat(),
            'total_detections': person.total_detections,
            'cameras_seen': person.cameras_seen,
            'average_confidence': person.average_confidence,
            'description': person.description,
            'estimated_height_cm': person.estimated_height_cm,
            'dominant_clothing_colors': person.dominant_clothing_colors,
            'recent_detections': person.characteristics_history[-5:]  # Últimas 5
        }
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """Obtiene información de todas las personas registradas"""
        return [self.get_person_info(pid) for pid in self.persons_db.keys()]
    
    def cleanup_old_persons(self, days_threshold: int = 30):
        """Limpia personas que no han sido vistas en X días"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        to_remove = []
        
        for physical_id, person in self.persons_db.items():
            if (person.last_seen < cutoff_date and 
                person.total_detections < self.min_detections_for_persistence):
                to_remove.append(physical_id)
        
        for physical_id in to_remove:
            del self.persons_db[physical_id]
        
        if to_remove:
            print(f"[INFO] Cleaned up {len(to_remove)} old persons from database")
            self._save_database()
    
    def save_database_now(self):
        """Fuerza el guardado inmediato de la base de datos"""
        self._save_database()

# Instancia global
physical_identification_service = PhysicalIdentificationService()