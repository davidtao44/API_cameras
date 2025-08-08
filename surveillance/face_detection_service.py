"""Face Detection Service

Handles all face detection and recognition functionality using InsightFace.
Separated from the main surveillance service for better modularity.
"""

from cgi import print_form
import numpy as np
from insightface.app import FaceAnalysis
import json
import os
from typing import List, Tuple, Dict, Any
import cv2

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.55

class FaceDetectionService:
    """Service for face detection and recognition using InsightFace"""
    
    def __init__(self):
        self.app = None
        self.known_faces = {}
        self._setup_insightface()
        self._load_embeddings()
    
    def _setup_insightface(self):
        """Configure InsightFace with GPU/CPU support"""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"[INFO] Available ONNX providers: {available_providers}")
            
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                print("[INFO] Configuring CUDA for GPU")
            
            if 'CPUExecutionProvider' in available_providers:
                providers.append('CPUExecutionProvider')
                print("[INFO] CPU fallback available")
            
            if not providers:
                providers = ['CPUExecutionProvider']
                print("[WARNING] Only CPU available")
            
            self.app = FaceAnalysis(providers=providers)
            
            if 'CUDAExecutionProvider' in [p[0] if isinstance(p, tuple) else p for p in providers]:
                self.app.prepare(ctx_id=0, det_size=(2560, 1440))
                print("[INFO] InsightFace configured with GPU")
            else:
                self.app.prepare(ctx_id=-1, det_size=(320, 320))
                print("[INFO] InsightFace configured with CPU")
                
        except Exception as e:
            print(f"[ERROR] Error configuring InsightFace: {str(e)}")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("[INFO] Using CPU fallback")
    
    def _load_embeddings(self):
        """Load known face embeddings from file"""
        try:
            if os.path.exists(EMBEDDINGS_FILE):
                with open(EMBEDDINGS_FILE, 'r') as f:
                    data = json.load(f)
                    for name, embeddings in data.items():
                        normalized_embeddings = []
                        for emb in embeddings:
                            emb_array = np.array(emb)
                            # Normalize the embedding
                            norm = np.linalg.norm(emb_array)
                            if norm > 0:
                                emb_array = emb_array / norm
                            normalized_embeddings.append(emb_array)
                        self.known_faces[name] = normalized_embeddings
                print(f"[INFO] Loaded {len(self.known_faces)} known faces")
            else:
                print(f"[WARNING] Embeddings file {EMBEDDINGS_FILE} not found")
                self.known_faces = {}
        except Exception as e:
            print(f"[ERROR] Error loading embeddings: {e}")
            self.known_faces = {}
    
    def detect_faces(self, img: np.ndarray, detection_threshold: float = 0.5) -> List[Any]:
        """Detect faces in an image
        
        Args:
            img: Input image as numpy array
            detection_threshold: Confidence threshold for face detection (0.0 to 1.0)
            
        Returns:
            List of detected face objects
        """
        if self.app is None:
            return []
        
        try:
            # Get faces and filter by detection confidence
            faces = self.app.get(img)
            filtered_faces = [face for face in faces if face.det_score >= detection_threshold]
            return filtered_faces
        except Exception as e:
            print(f"[ERROR] Error detecting faces: {e}")
            return []
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize a face from its embedding
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            Tuple of (name, confidence_score)
        """
        
        # Iniciando reconocimiento de rostro
        
        best_match = "Unknown"
        max_similarity = -1
        
        # Para cada persona conocida, encontrar la mejor coincidencia entre sus embeddings
        for name, embeddings in self.known_faces.items():
            person_best_similarity = -1
            # Encontrar la mejor coincidencia para esta persona específica
            for i, known_embedding in enumerate(embeddings):
                similarity = self.calculate_similarity(face_embedding, known_embedding)
                
                if similarity > person_best_similarity:
                    person_best_similarity = similarity
            
            # Solo considerar esta persona si su mejor coincidencia supera el threshold
            # y es mejor que cualquier otra persona evaluada hasta ahora
            if person_best_similarity > THRESHOLD and person_best_similarity > max_similarity:
                best_match = name
                max_similarity = person_best_similarity
        return best_match, max_similarity
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two normalized embeddings
        
        Args:
            embedding1: First normalized embedding vector
            embedding2: Second normalized embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            # Since embeddings are normalized, dot product gives cosine similarity
            return np.dot(embedding1, embedding2)
        except Exception as e:
            print(f"[ERROR] Error calculating similarity: {e}")
            return 0.0
    
    def process_faces_in_image(self, img: np.ndarray, min_face_size: int = 30) -> List[Dict[str, Any]]:
        """Detect and recognize all faces in an image
        
        Args:
            img: Input image as numpy array
            min_face_size: Minimum size (width or height) for a valid face bounding box in pixels
            
        Returns:
            List of dictionaries containing face information
        """
        faces = self.detect_faces(img)
        
        results = []
        
        for i, face in enumerate(faces):
            # Get bounding box first to validate size
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox
            
            # Calculate bounding box dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Validate minimum face size
            if width < min_face_size or height < min_face_size:
                continue  # Skip faces that are too small
            
            # Normalize the face embedding
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            
            if norm > 0:
                embedding = embedding / norm
            else:
                continue
            
            name, confidence = self.recognize_face(embedding)
            
            face_info = {
                'name': name,
                'confidence': confidence,
                'bbox': bbox,
                'embedding': embedding,
                'face_size': {'width': width, 'height': height}
            }
            results.append(face_info)
        return results
    
    def draw_face_detection(self, frame: np.ndarray, face_info: Dict[str, Any]) -> np.ndarray:
        """Draw face detection on frame
        
        Args:
            frame: Image frame to draw on
            face_info: Face information dictionary
            
        Returns:
            Frame with detection drawn
        """
        bbox = face_info['bbox']
        name = face_info['name']
        confidence = face_info['confidence']
        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label with name and confidence
        label = f"{name}"
        if confidence > 0:
            label += f" ({confidence:.2f})"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10), 
                     (bbox[0] + text_width, bbox[1]), color, -1)
        
        # Text
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def reload_embeddings(self):
        """Reload embeddings from file"""
        self._load_embeddings()
    
    def get_known_faces_count(self) -> int:
        """Get number of known faces"""
        return len(self.known_faces)
    
    def get_known_faces_names(self) -> List[str]:
        """Get list of known face names"""
        return list(self.known_faces.keys())

# Global instance
face_detection_service = FaceDetectionService()