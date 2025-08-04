"""Face Detection Service

Handles all face detection and recognition functionality using InsightFace.
Separated from the main surveillance service for better modularity.
"""

import numpy as np
from insightface.app import FaceAnalysis
import json
import os
from typing import List, Tuple, Dict, Any
import cv2

EMBEDDINGS_FILE = "embeddings_arcface.json"
THRESHOLD = 0.50

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
                self.app.prepare(ctx_id=0, det_size=(640, 640))
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
                        self.known_faces[name] = [np.array(emb) for emb in embeddings]
                print(f"[INFO] Loaded {len(self.known_faces)} known faces")
            else:
                print(f"[WARNING] Embeddings file {EMBEDDINGS_FILE} not found")
                self.known_faces = {}
        except Exception as e:
            print(f"[ERROR] Error loading embeddings: {e}")
            self.known_faces = {}
    
    def detect_faces(self, img: np.ndarray) -> List[Any]:
        """Detect faces in an image
        
        Args:
            img: Input image as numpy array
            
        Returns:
            List of detected face objects
        """
        if self.app is None:
            return []
        
        try:
            faces = self.app.get(img)
            return faces
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
        if not self.known_faces:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        max_similarity = -1
        
        for name, embeddings in self.known_faces.items():
            for known_embedding in embeddings:
                similarity = self.calculate_similarity(face_embedding, known_embedding)
                
                if similarity > THRESHOLD and similarity > max_similarity:
                    best_match = name
                    max_similarity = similarity
        
        return best_match, max_similarity
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        except Exception as e:
            print(f"[ERROR] Error calculating similarity: {e}")
            return 0.0
    
    def process_faces_in_image(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize all faces in an image
        
        Args:
            img: Input image as numpy array
            
        Returns:
            List of dictionaries containing face information
        """
        faces = self.detect_faces(img)
        results = []
        
        for face in faces:
            name, confidence = self.recognize_face(face.embedding)
            
            face_info = {
                'name': name,
                'confidence': confidence,
                'bbox': face.bbox.astype(int).tolist(),
                'embedding': face.embedding
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