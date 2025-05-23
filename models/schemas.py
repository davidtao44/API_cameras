from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional

class RecognitionStats(BaseModel):
    total_detections: int
    recognized: int
    unrecognized: int
    recognition_rate: float
    last_updated: datetime
    recognized_names: List[str]

class FirebaseProcessRequest(BaseModel):
    collection_name: str
    document_id: Optional[str] = None

class ProcessResponse(BaseModel):
    success: bool
    message: str
    processed_names: List[str] = []
    firebase_data: Dict = {}