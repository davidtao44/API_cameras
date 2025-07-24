from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid

class AlertLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str  # "Acceso", "Intruso"
    camera_id: str
    camera_location: str
    timestamp: datetime
    detected_names: Optional[List[str]] = None  # Para alertas de acceso concedido
    confidence: Optional[float] = None
    description: Optional[str] = None
    image_base64: Optional[str] = None  # Imagen en base64

class AlertLogResponse(BaseModel):
    id: str
    alert_type: str
    camera_id: str
    camera_location: str
    timestamp: datetime
    detected_names: Optional[List[str]] = None
    confidence: Optional[float] = None
    description: Optional[str] = None
    image_base64: Optional[str] = None