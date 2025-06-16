from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class AccessLog(BaseModel):
    person_id: str
    name: str
    timestamp: datetime
    event_type: str  # "ENTRY" o "EXIT"
    camera_id: str
    confidence: float

class AccessLogResponse(BaseModel):
    person_id: str
    name: str
    entry_time: Optional[datetime]
    exit_time: Optional[datetime]
    camera_id: str 