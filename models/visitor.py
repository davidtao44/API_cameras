from pydantic import BaseModel, Field
from typing import Optional

class VisitorModel(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=100)
    apellido: str = Field(..., min_length=1, max_length=100)
    documento: str = Field(..., min_length=1, max_length=20)
    telefono: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=100)
    motivo_visita: str = Field(..., min_length=1, max_length=200)
    persona_visitada: str = Field(..., min_length=1, max_length=100)
    fecha_visita: Optional[str] = None
    estado: str = Field(default="pendiente")

class StatusUpdateModel(BaseModel):
    estado: str = Field(..., pattern="^(pendiente|aprobado|rechazado|completado)$")

class VisitorResponse(BaseModel):
    id: str
    nombre: Optional[str] = None
    apellido: Optional[str] = None
    documento: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    motivo_visita: Optional[str] = None
    persona_visitada: Optional[str] = None
    fecha_visita: Optional[str] = None
    estado: Optional[str] = "pendiente"
    fecha_registro: Optional[str] = None
    faceImage: Optional[str] = None