import logging
from typing import List
from datetime import datetime
from fastapi import HTTPException, status
from models.visitor import VisitorModel, VisitorResponse
from services.firebase_service import get_firebase_db_service

logger = logging.getLogger(__name__)

def convert_base64_to_image_url(visitor_id: str) -> str:
    """Generar URL para obtener imagen base64 dinámicamente"""
    return f"/images/{visitor_id}"

class VisitorService:
    def __init__(self):
        self.db = get_firebase_db_service().get_db()
    
    def get_all_visitors(self) -> List[VisitorResponse]:
        """Obtener todos los registros de visitantes"""
        try:
            visitors_ref = self.db.collection('visitors')
            docs = visitors_ref.stream()
            
            visitors = []
            for doc in docs:
                visitor_data = doc.to_dict()
                visitor_data['id'] = doc.id
                
                # Función auxiliar para convertir fechas
                def convert_datetime_to_string(value):
                    if hasattr(value, 'isoformat'):
                        return value.isoformat()
                    elif isinstance(value, str):
                        return value
                    else:
                        return str(value) if value else ''
                
                # Generar URL para imagen base64 dinámica si existe
                face_image_base64 = visitor_data.get('faceImage', '')
                image_url = None
                if face_image_base64:
                    image_url = convert_base64_to_image_url(doc.id)
                
                # Asegurar que todos los campos requeridos estén presentes
                visitor_response = {
                    'id': doc.id,
                    'nombre': visitor_data.get('nombre', visitor_data.get('firstName', '')),
                    'apellido': visitor_data.get('apellido', visitor_data.get('lastName', '')),
                    'documento': visitor_data.get('documento', visitor_data.get('documentNumber', '')),
                    'telefono': visitor_data.get('telefono', visitor_data.get('phone', '')),
                    'email': visitor_data.get('email', ''),
                    'motivo_visita': visitor_data.get('motivo_visita', visitor_data.get('visitReason', '')),
                    'persona_visitada': visitor_data.get('persona_visitada', visitor_data.get('personToVisit', '')),
                    'fecha_visita': convert_datetime_to_string(visitor_data.get('fecha_visita', visitor_data.get('visitDate', ''))),
                    'estado': visitor_data.get('estado', visitor_data.get('status', 'pendiente')),
                    'fecha_registro': convert_datetime_to_string(visitor_data.get('fecha_registro', visitor_data.get('registrationDate', visitor_data.get('timestamp', '')))),
                    'faceImage': image_url  # Ahora contiene la URL para obtener la imagen base64
                }
                
                visitors.append(VisitorResponse(**visitor_response))
            
            logger.info(f"Obtenidos {len(visitors)} registros de visitantes")
            return visitors
            
        except Exception as e:
            logger.error(f"Error al obtener registros: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error interno del servidor: {str(e)}"
            )
    
    def create_visitor(self, visitor: VisitorModel) -> dict:
        """Crear un nuevo visitante"""
        try:
            visitor_data = visitor.dict()
            visitor_data['fecha_registro'] = datetime.now().isoformat()
            
            doc_ref = self.db.collection('visitors').add(visitor_data)
            visitor_id = doc_ref[1].id
            
            logger.info(f"Visitante creado con ID: {visitor_id}")
            return {
                "success": True,
                "message": "Visitante registrado exitosamente",
                "id": visitor_id,
                "timestamp": visitor_data['fecha_registro']
            }
            
        except Exception as e:
            logger.error(f"Error al crear visitante: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error interno del servidor: {str(e)}"
            )
    
    def update_visitor_status(self, visitor_id: str, nuevo_estado: str) -> dict:
        """Actualizar el estado de un visitante"""
        try:
            doc_ref = self.db.collection('visitors').document(visitor_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Visitante no encontrado"
                )
            
            doc_ref.update({
                'estado': nuevo_estado,
                'fecha_actualizacion': datetime.now().isoformat()
            })
            
            logger.info(f"Estado del visitante {visitor_id} actualizado a: {nuevo_estado}")
            return {
                "success": True,
                "message": f"Estado actualizado a: {nuevo_estado}",
                "visitor_id": visitor_id,
                "nuevo_estado": nuevo_estado,
                "timestamp": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al actualizar estado del visitante {visitor_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error interno del servidor: {str(e)}"
            )
    
    def get_visitor_by_id(self, visitor_id: str) -> dict:
        """Obtener un visitante por ID"""
        try:
            doc_ref = self.db.collection('visitors').document(visitor_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Visitante no encontrado"
                )
            
            return doc.to_dict()
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al obtener visitante {visitor_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error interno del servidor: {str(e)}"
            )

# Instancia global del servicio (se inicializa bajo demanda)
visitor_service = None

def get_visitor_service():
    global visitor_service
    if visitor_service is None:
        visitor_service = VisitorService()
    return visitor_service