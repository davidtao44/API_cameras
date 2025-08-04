import logging
import base64
from typing import Optional
from fastapi import HTTPException, status, Response
from services.firebase_service import get_firebase_db_service

logger = logging.getLogger(__name__)

def convert_base64_to_image_url(visitor_id: str) -> str:
    """Generar URL para obtener imagen base64 dinámicamente"""
    return f"/images/{visitor_id}"

def convert_base64_to_jpg_bytes(base64_string: str) -> Optional[bytes]:
    """Convertir imagen base64 a bytes JPG sin guardar archivo - TEMPORALMENTE DESHABILITADO"""
    # Función temporalmente deshabilitada debido a dependencia PIL
    logger.warning("Función convert_base64_to_jpg_bytes deshabilitada - PIL no disponible")
    return None

class ImageService:
    def __init__(self):
        self.db = get_firebase_db_service().get_db()
    
    def get_visitor_image(self, visitor_id: str) -> Response:
        """Obtener imagen de visitante como archivo de imagen"""
        try:
            # Obtener datos del visitante desde Firebase
            doc_ref = self.db.collection('visitors').document(visitor_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Visitante no encontrado"
                )
            
            visitor_data = doc.to_dict()
            face_image_base64 = visitor_data.get('faceImage', '')
            
            if not face_image_base64:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Imagen no encontrada para este visitante"
                )
            
            # Limpiar el string base64 (remover prefijo si existe)
            if face_image_base64.startswith('data:image'):
                # Extraer el tipo de imagen y los datos
                header, base64_data = face_image_base64.split(',', 1)
                # Determinar el tipo de contenido desde el header
                if 'jpeg' in header or 'jpg' in header:
                    media_type = "image/jpeg"
                elif 'png' in header:
                    media_type = "image/png"
                elif 'gif' in header:
                    media_type = "image/gif"
                elif 'webp' in header:
                    media_type = "image/webp"
                else:
                    media_type = "image/jpeg"  # Por defecto
            else:
                base64_data = face_image_base64
                media_type = "image/jpeg"  # Por defecto
            
            # Decodificar base64 a bytes
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                logger.error(f"Error al decodificar base64: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error al procesar la imagen"
                )
            
            # Retornar imagen como respuesta HTTP
            return Response(
                content=image_bytes,
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=3600",  # Cache por 1 hora
                    "Content-Disposition": f"inline; filename={visitor_id}.jpg"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al obtener imagen del visitante {visitor_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error interno del servidor: {str(e)}"
            )

# Instancia global del servicio (se inicializa bajo demanda)
image_service = None

def get_image_service():
    global image_service
    if image_service is None:
        image_service = ImageService()
    return image_service