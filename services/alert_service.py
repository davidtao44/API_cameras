import logging
from typing import List, Dict, Any
from datetime import datetime
from fastapi import HTTPException, status
from models.alert import AlertLog, AlertResponse
from services.firebase_service import get_firebase_db_service
from services.notification_service import get_notification_service

logger = logging.getLogger(__name__)

class AlertService:
    def __init__(self):
        self.db = get_firebase_db_service().get_db()
    
    def get_all_alerts(self) -> List[AlertResponse]:
        """Obtener todas las alertas de la colección 'alerts'"""
        try:
            alerts_ref = self.db.collection('alerts')
            docs = alerts_ref.stream()
            
            alerts = []
            for doc in docs:
                alert_data = doc.to_dict()
                if alert_data:
                    # Agregar el ID del documento
                    alert_data['id'] = doc.id
                    
                    # Convertir timestamp si es necesario
                    timestamp = alert_data.get('timestamp')
                    if timestamp and hasattr(timestamp, 'timestamp'):
                        # Si es un timestamp de Firestore, convertir a datetime
                        timestamp = datetime.fromtimestamp(timestamp.timestamp())
                    elif isinstance(timestamp, str):
                        # Si es string, intentar parsearlo
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except:
                            timestamp = None
                    
                    # Crear respuesta con los datos disponibles (sin image_base64 para optimizar)
                    alert_response = AlertResponse(
                        id=alert_data.get('id', doc.id),
                        alert_type=alert_data.get('alert_type'),
                        camera_id=alert_data.get('camera_id'),
                        camera_location=alert_data.get('camera_location'),
                        timestamp=timestamp,
                        detected_names=alert_data.get('detected_names'),
                        confidence=alert_data.get('confidence'),
                        description=alert_data.get('description'),
                        image_base64=None  # Excluir imagen para optimizar respuesta
                    )
                    alerts.append(alert_response)
            
            logger.info(f"Se obtuvieron {len(alerts)} alertas")
            return alerts
            
        except Exception as e:
            logger.error(f"Error al obtener alertas: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al obtener alertas: {str(e)}"
            )
    
    def get_alert_by_id(self, alert_id: str) -> Dict[str, Any]:
        """Obtener una alerta específica por ID"""
        try:
            doc_ref = self.db.collection('alerts').document(alert_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Alerta con ID {alert_id} no encontrada"
                )
            
            alert_data = doc.to_dict()
            alert_data['id'] = doc.id
            
            # Convertir timestamp si es necesario
            timestamp = alert_data.get('timestamp')
            if timestamp and hasattr(timestamp, 'timestamp'):
                alert_data['timestamp'] = datetime.fromtimestamp(timestamp.timestamp()).isoformat()
            elif isinstance(timestamp, datetime):
                alert_data['timestamp'] = timestamp.isoformat()
            
            logger.info(f"Alerta {alert_id} obtenida exitosamente")
            return {
                "success": True,
                "data": alert_data,
                "message": "Alerta obtenida exitosamente"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al obtener alerta {alert_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al obtener alerta: {str(e)}"
            )
    
    async def create_alert(self, alert: AlertLog) -> Dict[str, Any]:
        """Crear una nueva alerta"""
        try:
            # Preparar datos para Firebase
            alert_data = alert.dict()
            
            # Convertir datetime a timestamp para Firebase
            if isinstance(alert_data.get('timestamp'), datetime):
                alert_data['timestamp'] = alert_data['timestamp']
            
            # Crear documento en Firebase
            doc_ref = self.db.collection('alerts').document(alert.id)
            doc_ref.set(alert_data)
            
            logger.info(f"Alerta {alert.id} creada exitosamente")
            
            # Enviar notificación en tiempo real
            try:
                notification_service = get_notification_service()
                await notification_service.broadcast_alert({
                    "id": alert.id,
                    **alert_data
                })
                logger.info(f"Notificación enviada para alerta {alert.id}")
            except Exception as notification_error:
                logger.error(f"Error enviando notificación para alerta {alert.id}: {notification_error}")
                # No fallar la creación de la alerta si falla la notificación
            
            return {
                "success": True,
                "data": {
                    "id": alert.id,
                    **alert_data
                },
                "message": "Alerta creada exitosamente"
            }
            
        except Exception as e:
            logger.error(f"Error al crear alerta: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al crear alerta: {str(e)}"
            )
    
    def get_alert_image(self, alert_id: str) -> Dict[str, Any]:
        """Obtener solo la imagen de una alerta específica"""
        try:
            doc_ref = self.db.collection('alerts').document(alert_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Alerta con ID {alert_id} no encontrada"
                )
            
            alert_data = doc.to_dict()
            image_base64 = alert_data.get('image_base64')
            
            if not image_base64:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No hay imagen disponible para la alerta {alert_id}"
                )
            
            logger.info(f"Imagen de alerta {alert_id} obtenida exitosamente")
            return {
                "success": True,
                "data": {
                    "alert_id": alert_id,
                    "image_base64": image_base64
                },
                "message": "Imagen obtenida exitosamente"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al obtener imagen de alerta {alert_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al obtener imagen: {str(e)}"
            )

# Instancia global del servicio
alert_service = None

def get_alert_service():
    global alert_service
    if alert_service is None:
        alert_service = AlertService()
    return alert_service