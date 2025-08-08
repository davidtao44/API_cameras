from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from config.camera_config import CAMERAS
from auth.dependencies import get_current_active_user, get_admin_user
from models.auth import UserResponse
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service
import json
import os

router = APIRouter(prefix="/config", tags=["config"])

class CameraConfigUpdate(BaseModel):
    access_control_enabled: Optional[bool] = None
    alert_control_enabled: Optional[bool] = None
    whatsapp_notifications_enabled: Optional[bool] = None

class CameraConfigResponse(BaseModel):
    camera_id: str
    access_control_enabled: bool
    alert_control_enabled: bool
    whatsapp_notifications_enabled: bool

@router.get("/cameras")
def get_all_cameras_config(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener la configuración de todas las cámaras"""
    cameras_config = []
    for camera_id, config in CAMERAS.items():
        cameras_config.append({
            "camera_id": camera_id,
            "access_control_enabled": config.get("access_control_enabled", False),
            "alert_control_enabled": config.get("alert_control_enabled", False),
            "whatsapp_notifications_enabled": config.get("whatsapp_notifications_enabled", False)
        })
    return {"cameras": cameras_config}

@router.get("/cameras/{camera_id}")
def get_camera_config(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener la configuración de una cámara específica"""
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    config = CAMERAS[camera_id]
    return CameraConfigResponse(
        camera_id=camera_id,
        access_control_enabled=config.get("access_control_enabled", False),
        alert_control_enabled=config.get("alert_control_enabled", False),
        whatsapp_notifications_enabled=config.get("whatsapp_notifications_enabled", False)
    )

@router.put("/cameras/{camera_id}")
def update_camera_config(camera_id: str, config_update: CameraConfigUpdate, current_admin: UserResponse = Depends(get_admin_user)):
    """Actualizar la configuración de una cámara específica"""
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    # Actualizar solo los campos proporcionados
    updated_fields = []
    if config_update.access_control_enabled is not None:
        CAMERAS[camera_id]["access_control_enabled"] = config_update.access_control_enabled
        updated_fields.append("access_control_enabled")
    
    if config_update.alert_control_enabled is not None:
        CAMERAS[camera_id]["alert_control_enabled"] = config_update.alert_control_enabled
        updated_fields.append("alert_control_enabled")
    
    if config_update.whatsapp_notifications_enabled is not None:
        CAMERAS[camera_id]["whatsapp_notifications_enabled"] = config_update.whatsapp_notifications_enabled
        updated_fields.append("whatsapp_notifications_enabled")
    
    if not updated_fields:
        raise HTTPException(status_code=400, detail="No se proporcionaron campos para actualizar")
    
    # Guardar cambios en el archivo de configuración
    try:
        save_config_to_file()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando configuración: {str(e)}")
    
    # Retornar la configuración actualizada
    config = CAMERAS[camera_id]
    return {
        "message": f"Configuración actualizada para {camera_id}",
        "updated_fields": updated_fields,
        "current_config": {
            "camera_id": camera_id,
            "access_control_enabled": config.get("access_control_enabled", False),
            "alert_control_enabled": config.get("alert_control_enabled", False),
            "whatsapp_notifications_enabled": config.get("whatsapp_notifications_enabled", False)
        }
    }

def save_config_to_file():
    """Guardar la configuración actualizada en el archivo settings.py"""
    settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.py")
    
    # Leer el contenido actual del archivo
    with open(settings_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Generar el nuevo contenido de CAMERAS con comentarios
    cameras_str = "CAMERAS = {\n"
    for camera_id, config in CAMERAS.items():
        cameras_str += f'    "{camera_id}": {{\n'
        for key, value in config.items():
            comment = ""
            if key == "access_control_enabled":
                comment = "  # Control de acceso y envio de mensajes por http"
            elif key == "alert_control_enabled":
                comment = "  # Control de alarma y bloqueo de peticiones al relé"
            elif key == "whatsapp_notifications_enabled":
                comment = "  # Envío de alertas por WhatsApp"
            elif key == "notification_phones":
                comment = "  # Números específicos por cámara"
            elif key == "roi" and isinstance(value, dict):
                cameras_str += f'        "{key}": {{\n'
                for sub_key, sub_value in value.items():
                    sub_comment = ""
                    if sub_key == "enabled":
                        sub_comment = "        # ROI habilitada/deshabilitada"
                    elif sub_key == "x":
                        sub_comment = "               # Coordenada X de inicio (0.0-1.0 para porcentajes)"
                    elif sub_key == "y":
                        sub_comment = "               # Coordenada Y de inicio (0.0-1.0 para porcentajes)"
                    elif sub_key == "width":
                        sub_comment = "           # Ancho de la ROI (0.0-1.0 para porcentajes)"
                    elif sub_key == "height":
                        sub_comment = "          # Alto de la ROI (0.0-1.0 para porcentajes)"
                    elif sub_key == "use_percentage":
                        sub_comment = " # Usar porcentajes (True) o píxeles (False)"
                    elif sub_key == "rotation":
                        sub_comment = "          # Rotación en grados (0, 90, 180, 270)"
                    elif sub_key == "shape":
                        sub_comment = "   # Forma: \"rectangle\", \"circle\", \"polygon\""
                    elif sub_key == "polygon_points":
                        if sub_value:
                            cameras_str += f'            "{sub_key}": {sub_value},   # Puntos para ROI poligonal [(x1,y1), (x2,y2), ...]\n'
                            continue
                        else:
                            sub_comment = ""
                    elif sub_key == "adaptive":
                        sub_comment = "      # ROI adaptativa basada en detecciones previas"
                    elif sub_key == "margin":
                        sub_comment = "          # Margen adicional para ROI adaptativa"
                    
                    if isinstance(sub_value, str):
                        cameras_str += f'            "{sub_key}": "{sub_value}",{sub_comment}\n'
                    elif isinstance(sub_value, bool):
                        cameras_str += f'            "{sub_key}": {sub_value},{sub_comment}\n'
                    elif isinstance(sub_value, (int, float)):
                        cameras_str += f'            "{sub_key}": {sub_value},{sub_comment}\n'
                    elif isinstance(sub_value, list) and sub_key != "polygon_points":
                        cameras_str += f'            "{sub_key}": {sub_value},{sub_comment}\n'
                cameras_str += '        },\n'
                continue
            
            if isinstance(value, str):
                cameras_str += f'        "{key}": "{value}",{comment}\n'
            elif isinstance(value, bool):
                cameras_str += f'        "{key}": {value},{comment}\n'
            elif isinstance(value, list):
                cameras_str += f'        "{key}": {value},{comment}\n'
            else:
                cameras_str += f'        "{key}": {value},{comment}\n'
        cameras_str += '    },\n'
    cameras_str += '}\n'
    
    # Reemplazar la definición de CAMERAS en el contenido
    import re
    pattern = r'CAMERAS = \{.*?\n\}'
    new_content = re.sub(pattern, cameras_str.rstrip(), content, flags=re.DOTALL)
    
    # Escribir el archivo actualizado
    with open(settings_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

# Face Detection Configuration Models
class FaceDetectionConfigUpdate(BaseModel):
    min_face_size: int

class FaceDetectionConfigResponse(BaseModel):
    min_face_size: int
    description: str

# Tracking Distance Configuration Models
class TrackingDistanceConfigUpdate(BaseModel):
    max_distance_threshold: int

class TrackingDistanceConfigResponse(BaseModel):
    max_distance_threshold: int
    description: str

@router.get("/face-detection")
def get_face_detection_config(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener la configuración actual de detección de rostros"""
    min_face_size = enhanced_frame_processing_service.get_min_face_size()
    return FaceDetectionConfigResponse(
        min_face_size=min_face_size,
        description=f"Tamaño mínimo de rostro: {min_face_size}x{min_face_size} píxeles. Los rostros más pequeños serán ignorados para evitar falsos positivos."
    )

@router.put("/face-detection")
def update_face_detection_config(
    config_update: FaceDetectionConfigUpdate, 
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Actualizar la configuración de detección de rostros (solo administradores)"""
    try:
        # Validar el rango del tamaño mínimo
        if config_update.min_face_size < 10 or config_update.min_face_size > 200:
            raise HTTPException(
                status_code=400, 
                detail="El tamaño mínimo de rostro debe estar entre 10 y 200 píxeles"
            )
        
        # Actualizar la configuración
        enhanced_frame_processing_service.set_min_face_size(config_update.min_face_size)
        
        return {
            "message": "Configuración de detección de rostros actualizada exitosamente",
            "min_face_size": config_update.min_face_size,
            "description": f"Los rostros menores a {config_update.min_face_size}x{config_update.min_face_size} píxeles serán ignorados"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar configuración: {str(e)}")

# Modelos para configuración específica por cámara
class CameraDetectionConfigUpdate(BaseModel):
    min_face_size: Optional[int] = None
    max_distance_threshold: Optional[int] = None
    person_confidence: Optional[float] = None
    face_confidence: Optional[float] = None

class CameraDetectionConfigResponse(BaseModel):
    camera_id: str
    min_face_size: int
    max_distance_threshold: int
    person_confidence: float
    face_confidence: float
    description: str

@router.get("/cameras/{camera_id}/detection")
def get_camera_detection_config(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener la configuración de detección específica de una cámara"""
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    min_face_size = enhanced_frame_processing_service.get_camera_config(camera_id, 'min_face_size', 30)
    max_distance_threshold = enhanced_frame_processing_service.get_camera_config(camera_id, 'max_distance_threshold', 250)
    person_confidence = enhanced_frame_processing_service.get_camera_config(camera_id, 'person_confidence', 0.5)
    face_confidence = enhanced_frame_processing_service.get_camera_config(camera_id, 'face_confidence', 0.6)
    
    return CameraDetectionConfigResponse(
        camera_id=camera_id,
        min_face_size=min_face_size,
        max_distance_threshold=max_distance_threshold,
        person_confidence=person_confidence,
        face_confidence=face_confidence,
        description=f"Configuración de detección para {camera_id}: tamaño mínimo de rostro {min_face_size}px, distancia máxima {max_distance_threshold}px, confianza persona {person_confidence}, confianza rostro {face_confidence}"
    )

@router.put("/cameras/{camera_id}/detection")
def update_camera_detection_config(
    camera_id: str,
    config_update: CameraDetectionConfigUpdate,
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Actualizar la configuración de detección específica de una cámara"""
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    try:
        updated_configs = []
        
        if config_update.min_face_size is not None:
            if not (10 <= config_update.min_face_size <= 200):
                raise HTTPException(status_code=400, detail="min_face_size debe estar entre 10 y 200 píxeles")
            enhanced_frame_processing_service.update_camera_config(camera_id, 'min_face_size', config_update.min_face_size)
            updated_configs.append(f"min_face_size: {config_update.min_face_size}px")
        
        if config_update.max_distance_threshold is not None:
            if not (50 <= config_update.max_distance_threshold <= 1000):
                raise HTTPException(status_code=400, detail="max_distance_threshold debe estar entre 50 y 1000 píxeles")
            enhanced_frame_processing_service.update_camera_config(camera_id, 'max_distance_threshold', config_update.max_distance_threshold)
            updated_configs.append(f"max_distance_threshold: {config_update.max_distance_threshold}px")
        
        if config_update.person_confidence is not None:
            if not (0.1 <= config_update.person_confidence <= 1.0):
                raise HTTPException(status_code=400, detail="person_confidence debe estar entre 0.1 y 1.0")
            enhanced_frame_processing_service.update_camera_config(camera_id, 'person_confidence', config_update.person_confidence)
            updated_configs.append(f"person_confidence: {config_update.person_confidence}")
        
        if config_update.face_confidence is not None:
            if not (0.1 <= config_update.face_confidence <= 1.0):
                raise HTTPException(status_code=400, detail="face_confidence debe estar entre 0.1 y 1.0")
            enhanced_frame_processing_service.update_camera_config(camera_id, 'face_confidence', config_update.face_confidence)
            updated_configs.append(f"face_confidence: {config_update.face_confidence}")
        
        if not updated_configs:
            raise HTTPException(status_code=400, detail="No se proporcionaron configuraciones para actualizar")
        
        return {
            "message": f"Configuración de detección actualizada para {camera_id}",
            "camera_id": camera_id,
            "updated_configs": updated_configs,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar configuración: {str(e)}")

@router.get("/cameras/detection/all")
def get_all_cameras_detection_config(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener la configuración de detección de todas las cámaras"""
    cameras_config = {}
    
    for camera_id in CAMERAS.keys():
        cameras_config[camera_id] = {
            "min_face_size": enhanced_frame_processing_service.get_camera_config(camera_id, 'min_face_size', 30),
            "max_distance_threshold": enhanced_frame_processing_service.get_camera_config(camera_id, 'max_distance_threshold', 250),
            "person_confidence": enhanced_frame_processing_service.get_camera_config(camera_id, 'person_confidence', 0.5),
            "face_confidence": enhanced_frame_processing_service.get_camera_config(camera_id, 'face_confidence', 0.6)
        }
    
    return {
        "cameras_detection_config": cameras_config,
        "description": "Configuraciones de detección específicas por cámara"
    }

@router.get("/tracking-distance")
def get_tracking_distance_config(current_user: UserResponse = Depends(get_current_active_user)):
    """Obtener la configuración actual de distancia máxima para asociación rostro-persona"""
    max_distance = enhanced_frame_processing_service.get_max_distance_threshold()
    return TrackingDistanceConfigResponse(
        max_distance_threshold=max_distance,
        description=f"Distancia máxima: {max_distance} píxeles. Los rostros que estén a mayor distancia del centroide de una persona no se asociarán con ella."
    )

@router.put("/tracking-distance")
def update_tracking_distance_config(
    config_update: TrackingDistanceConfigUpdate, 
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Actualizar la configuración de distancia máxima para asociación rostro-persona (solo administradores)"""
    try:
        # Validar el rango de la distancia máxima
        if config_update.max_distance_threshold < 50 or config_update.max_distance_threshold > 1000:
            raise HTTPException(
                status_code=400, 
                detail="La distancia máxima debe estar entre 50 y 1000 píxeles"
            )
        
        # Actualizar la configuración
        enhanced_frame_processing_service.set_max_distance_threshold(config_update.max_distance_threshold)
        
        return {
            "message": "Configuración de distancia de tracking actualizada exitosamente",
            "max_distance_threshold": config_update.max_distance_threshold,
            "description": f"Los rostros que estén a más de {config_update.max_distance_threshold} píxeles del centroide de una persona no se asociarán con ella"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar configuración: {str(e)}")