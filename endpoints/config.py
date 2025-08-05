from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from config.camera_config import CAMERAS
from auth.dependencies import get_current_active_user, get_admin_user
from models.auth import UserResponse
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