"""Tracking Status Endpoints

Endpoints para verificar el estado del sistema de tracking de personas.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from surveillance.enhanced_frame_processing_service import enhanced_frame_processing_service
from auth.dependencies import get_current_active_user
from models.auth import UserResponse

router = APIRouter(prefix="/tracking", tags=["tracking"])

@router.get("/status")
def get_tracking_status(current_user: UserResponse = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Obtiene el estado actual del sistema de tracking de personas"""
    try:
        status = enhanced_frame_processing_service.get_service_status()
        
        # Agregar información específica sobre el tracking
        tracking_info = {
            "tracking_enabled": True,
            "tracker_type": enhanced_frame_processing_service.tracker_type,
            "max_distance_threshold": enhanced_frame_processing_service.max_distance_threshold,
            "max_disappeared_frames": enhanced_frame_processing_service.max_disappeared_frames,
            "identity_persistence_frames": enhanced_frame_processing_service.identity_persistence_frames,
            "use_opencv_trackers": enhanced_frame_processing_service.use_opencv_trackers,
            "min_face_size": enhanced_frame_processing_service.min_face_size
        }
        
        # Combinar con el estado del servicio
        status["tracking_configuration"] = tracking_info
        
        return {
            "success": True,
            "message": "Estado del tracking obtenido exitosamente",
            "data": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado del tracking: {str(e)}")

@router.get("/cameras/{camera_id}/persons")
def get_tracked_persons(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Obtiene las personas actualmente trackeadas para una cámara específica"""
    try:
        tracked_persons = enhanced_frame_processing_service.tracked_persons.get(camera_id, {})
        
        persons_info = []
        for person_id, person in tracked_persons.items():
            person_info = {
                "person_id": person.person_id,
                "name": person.name,
                "confidence": person.confidence,
                "identity_confirmed": person.identity_confirmed,
                "centroid": person.centroid,
                "bbox": person.bbox,
                "total_frames_tracked": person.total_frames_tracked,
                "frames_without_face": person.frames_without_face,
                "tracker_initialized": person.tracker_initialized,
                "last_seen": person.last_seen,
                "should_maintain_identity": person.should_maintain_identity(enhanced_frame_processing_service.identity_persistence_frames)
            }
            persons_info.append(person_info)
        
        return {
            "success": True,
            "message": f"Personas trackeadas para {camera_id} obtenidas exitosamente",
            "data": {
                "camera_id": camera_id,
                "tracked_persons_count": len(persons_info),
                "tracked_persons": persons_info,
                "next_person_id": enhanced_frame_processing_service.next_person_id.get(camera_id, 1)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo personas trackeadas: {str(e)}")

@router.post("/cameras/{camera_id}/reset")
def reset_tracking(camera_id: str, current_user: UserResponse = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Reinicia el tracking para una cámara específica"""
    try:
        # Limpiar personas trackeadas
        if camera_id in enhanced_frame_processing_service.tracked_persons:
            # Limpiar trackers de OpenCV
            for person in enhanced_frame_processing_service.tracked_persons[camera_id].values():
                if person.tracker is not None:
                    del person.tracker
            
            enhanced_frame_processing_service.tracked_persons[camera_id] = {}
            enhanced_frame_processing_service.next_person_id[camera_id] = 1
        
        return {
            "success": True,
            "message": f"Tracking reiniciado exitosamente para {camera_id}",
            "data": {
                "camera_id": camera_id,
                "tracked_persons_count": 0,
                "next_person_id": 1
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reiniciando tracking: {str(e)}")

@router.get("/configuration")
def get_tracking_configuration(current_user: UserResponse = Depends(get_current_active_user)) -> Dict[str, Any]:
    """Obtiene la configuración actual del sistema de tracking"""
    try:
        config = {
            "tracker_type": enhanced_frame_processing_service.tracker_type,
            "use_opencv_trackers": enhanced_frame_processing_service.use_opencv_trackers,
            "max_distance_threshold": enhanced_frame_processing_service.max_distance_threshold,
            "max_disappeared_frames": enhanced_frame_processing_service.max_disappeared_frames,
            "identity_persistence_frames": enhanced_frame_processing_service.identity_persistence_frames,
            "min_face_size": enhanced_frame_processing_service.min_face_size,
            "cache_duration": enhanced_frame_processing_service.cache_duration,
            "max_queue_size": enhanced_frame_processing_service.max_queue_size
        }
        
        return {
            "success": True,
            "message": "Configuración del tracking obtenida exitosamente",
            "data": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")