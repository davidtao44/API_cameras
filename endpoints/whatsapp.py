"""WhatsApp Video Endpoints

Endpoints for video generation from camera buffers.
"""

from fastapi import APIRouter, HTTPException, Response
from typing import Optional, List
from pydantic import BaseModel
from services.whatsapp_service import whatsapp_service
from config.settings import CAMERAS
import os

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

# Request models
class CustomVideoRequest(BaseModel):
    camera_id: str
    duration: int
    phone_numbers: Optional[List[str]] = None
    message: Optional[str] = None

@router.get("/video/{camera_id}")
def send_video_last_10_seconds(camera_id: str):
    """Create and send video from the last 10 seconds of the specified camera to WhatsApp"""
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        camera_config = CAMERAS[camera_id]
        
        # Check if WhatsApp is enabled for this camera
        if not camera_config.get('whatsapp_notifications_enabled', False):
            raise HTTPException(status_code=400, detail="WhatsApp notifications disabled for this camera")
        
        # Get phone numbers from camera config
        phone_numbers = camera_config.get('notification_phones', [])
        
        if not phone_numbers:
            raise HTTPException(status_code=400, detail="No phone numbers configured for this camera")
        
        # Create video
        video_path = whatsapp_service._create_video_from_buffer(camera_id, 10)
        
        if video_path:
            file_size = os.path.getsize(video_path)
            
            try:
                # Send video via WhatsApp
                success = whatsapp_service.send_whatsapp_video(
                    phone_numbers=phone_numbers,
                    video_path=video_path,
                    message=f"Video de los últimos 10 segundos - Cámara {camera_id}",
                    camera_id=camera_id
                )
                
                if success:
                    return {
                        "success": True,
                        "message": f"Video sent successfully to {len(phone_numbers)} numbers",
                        "camera_id": camera_id,
                        "duration_seconds": 10,
                        "file_size_bytes": file_size,
                        "recipients": phone_numbers
                    }
                else:
                    return {
                        "success": False,
                        "message": "Video created but failed to send via WhatsApp"
                    }
                    
            finally:
                # Clean up video file
                try:
                    os.unlink(video_path)
                except:
                    pass
        else:
            return {
                "success": False,
                "message": "Failed to create video (no frames available or error occurred)"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating and sending video: {str(e)}")

@router.post("/video/custom")
def send_custom_duration_video(request: CustomVideoRequest):
    """Create and send a video with custom duration from the specified camera to WhatsApp"""
    if request.camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if request.duration <= 0 or request.duration > 60:
        raise HTTPException(status_code=400, detail="Duration must be between 1 and 60 seconds")
    
    try:
        camera_config = CAMERAS[request.camera_id]
        
        # Check if WhatsApp is enabled for this camera
        if not camera_config.get('whatsapp_notifications_enabled', False):
            raise HTTPException(status_code=400, detail="WhatsApp notifications disabled for this camera")
        
        # Use provided phone numbers or default from camera config
        phone_numbers = request.phone_numbers or camera_config.get('notification_phones', [])
        
        if not phone_numbers:
            raise HTTPException(status_code=400, detail="No phone numbers provided or configured")
        
        # Create video
        video_path = whatsapp_service._create_video_from_buffer(request.camera_id, request.duration)
        
        if video_path:
            file_size = os.path.getsize(video_path)
            
            try:
                # Prepare message
                message = request.message or f"Video personalizado de {request.duration} segundos - Cámara {request.camera_id}"
                
                # Send video via WhatsApp
                success = whatsapp_service.send_whatsapp_video(
                    phone_numbers=phone_numbers,
                    video_path=video_path,
                    message=message,
                    camera_id=request.camera_id
                )
                
                if success:
                    return {
                        "success": True,
                        "message": f"Video sent successfully to {len(phone_numbers)} numbers",
                        "camera_id": request.camera_id,
                        "duration_seconds": request.duration,
                        "file_size_bytes": file_size,
                        "recipients": phone_numbers
                    }
                else:
                    return {
                        "success": False,
                        "message": "Video created but failed to send via WhatsApp"
                    }
                    
            finally:
                # Clean up video file
                try:
                    os.unlink(video_path)
                except:
                    pass
        else:
            return {
                "success": False,
                "message": "Failed to create video (no frames available or error occurred)"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating and sending video: {str(e)}")