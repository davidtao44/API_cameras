from fastapi import APIRouter, HTTPException, status, Depends, Request
from typing import List, Optional
from pydantic import BaseModel
from services.service_token_service import service_token_service
from auth.dependencies import get_admin_user
from auth.service_dependencies import require_tls_service_token, admin_service_token
from models.auth import UserResponse

router = APIRouter(prefix="/service-auth", tags=["service-authentication"])

# Modelos para service tokens
class ServiceTokenCreate(BaseModel):
    service_name: str
    permissions: List[str] = []
    description: Optional[str] = None

class ServiceTokenResponse(BaseModel):
    token: str
    token_id: str
    service_name: str
    permissions: List[str]
    expires_at: str

class ServiceTokenInfo(BaseModel):
    token_id: str
    service_name: str
    permissions: List[str]
    created_at: str
    expires_at: str
    revoked: bool

@router.post("/create-token", response_model=ServiceTokenResponse)
async def create_service_token(
    token_data: ServiceTokenCreate,
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Crea un nuevo service token (solo administradores)"""
    try:
        result = service_token_service.generate_service_token(
            service_name=token_data.service_name,
            permissions=token_data.permissions
        )
        
        return ServiceTokenResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating service token"
        )

@router.post("/verify-token")
async def verify_service_token(
    request: Request,
    token_data: dict = Depends(require_tls_service_token)
):
    """Verifica un service token (requiere TLS)"""
    return {
        "valid": True,
        "service_name": token_data["service_name"],
        "permissions": token_data["permissions"],
        "token_id": token_data["token_id"]
    }

@router.post("/revoke-token/{token_id}")
async def revoke_service_token(
    token_id: str,
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Revoca un service token (solo administradores)"""
    success = service_token_service.revoke_service_token(token_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found or already revoked"
        )
    
    return {"message": "Token revoked successfully"}

@router.get("/list-tokens", response_model=List[ServiceTokenInfo])
async def list_service_tokens(
    service_name: Optional[str] = None,
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Lista service tokens activos (solo administradores)"""
    tokens = service_token_service.list_service_tokens(service_name)
    return tokens

@router.get("/permissions")
async def get_available_permissions(
    current_admin: UserResponse = Depends(get_admin_user)
):
    """Obtiene la lista de permisos disponibles para service tokens"""
    return {
        "permissions": [
            "camera:access",
            "camera:control",
            "surveillance:access",
            "surveillance:control",
            "relay:access",
            "relay:control",
            "stats:read",
            "config:read",
            "config:write",
            "admin:*",
            "*"  # Todos los permisos
        ],
        "description": {
            "camera:access": "Acceso de lectura a endpoints de cámaras",
            "camera:control": "Control total de cámaras",
            "surveillance:access": "Acceso a sistema de vigilancia",
            "surveillance:control": "Control del sistema de vigilancia",
            "relay:access": "Acceso a endpoints de relés",
            "relay:control": "Control de relés",
            "stats:read": "Lectura de estadísticas",
            "config:read": "Lectura de configuración",
            "config:write": "Escritura de configuración",
            "admin:*": "Todos los permisos administrativos",
            "*": "Acceso completo a todos los endpoints"
        }
    }

# Endpoint de ejemplo protegido con service token
@router.get("/protected-example")
async def protected_service_endpoint(
    request: Request,
    token_data: dict = Depends(admin_service_token)
):
    """Ejemplo de endpoint protegido con service token"""
    return {
        "message": "Access granted with service token",
        "service_name": token_data["service_name"],
        "permissions": token_data["permissions"],
        "tls_enabled": request.url.scheme == "https"
    }