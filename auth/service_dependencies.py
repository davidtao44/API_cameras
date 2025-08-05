from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from services.service_token_service import service_token_service
import ssl

# Configuración del esquema de seguridad para service tokens
service_security = HTTPBearer()

def verify_tls_connection(request: Request) -> bool:
    """Verifica que la conexión use TLS/HTTPS"""
    # En desarrollo, permitir HTTP
    if request.url.scheme == "http" and request.client.host in ["127.0.0.1", "localhost"]:
        return True
    
    # En producción, requerir HTTPS
    return request.url.scheme == "https"

def get_service_token_data(credentials: HTTPAuthorizationCredentials = Depends(service_security)) -> Dict[str, Any]:
    """Obtiene y verifica un service token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid service token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verificar el service token
        token_data = service_token_service.verify_service_token(credentials.credentials)
        if token_data is None:
            raise credentials_exception
            
        return token_data
        
    except Exception:
        raise credentials_exception

def require_tls_service_token(request: Request, token_data: Dict[str, Any] = Depends(get_service_token_data)) -> Dict[str, Any]:
    """Requiere TLS y un service token válido"""
    # Verificar TLS
    if not verify_tls_connection(request):
        raise HTTPException(
            status_code=status.HTTP_426_UPGRADE_REQUIRED,
            detail="TLS/HTTPS required for service token authentication"
        )
    
    return token_data

def require_service_permission(required_permission: str):
    """Decorator para requerir un permiso específico en el service token"""
    def permission_checker(token_data: Dict[str, Any] = Depends(require_tls_service_token)) -> Dict[str, Any]:
        permissions = token_data.get("permissions", [])
        
        if required_permission not in permissions and "*" not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service token lacks required permission: {required_permission}"
            )
        
        return token_data
    
    return permission_checker

def require_service_name(required_service: str):
    """Decorator para requerir un nombre de servicio específico"""
    def service_checker(token_data: Dict[str, Any] = Depends(require_tls_service_token)) -> Dict[str, Any]:
        service_name = token_data.get("service_name")
        
        if service_name != required_service:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service token not valid for service: {required_service}"
            )
        
        return token_data
    
    return service_checker

# Dependencias combinadas comunes
def camera_service_token(token_data: Dict[str, Any] = Depends(require_service_permission("camera:access"))) -> Dict[str, Any]:
    """Requiere permiso de acceso a cámaras"""
    return token_data

def admin_service_token(token_data: Dict[str, Any] = Depends(require_service_permission("admin:*"))) -> Dict[str, Any]:
    """Requiere permisos de administrador"""
    return token_data

def surveillance_service_token(token_data: Dict[str, Any] = Depends(require_service_permission("surveillance:access"))) -> Dict[str, Any]:
    """Requiere permiso de acceso a vigilancia"""
    return token_data