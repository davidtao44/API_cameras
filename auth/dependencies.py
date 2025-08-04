from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from services.auth_service import verify_token, get_user_by_id
from models.auth import UserResponse

# Configuración del esquema de seguridad
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """Obtiene el usuario actual desde el token JWT usando JOSE"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verificar el token usando JOSE
        token_data = verify_token(credentials.credentials)
        if token_data is None or token_data.user_id is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    
    # Obtener el usuario de la base de datos
    user = get_user_by_id(token_data.user_id)
    if user is None:
        raise credentials_exception
    
    return user

def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Obtiene el usuario actual y verifica que esté activo"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def get_admin_user(current_user: UserResponse = Depends(get_current_active_user)) -> UserResponse:
    """Verifica que el usuario actual sea administrador"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[UserResponse]:
    """Obtiene el usuario actual de forma opcional (puede ser None)"""
    if credentials is None:
        return None
    
    try:
        # Verificar el token usando JOSE
        token_data = verify_token(credentials.credentials)
        if token_data is None or token_data.user_id is None:
            return None
        
        # Obtener el usuario de la base de datos
        user = get_user_by_id(token_data.user_id)
        return user
    except Exception:
        return None