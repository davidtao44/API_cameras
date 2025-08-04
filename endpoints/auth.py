from fastapi import APIRouter, HTTPException, status, Depends
from models.auth import UserLogin, UserCreate, UserResponse, Token
from services.auth_service import login_user, create_user
from auth.dependencies import get_current_active_user, get_admin_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Endpoint para iniciar sesión de usuarios"""
    result = login_user(user_credentials.email, user_credentials.password)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return Token(
        access_token=result["access_token"],
        token_type=result["token_type"],
        expires_in=result["expires_in"],
        user=result["user"]
    )

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, current_admin: UserResponse = Depends(get_admin_user)):
    """Endpoint para registrar nuevos usuarios (solo administradores)"""
    user = create_user(user_data)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    return user

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para obtener información del usuario actual"""
    return current_user

@router.post("/verify-token")
async def verify_token_endpoint(current_user: UserResponse = Depends(get_current_active_user)):
    """Endpoint para verificar si un token es válido"""
    return {"valid": True, "user_id": current_user.id, "email": current_user.email}