from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import pyseto
from pyseto import Key
from firebase_admin import firestore
from models.auth import UserCreate, UserResponse, TokenData
from config.firebase import db
import os
import json
import secrets
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de PASETO
SECRET_KEY = os.getenv("PASETO_SECRET_KEY", secrets.token_urlsafe(32))
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuración de hash de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuración de claves PASETO
paseto_key = Key.new(version=4, purpose="local", key=SECRET_KEY.encode())

def hash_password(password: str) -> str:
    """Hashea una contraseña usando bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica una contraseña contra su hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Crea un token PASETO de acceso"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire.isoformat()})
    
    # Crear token PASETO v4.local (simétrico)
    token = pyseto.encode(paseto_key, json.dumps(to_encode).encode())
    return token.decode()

def verify_token(token: str) -> Optional[TokenData]:
    """Verifica y decodifica un token PASETO"""
    try:
        # Decodificar token PASETO v4.local
        decoded_token = pyseto.decode(paseto_key, token.encode())
        
        # El objeto Token tiene un atributo payload que contiene los datos
        if hasattr(decoded_token, 'payload'):
            payload_bytes = decoded_token.payload
        else:
            # Si es un string/bytes directamente
            payload_bytes = decoded_token
        
        # Convertir a string si es necesario
        if isinstance(payload_bytes, bytes):
            payload_str = payload_bytes.decode('utf-8')
        else:
            payload_str = str(payload_bytes)
            
        payload = json.loads(payload_str)
        
        # Verificar expiración
        exp_str = payload.get("exp")
        if exp_str:
            exp_time = datetime.fromisoformat(exp_str)
            if datetime.utcnow() > exp_time:
                return None
        
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        
        if email is None:
            return None
            
        token_data = TokenData(email=email, user_id=user_id, role=role)
        return token_data
    except (json.JSONDecodeError, ValueError, AttributeError, Exception) as e:
        print(f"Error verifying token: {e}")
        return None

def create_user(user_data: UserCreate) -> Optional[UserResponse]:
    """Crea un nuevo usuario en Firebase"""
    try:
        # Verificar si el usuario ya existe
        users_ref = db.collection('users')
        existing_user = users_ref.where('email', '==', user_data.email).get()
        
        if existing_user:
            return None  # Usuario ya existe
        
        # Hashear la contraseña
        hashed_password = hash_password(user_data.password)
        
        # Crear el documento del usuario
        user_doc = {
            'email': user_data.email,
            'password_hash': hashed_password,
            'full_name': user_data.full_name,
            'role': user_data.role,
            'is_active': user_data.is_active,
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        
        # Guardar en Firebase
        doc_ref = users_ref.add(user_doc)
        user_id = doc_ref[1].id
        
        return UserResponse(
            id=user_id,
            email=user_data.email,
            full_name=user_data.full_name,
            role=user_data.role,
            is_active=user_data.is_active,
            created_at=user_doc['created_at']
        )
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def authenticate_user(email: str, password: str) -> Optional[UserResponse]:
    """Autentica un usuario verificando email y contraseña"""
    try:
        users_ref = db.collection('users')
        user_docs = users_ref.where('email', '==', email).get()
        
        if not user_docs:
            return None
        
        user_doc = user_docs[0]
        user_data = user_doc.to_dict()
        
        # Verificar contraseña
        if not verify_password(password, user_data['password_hash']):
            return None
        
        # Actualizar último login
        user_doc.reference.update({'last_login': datetime.utcnow()})
        
        return UserResponse(
            id=user_doc.id,
            email=user_data['email'],
            full_name=user_data['full_name'],
            role=user_data['role'],
            is_active=user_data['is_active'],
            created_at=user_data['created_at'],
            last_login=datetime.utcnow()
        )
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return None

def get_user_by_id(user_id: str) -> Optional[UserResponse]:
    """Obtiene un usuario por su ID"""
    try:
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            return None
        
        user_data = user_doc.to_dict()
        return UserResponse(
            id=user_doc.id,
            email=user_data['email'],
            full_name=user_data['full_name'],
            role=user_data['role'],
            is_active=user_data['is_active'],
            created_at=user_data['created_at'],
            last_login=user_data.get('last_login')
        )
    except Exception as e:
        print(f"Error getting user by ID: {e}")
        return None

def login_user(email: str, password: str) -> Optional[dict]:
    """Inicia sesión de un usuario y retorna token de acceso"""
    user = authenticate_user(email, password)
    if not user:
        return None
    
    if not user.is_active:
        return None
    
    # Crear token de acceso
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "role": user.role
        },
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": user
    }