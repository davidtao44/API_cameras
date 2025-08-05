from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import hashlib
import hmac
import json
from config.firebase import db
from firebase_admin import firestore
import os

# Configuración de Service Tokens
SERVICE_TOKEN_SECRET = os.getenv("SERVICE_TOKEN_SECRET", secrets.token_urlsafe(64))
SERVICE_TOKEN_EXPIRE_HOURS = 24

class ServiceTokenService:
    """Servicio para manejar tokens de servicio seguros con TLS"""
    
    def __init__(self):
        self.secret = SERVICE_TOKEN_SECRET.encode()
        
    def generate_service_token(self, service_name: str, permissions: list = None) -> Dict[str, Any]:
        """Genera un token de servicio con permisos específicos"""
        if permissions is None:
            permissions = []
            
        # Crear payload del token
        token_id = secrets.token_urlsafe(16)
        issued_at = datetime.utcnow()
        expires_at = issued_at + timedelta(hours=SERVICE_TOKEN_EXPIRE_HOURS)
        
        payload = {
            "token_id": token_id,
            "service_name": service_name,
            "permissions": permissions,
            "issued_at": issued_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "type": "service_token"
        }
        
        # Crear firma HMAC
        payload_json = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret,
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Token final
        token = f"{payload_json.encode().hex()}.{signature}"
        
        # Guardar en base de datos para revocación
        self._store_service_token(token_id, service_name, permissions, expires_at)
        
        return {
            "token": token,
            "token_id": token_id,
            "service_name": service_name,
            "permissions": permissions,
            "expires_at": expires_at.isoformat()
        }
    
    def verify_service_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token de servicio"""
        try:
            # Separar payload y firma
            parts = token.split('.')
            if len(parts) != 2:
                return None
                
            payload_hex, signature = parts
            
            # Decodificar payload
            payload_json = bytes.fromhex(payload_hex).decode()
            payload = json.loads(payload_json)
            
            # Verificar firma
            expected_signature = hmac.new(
                self.secret,
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Verificar expiración
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.utcnow() > expires_at:
                return None
            
            # Verificar que el token no esté revocado
            if self._is_token_revoked(payload["token_id"]):
                return None
                
            return payload
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return None
    
    def revoke_service_token(self, token_id: str) -> bool:
        """Revoca un token de servicio"""
        try:
            doc_ref = db.collection("service_tokens").document(token_id)
            doc_ref.update({"revoked": True, "revoked_at": firestore.SERVER_TIMESTAMP})
            return True
        except Exception as e:
            return False
    
    def list_service_tokens(self, service_name: str = None) -> list:
        """Lista tokens de servicio activos"""
        try:
            query = db.collection("service_tokens").where("revoked", "==", False)
            
            if service_name:
                query = query.where("service_name", "==", service_name)
            
            docs = query.stream()
            tokens = []
            
            for doc in docs:
                data = doc.to_dict()
                data["token_id"] = doc.id
                tokens.append(data)
                
            return tokens
        except Exception as e:
            return []
    
    def _store_service_token(self, token_id: str, service_name: str, permissions: list, expires_at: datetime):
        """Almacena información del token en la base de datos"""
        try:
            doc_ref = db.collection("service_tokens").document(token_id)
            doc_ref.set({
                "service_name": service_name,
                "permissions": permissions,
                "created_at": firestore.SERVER_TIMESTAMP,
                "expires_at": expires_at,
                "revoked": False
            })
        except Exception as e:
            pass  # Log error in production
    
    def _is_token_revoked(self, token_id: str) -> bool:
        """Verifica si un token está revocado"""
        try:
            doc_ref = db.collection("service_tokens").document(token_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                return data.get("revoked", False)
            
            return True  # Si no existe, considerarlo revocado
        except Exception as e:
            return True  # En caso de error, considerarlo revocado

# Instancia global del servicio
service_token_service = ServiceTokenService()