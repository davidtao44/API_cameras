import ssl
import os
from pathlib import Path
from typing import Optional

# Configuración TLS/SSL
class TLSConfig:
    def __init__(self):
        self.cert_dir = Path("certs")
        self.cert_file = self.cert_dir / "server.crt"
        self.key_file = self.cert_dir / "server.key"
        self.ca_file = self.cert_dir / "ca.crt"
        
    def ensure_cert_directory(self):
        """Crea el directorio de certificados si no existe"""
        self.cert_dir.mkdir(exist_ok=True)
        
    def generate_self_signed_cert(self):
        """Genera certificados auto-firmados para desarrollo"""
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import datetime
            import ipaddress
            
            # Generar clave privada
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Crear certificado
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "ES"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Madrid"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Madrid"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Camera API"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Guardar certificado
            self.ensure_cert_directory()
            
            with open(self.cert_file, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
                
            with open(self.key_file, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
                
            print(f"[INFO] Certificados auto-firmados generados en {self.cert_dir}")
            return True
            
        except ImportError:
            print("[WARNING] cryptography no disponible. No se pueden generar certificados.")
            return False
        except Exception as e:
            print(f"[ERROR] Error generando certificados: {e}")
            return False
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Obtiene el contexto SSL para el servidor"""
        if not (self.cert_file.exists() and self.key_file.exists()):
            print("[WARNING] Certificados no encontrados. Generando certificados auto-firmados...")
            if not self.generate_self_signed_cert():
                return None
                
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(self.cert_file), str(self.key_file))
            
            # Configuraciones de seguridad
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            
            return context
            
        except Exception as e:
            print(f"[ERROR] Error configurando SSL: {e}")
            return None
    
    def is_tls_available(self) -> bool:
        """Verifica si TLS está disponible"""
        return self.get_ssl_context() is not None

# Instancia global
tls_config = TLSConfig()

def get_uvicorn_ssl_config() -> dict:
    """Obtiene la configuración SSL para uvicorn"""
    if tls_config.is_tls_available():
        return {
            "ssl_keyfile": str(tls_config.key_file),
            "ssl_certfile": str(tls_config.cert_file),
            "ssl_version": ssl.PROTOCOL_TLS_SERVER,
            "ssl_cert_reqs": ssl.CERT_NONE,
        }
    return {}

def print_tls_info():
    """Imprime información sobre la configuración TLS"""
    if tls_config.is_tls_available():
        print("[INFO] TLS/HTTPS habilitado")
        print(f"[INFO] Certificado: {tls_config.cert_file}")
        print(f"[INFO] Clave privada: {tls_config.key_file}")
        print("[INFO] Servidor disponible en: https://localhost:8000")
        print("[INFO] Service tokens requieren HTTPS para funcionar")
    else:
        print("[WARNING] TLS/HTTPS no disponible")
        print("[WARNING] Service tokens funcionarán solo en localhost")
        print("[INFO] Servidor disponible en: http://localhost:8000")