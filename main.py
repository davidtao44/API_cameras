from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
from config.firebase import initialize_firebase
from endpoints import camera, stats, access, relay, access_control
from services.firebase_service import monitor_firebase_visitors
from services.surveillance_service import surveillance_service

# Importaciones para el servidor webhook HTTP
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
import urllib.parse
import sys

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(camera.router)
app.include_router(stats.router)
app.include_router(access.router)
app.include_router(relay.router)
app.include_router(access_control.router)

# Configurar logging para webhook con codificación UTF-8
webhook_logger = logging.getLogger('webhook')
webhook_logger.setLevel(logging.INFO)
webhook_handler = logging.FileHandler('webhook_alarmas.log', encoding='utf-8')
webhook_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
webhook_logger.addHandler(webhook_handler)

# Configurar la salida estándar para UTF-8 en Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class WebhookHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Manejar peticiones GET"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        # Log de la petición GET
        webhook_logger.info(f"GET recibido desde {self.client_address[0]}")
        webhook_logger.info(f"Path: {self.path}")
        webhook_logger.info(f"Headers: {dict(self.headers)}")
        
        # Respuesta HTML básica
        response = """
        <html>
        <head>
            <title>Webhook Alarmas TP-Link</title>
            <meta charset="utf-8">
        </head>
        <body>
        <h1>Servidor Webhook Activo</h1>
        <p>Timestamp: {}</p>
        <p>Listo para recibir alertas de alarmas</p>
        </body>
        </html>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        self.wfile.write(response.encode('utf-8'))
    
    def do_POST(self):
        """Manejar peticiones POST de alarmas"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        # Log detallado de la alarma recibida
        webhook_logger.info("=" * 50)
        webhook_logger.info(f"ALARMA RECIBIDA desde {self.client_address[0]}")
        webhook_logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        webhook_logger.info(f"Path: {self.path}")
        webhook_logger.info(f"Headers: {dict(self.headers)}")
        
        # Procesar datos recibidos
        try:
            if self.headers.get('Content-Type', '').startswith('application/json'):
                # Datos JSON
                data = json.loads(post_data.decode('utf-8'))
                webhook_logger.info(f"Datos JSON: {json.dumps(data, indent=2, ensure_ascii=False)}")
                self.process_json_alarm(data)
            else:
                # Datos form-encoded o texto plano
                data_str = post_data.decode('utf-8')
                webhook_logger.info(f"Datos recibidos: {data_str}")
                self.process_form_alarm(data_str)
                
        except Exception as e:
            webhook_logger.error(f"Error procesando datos: {e}")
            webhook_logger.info(f"Datos raw: {post_data}")
        
        # Responder OK
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write('{"status": "ok", "message": "Alarma recibida"}'.encode('utf-8'))
    
    def process_json_alarm(self, data):
        """Procesar alarma en formato JSON"""
        # Extraer información de la estructura TP-Link
        device_name = data.get('device_name', 'unknown')
        ip = data.get('ip', 'unknown')
        mac = data.get('mac', 'unknown')
        
        # Procesar eventos
        event_list = data.get('event_list', [])
        
        webhook_logger.info(f"Dispositivo: {device_name}")
        webhook_logger.info(f"IP: {ip}")
        webhook_logger.info(f"MAC: {mac}")
        
        for event in event_list:
            date_time = event.get('dateTime', '')
            event_types = event.get('event_type', [])
            
            # Formatear fecha
            if date_time:
                try:
                    # Convertir formato 20250704115036 a fecha legible
                    dt = datetime.strptime(date_time, '%Y%m%d%H%M%S')
                    formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    webhook_logger.info(f"Fecha del evento: {formatted_date}")
                except:
                    webhook_logger.info(f"Fecha del evento: {date_time}")
            
            for event_type in event_types:
                webhook_logger.info(f"Tipo de evento: {event_type}")
                self.handle_alarm_action(event_type, device_name, ip)
    
    def process_form_alarm(self, data_str):
        """Procesar alarma en formato form-encoded"""
        try:
            # Intentar parsear como form data
            parsed_data = urllib.parse.parse_qs(data_str)
            webhook_logger.info(f"Datos parseados: {parsed_data}")
            
            # Extraer información común
            for key, value in parsed_data.items():
                webhook_logger.info(f"{key}: {value[0] if isinstance(value, list) else value}")
                
        except Exception as e:
            webhook_logger.error(f"Error parseando form data: {e}")
    
    def handle_alarm_action(self, alarm_type, device_name, ip):
        """Manejar acciones específicas según tipo de alarma"""
        if alarm_type == 'MOTION' or 'motion' in alarm_type.lower():
            webhook_logger.info("[ALERTA] MOVIMIENTO DETECTADO")
            webhook_logger.info(f"Dispositivo: {device_name} ({ip})")

            print(f"🔔 ALERTA DE MOVIMIENTO DETECTADO EN LA CAMARA: {device_name}")
            
        elif alarm_type == 'INTRUSION' or 'intrusion' in alarm_type.lower():
            webhook_logger.info("[CRITICO] ALERTA DE INTRUSION")
            webhook_logger.info(f"Dispositivo: {device_name} ({ip})")
            print(f"🔔 ALERTA DE INTRUSION DETECTADA EN LA CAMARA: {device_name}")
            
        elif 'offline' in alarm_type.lower():
            webhook_logger.info("[WARNING] DISPOSITIVO DESCONECTADO")
            webhook_logger.info(f"Dispositivo: {device_name} ({ip})")
            
        else:
            webhook_logger.info(f"[INFO] Evento: {alarm_type}")
            webhook_logger.info(f"Dispositivo: {device_name} ({ip})")
    
    def log_message(self, format, *args):
        """Sobrescribir para evitar logs automáticos del servidor"""
        pass

def run_webhook_server(host='0.0.0.0', port=5001):
    """Ejecutar el servidor webhook HTTP"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, WebhookHandler)
    
    webhook_logger.info(f"Servidor webhook iniciado en http://{host}:{port}")
    webhook_logger.info("Servidor webhook ejecutándose en paralelo con FastAPI")
    
    try:
        httpd.serve_forever()
    except Exception as e:
        webhook_logger.error(f"Error en servidor webhook: {e}")

# Inicializar Firebase
firebase_initialized = initialize_firebase()

@app.get("/")
def root():
    return {
        "message": "API cámaras funcionando!",
        "webhook_status": "Servidor HTTP webhook activo en puerto 5001"
    }

if __name__ == "__main__":
    # Iniciar el hilo de monitoreo de Firebase
    firebase_monitor_thread = threading.Thread(
        target=monitor_firebase_visitors, 
        daemon=True,
        kwargs={"firebase_initialized": firebase_initialized}
    )
    firebase_monitor_thread.start()
    
    surveillance_thread = threading.Thread(
        target=lambda: surveillance_service.start_monitoring("cam1"),
        daemon=True,
        name="surveillance_cam1"
    )
    surveillance_thread.start()
    print("🎥 Iniciando surveillance automático para cam1")

    surveillance_thread = threading.Thread(
        target=lambda: surveillance_service.start_monitoring("cam2"),
        daemon=True,
        name="surveillance_cam2"
    )
    surveillance_thread.start()
    print("🎥 Iniciando surveillance automático para cam2")
    
    # Iniciar el servidor webhook HTTP en un hilo separado
    webhook_thread = threading.Thread(
        target=run_webhook_server,
        kwargs={"host": "0.0.0.0", "port": 5002},  # Cambiar a puerto 5002
        daemon=True,
        name="webhook_server"
    )
    webhook_thread.start()
    print("📡 Servidor webhook HTTP iniciado en puerto 5001")
    
    # Iniciar el servidor FastAPI
    print("🚀 Iniciando servidor FastAPI en puerto 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)