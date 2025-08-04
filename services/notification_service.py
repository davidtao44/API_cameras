import asyncio
import json
import logging
from typing import Dict, Set, Any
from datetime import datetime
from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self.clients: Set[asyncio.Queue] = set()
        self.logger = logging.getLogger(__name__)
    
    async def add_client(self, client_queue: asyncio.Queue):
        """Agregar un cliente a la lista de suscriptores"""
        self.clients.add(client_queue)
        self.logger.info(f"Cliente agregado. Total de clientes: {len(self.clients)}")
    
    async def remove_client(self, client_queue: asyncio.Queue):
        """Remover un cliente de la lista de suscriptores"""
        self.clients.discard(client_queue)
        self.logger.info(f"Cliente removido. Total de clientes: {len(self.clients)}")
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Enviar notificación de nueva alerta a todos los clientes conectados"""
        if not self.clients:
            self.logger.info("No hay clientes conectados para enviar la notificación")
            return
        
        # Preparar el mensaje de notificación
        notification = {
            "type": "new_alert",
            "data": alert_data,
            "timestamp": datetime.now().isoformat()
        }
        
        message = f"data: {json.dumps(notification)}\n\n"
        
        # Enviar a todos los clientes conectados
        disconnected_clients = set()
        for client_queue in self.clients.copy():
            try:
                await client_queue.put(message)
                self.logger.debug(f"Notificación enviada a cliente")
            except Exception as e:
                self.logger.error(f"Error enviando notificación a cliente: {e}")
                disconnected_clients.add(client_queue)
        
        # Remover clientes desconectados
        for client in disconnected_clients:
            await self.remove_client(client)
    
    async def create_event_stream(self, request: Request):
        """Crear un stream de eventos para Server-Sent Events"""
        client_queue = asyncio.Queue()
        await self.add_client(client_queue)
        
        try:
            # Enviar mensaje inicial de conexión
            initial_message = {
                "type": "connection",
                "message": "Conectado al sistema de notificaciones",
                "timestamp": datetime.now().isoformat()
            }
            await client_queue.put(f"data: {json.dumps(initial_message)}\n\n")
            
            while True:
                # Verificar si el cliente sigue conectado
                if await request.is_disconnected():
                    break
                
                try:
                    # Esperar por mensajes con timeout
                    message = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                    yield message
                except asyncio.TimeoutError:
                    # Enviar heartbeat cada 30 segundos
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                
        except Exception as e:
            self.logger.error(f"Error en event stream: {e}")
        finally:
            await self.remove_client(client_queue)

# Instancia global del servicio
notification_service = None

def get_notification_service():
    global notification_service
    if notification_service is None:
        notification_service = NotificationService()
    return notification_service