from datetime import datetime
from typing import List, Optional
from models.access_log import AccessLog, AccessLogResponse
from config.firebase import db

class AccessService:
    def __init__(self):
        self.collection = db.collection('access_logs')

    async def log_access(self, access_log: AccessLog) -> None:
        """Registra un evento de entrada o salida"""
        log_data = access_log.dict()
        log_data['timestamp'] = log_data['timestamp'].isoformat()
        self.collection.add(log_data)

    async def get_person_logs(self, person_id: str) -> List[AccessLogResponse]:
        """Obtiene todos los registros de una persona"""
        logs = self.collection.where('person_id', '==', person_id).order_by('timestamp').get()
        
        access_records = {}
        for log in logs:
            log_data = log.to_dict()
            if log_data['person_id'] not in access_records:
                access_records[log_data['person_id']] = {
                    'person_id': log_data['person_id'],
                    'name': log_data['name'],
                    'entry_time': None,
                    'exit_time': None,
                    'camera_id': log_data['camera_id']
                }
            
            if log_data['event_type'] == 'ENTRY':
                access_records[log_data['person_id']]['entry_time'] = datetime.fromisoformat(log_data['timestamp'])
            else:
                access_records[log_data['person_id']]['exit_time'] = datetime.fromisoformat(log_data['timestamp'])

        return list(access_records.values())

    async def get_current_people(self) -> List[AccessLogResponse]:
        """Obtiene las personas que están actualmente en el recinto"""
        logs = self.collection.order_by('timestamp').get()
        
        current_people = {}
        for log in logs:
            log_data = log.to_dict()
            if log_data['event_type'] == 'ENTRY':
                current_people[log_data['person_id']] = {
                    'person_id': log_data['person_id'],
                    'name': log_data['name'],
                    'entry_time': datetime.fromisoformat(log_data['timestamp']),
                    'exit_time': None,
                    'camera_id': log_data['camera_id']
                }
            elif log_data['person_id'] in current_people:
                del current_people[log_data['person_id']]

        return list(current_people.values())

access_service = AccessService()