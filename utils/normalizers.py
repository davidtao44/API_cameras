import unicodedata
import re

def normalize_text(text: str) -> str:
    """
    Normaliza un texto eliminando tildes y caracteres especiales.
    Convierte a minúsculas y luego capitaliza cada palabra.
    """
    if not text:
        return ""
    
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.replace('ñ', 'n')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(word.capitalize() for word in text.split())
    
    return text