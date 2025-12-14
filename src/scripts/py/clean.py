# 1. Configura tu API Key
# Reemplaza 'TU_API_KEY' con tu clave real
from annotator import ApiKeyProvider
from google import genai

client = genai.Client(api_key=ApiKeyProvider.api_key(id=1))
files = client.files.list()

for file in files:
    if file.name is not None:
        client.files.delete(name=file.name)
