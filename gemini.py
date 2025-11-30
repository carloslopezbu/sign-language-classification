import os

import dotenv
import videoproc as vp
from google import genai

dotenv.load_dotenv()
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

print(GEMINI_API_KEY)

client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="¿Puedes explicarme en simples términos el algoritmo de retroporpagación para redes neuronales multicapa?",
)


print(response.text)
