import os
import sys

import dotenv
from google import genai

dotenv.load_dotenv()
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY1")

client = genai.Client(api_key=GEMINI_API_KEY)
path: str = "varias-personas.mp4"
file = client.files.upload(file=path)

while file.state is not None and file.state.name == "PROCESSING":
    if file.name:
        file = client.files.get(name=file.name)

if file.state is not None and file.state.name == "FAILED":
    print("Error fallo en la desacarga del video", file=sys.stderr)

prompt = """
Actúa como un experto en análisis de video, accesibilidad y lingüística de la lengua de signos. Tu tarea es analizar el video proporcionado para generar una transcripción sincronizada, diarizada y localizada espacialmente.

Instrucciones de Análisis:

1. Fuente del Texto: Extrae el texto de los subtítulos incrustados. Si no hay, transcribe el audio.
2. Identificación (Diarización):
   - Asigna un ID (0, 1, 2...) a cada persona distinta. Conserva el ID si la persona reaparece.
   - CASO MULTITUD: Si en un momento hay muchas personas signando el mismo mensaje al unísono (coro), asigna el signer_id: -1.
3. Localización y Descripción:
   - Identifica visualmente al signante ACTIVO (ignora a quienes no mueven las manos).
   - Determina su posición relativa en la pantalla dividiendo el ancho en tres tercios (Izquierda, Centro, Derecha).

Formato de Salida:

Tu respuesta debe ser ESTRICTAMENTE un objeto JSON válido. Usa el siguiente esquema:

{
  "num_signers": [Número total de signantes únicos (sin contar el -1)],
  "annotations": [
    {
      "text": "Texto del subtítulo o audio",
      "start": [Float, segundos],
      "duration": [Float, segundos],
      "signer_id": [Entero],
      "signer_description": "Descripción visual breve (ej: 'Mujer pelo corto', 'Hombre gafas')",
      "screen_position": "Valor estricto de esta lista: ['left', 'center', 'right', 'multiple']"
    }
  ]
}

Reglas para 'screen_position':
- "left": El signante está en el tercio izquierdo de la imagen.
- "center": El signante está centrado.
- "right": El signante está en el tercio derecho.
- "multiple": Úsalo solo cuando signer_id sea -1 (varias personas en distintas posiciones).

Reglas para ausencia de texto:
    En caso de que no haya subtitulos inscritos en ningún instante del video el formato de salida sera un
    diccionario vacio con todas las claves-valor con null.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[file, prompt],
    config={"response_mime_type": "application/json", "temperature": 0.1},
)

print(response.text)

if file.name:
    client.files.delete(name=file.name)

client.close()
