import os

import dotenv


class exceptions:
    class LoadingDotEnvError(Exception):
        def __init__(self) -> None:
            self.message = "couldn't load or parse de .env file"
            super().__init__()


class Config:
    ok = dotenv.load_dotenv()
    if not ok:
        raise exceptions.LoadingDotEnvError

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", default="NO_API_KEY")
    CNSE_VIDEOS_DIR = os.getenv("CNSE_VIDEOS_DIR", default="NO_DIR")
    ANNOTATIONS_DIR = os.getenv("ANNOTATIONS_DIR", default="NO_DIR")

    PROMPT4ANNOTATION = """
    Actúa como un experto en visión artificial, lingüística de la lengua de signos española (LSE) y accesibilidad audiovisual.
    Tu objetivo es generar un dataset de alta calidad alineando texto en español con segmentos de video donde aparece una persona signando activamente.

    TAREA PRINCIPAL:
    Analiza el video y genera una transcripción sincronizada y diarizada, extrayendo el texto incrustado en pantalla (subtítulos/cártelas) SOLO cuando se cumplan las condiciones estrictas de validación.

    --- REGLAS DE ORO (CRITERIOS DE EXCLUSIÓN) ---
    1. NO ANOTAR SI NO HAY SIGNANTE: Si aparece texto en pantalla (títulos, diapositivas, transiciones, créditos) pero NO hay una persona visible haciendo señas, IGNORESE COMPLETAMENTE. El texto solo nos importa si hay un humano traduciéndolo simultáneamente.
    2. NO ANOTAR TEXTO INCOMPLETO (Efecto Fundido/Escritura): Los videos suelen mostrar el texto apareciendo palabra por palabra. NO generes una anotación por cada palabra nueva. Espera a que la frase o bloque de texto se complete y estabilice para crear UNA ÚNICA anotación que abarque todo el intervalo temporal de esa frase.
    3. NO ANOTAR SUBTÍTULOS MIXTOS: Si el texto en pantalla mezcla idiomas (ej: subtítulos pegados en inglés sobre fondo español, o viceversa) y no está claro que sea el target en Español, IGNORAR EL SEGMENTO.
    4. SOLO ESPAÑOL: Ignora cualquier texto que no esté en español.

    --- INSTRUCCIONES DE ANÁLISIS ---

    1. DETECCIÓN DE SIGNANTE (Diarización):
       - Identifica a las personas activas. Asigna IDs consistentes (0, 1...).
       - Si hay múltiples personas signando lo mismo a la vez (coro), usa signer_id: -1.
       - Si la persona está en pantalla pero está totalmente estática (ej: escuchando, o congelada al final del video), NO anotes ese tramo.

    2. EXTRACCIÓN Y SINCRONIZACIÓN DE TEXTO:
       - Extrae el contenido semántico completo de los subtítulos en pantalla.
       - Start/Duration: Deben coincidir con el momento en que el texto es legible Y el signante está ejecutando las señas correspondientes.
       - Prioriza el texto visual. Usa el audio solo para desambiguar si el texto visual está cortado, pero la fuente de verdad es el subtítulo visual en Español.

    3. LOCALIZACIÓN ESPACIAL:
       - "left": Tercio izquierdo.
       - "center": Tercio central.
       - "right": Tercio derecho.
       - "news-corner": Signante en una caja pequeña en una esquina (típico de noticiarios/TV).
       - "multiple": Solo para signer_id: -1.

    Si no se encuentra ningún segmento que cumpla TODAS las condiciones (texto + signante activo), devuelve:
    {{
      "num_signers": 0,
      "annotations": []
    }}

    --- INFORMACIÓN DEL VIDEO ---
    - Duración: {} segundos.
    - Frames totales: {}.
    """

    PROMPT4FIXING = ...
