import json
import os
import time
from enum import StrEnum
from typing import List, Literal

import click
import dotenv
import logger
from google import genai
from pydantic import BaseModel, Field

dotenv.load_dotenv()

METADATA_VIDEOS_FILE = os.getenv("METADATA_VIDEOS_FILE")
VIDEOS_PATH = os.getenv("VIDEOS_PATH")


class Annotation(BaseModel):
    text: str = Field(description="Texto del subtítulo o audio en español consolidado.")
    start: float = Field(description="Tiempo de inicio en segundos.")
    duration: float = Field(description="Duración en segundos.")
    signer_id: int = Field(description="ID del signante (0, 1...) o -1 para coros.")
    aligned: bool = Field(description="True si el movimiento coincide con el texto.")
    signer_description: str = Field(
        description="Descripción visual breve (ej: 'Mujer pelo corto')."
    )
    screen_position: Literal["left", "center", "right", "multiple", "news-corner"] = (
        Field(description="Posición del signante en la pantalla.")
    )


class VideoAnalysis(BaseModel):
    num_signers: int = Field(
        description="Número total de signantes únicos (sin contar -1)."
    )
    annotations: List[Annotation] = Field(
        description="Lista de anotaciones sincronizadas."
    )


class ApiKeyProvider:
    @staticmethod
    def api_key(id: int) -> str | None:
        return os.getenv(f"GEMINI_API_KEY{id}")


class Models(StrEnum):
    Gemini2dot0Flash = "gemini-2.0-flash"
    Gemini2dot5Flash = "gemini-2.5-flash"
    Gemini2dot5Pro = "gemini-2.5-pro"
    Gemini3dot0Pro = "gemini-3-pro"
    Gemini3dot0Flash = "gemini-3-flash"


class Annotator:
    def __init__(self, model: str, api_key: str | None, out_dir: str) -> None:
        self.model = model
        if api_key is None:
            raise RuntimeError("Api Key is None")
        self.client = genai.Client(api_key=api_key)
        self.out_dir = out_dir
        self.failed_dir = os.path.join(out_dir, "failed")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.failed_dir, exist_ok=True)

    def annotate(self, video: str, prompt: str) -> None:
        video_name = os.path.basename(video)
        logger.info(f"Processing video: {video_name}")

        # 1. Subida
        logger.debug("Uploading video to Gemini API...")
        try:
            file = self.client.files.upload(file=video)
            logger.success("Video uploaded successfully")
        except Exception as e:
            logger.error(f"Error uploading the video: {e}")
            return

        # 2. Espera de procesamiento
        logger.debug("Waiting the video to be proccessed...")
        while file.state is not None and file.state.name == "PROCESSING":
            time.sleep(2.0)
            if file.name:
                file = self.client.files.get(name=file.name)

        if file.state is not None and file.state.name == "FAILED":
            logger.error("The video processing failed in the API")
            if file.name:
                self.client.files.delete(name=file.name)
            return

        logger.success("The video was successfully processed")

        # 3. Generación con Pydantic
        try:
            logger.debug("Generating annotations with the model...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=[file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VideoAnalysis,
                    "temperature": 0.1,
                },
            )

            id_str: str = video.split(".mp4")[0].split("/")[-1]

            if response.parsed and isinstance(response.parsed, VideoAnalysis):
                if response.parsed.num_signers == 0:
                    out_path = os.path.join(self.failed_dir, f"{id_str}.json")
                    logger.warn(f"No signers detected → {out_path}")
                else:
                    out_path = os.path.join(self.out_dir, f"{id_str}.json")
                    logger.success(
                        f"Annotations generated ({response.parsed.num_signers} signers, {len(response.parsed.annotations)} annotations)"
                    )

                with open(out_path, "w", encoding="utf-8") as f:
                    response.parsed.model_dump()
                    f.write(response.parsed.model_dump_json(indent=3))

                logger.info(f"Saved: {out_path}")
            else:
                logger.warn("Unexpected schema returned")

        except Exception as e:
            logger.error(f"Error generating: {e}")

        finally:
            if file.name:
                self.client.files.delete(name=file.name)
                logger.debug("Temporal file deleted from the API")


# --- PROMPT SIMPLIFICADO ---
prompt = """
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


def split(id: int = 0, rest: bool = False) -> slice:
    return slice((id - 1) * 178, id * 178 if not rest else None)


@click.command()
@click.option("-m", "--metadata_file", type=str, required=True)
@click.option("-o", "--out_dir", type=str, required=True)
@click.option("-i", "--id", type=int, required=True)
@click.option("-r", "--rest", is_flag=True, default=False)
@click.option("-s", "--stats", is_flag=True, default=False)
def cli(metadata_file: str, out_dir: str, id: int, rest: bool, stats: bool):
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return

    logger.info(f"Loaded metadata file from: {metadata_file}")

    with open(metadata_file, "r") as f:
        info: list[dict] = json.load(f)["info"]
        info.sort(key=lambda entry: entry["duration"])
        mdata: list[dict] = info[split(id=id, rest=rest)]

        batch_size: int = len(mdata)
        whole: set[str] = set([entry["filename"].split(".mp4")[0] for entry in mdata])
        processed: set[str] = set(
            os.path.basename(file).split(".json")[0]
            for file in os.listdir(out_dir)
            if os.path.basename(file).endswith(".json")
        )

        failed: set[str] = set(
            os.path.basename(file).split(".json")[0]
            for file in os.listdir(os.path.join(out_dir, "failed"))
            if os.path.basename(file).endswith(".json")
        )

        f2p: set[str] = whole.difference(processed | failed)
        mdata = [entry for entry in mdata if entry["filename"].split(".mp4")[0] in f2p]

        logger.info(f"Skipping {batch_size - len(f2p)}, of {batch_size}")

    logger.info(f"Bacth {id}: {len(mdata)} waiting to be processed")

    if stats:
        total_anns: int = 0
        total_duration: float = 0.0
        done_annotations = [
            os.path.join(out_dir, file)
            for file in os.listdir(out_dir)
            if file.endswith(".json")
        ]

        for done in done_annotations:
            with open(done, "r") as f:
                anns = json.load(f)["annotations"]
                total_anns += len(anns)
                for ann in anns:
                    total_duration += ann["duration"]

        logger.info(f"{total_anns}")
        logger.info(f"{total_duration}")

    ann = Annotator(
        model=Models.Gemini2dot5Pro,
        api_key=ApiKeyProvider.api_key(id=1),
        out_dir=out_dir,
    )

    for idx, d in enumerate(mdata, 1):
        logger.info(f"[{idx}/{len(mdata)}] Starting the procedure")
        logger.info(f"Video duration in minutes {d['duration'] / 60}")
        ann.annotate(
            video=os.path.join("./datamining/CNSE/videos/", d["filename"]),
            prompt=prompt.format(d["duration"], d["frames"]),
        )
        logger.info("-" * 80)

    logger.success(f"Batch {id} completed. {len(mdata)} processed videos")


if __name__ == "__main__":
    cli()
