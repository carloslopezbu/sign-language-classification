import json
import os
import time
from enum import StrEnum
from typing import List, Literal

import click
import dotenv
from google import genai
from pydantic import BaseModel, Field
from utils import logger

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
