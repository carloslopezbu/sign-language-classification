import os
import time
from enum import StrEnum
from typing import List, Literal

import click
from config import Config
from google import genai
from google.genai.types import (
    ContentListUnionDict,
    File,
    GenerateContentResponse,
    SchemaUnionDict,
)
from pydantic import BaseModel, Field
from utils import Console, Table, load_json, logger, save_json


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
    def api_key(id: int | None) -> str | None:
        return os.getenv(f"GEMINI_API_KEY{id if id is not None else ''}")


class Models(StrEnum):
    Gemini2dot0Flash = "gemini-2.0-flash"
    Gemini2dot5Flash = "gemini-2.5-flash"
    Gemini2dot5Pro = "gemini-2.5-pro"
    Gemini3dot0Pro = "gemini-3-pro"
    Gemini3dot0Flash = "gemini-3-flash"


class Annotator:
    def __init__(self, model: str, api_key: str | None, output_dir: str) -> None:
        self.model = model
        if api_key is None:
            raise RuntimeError("api Key is None")
        self.client = genai.Client(api_key=api_key)
        self.output_dir = output_dir
        self.failed_dir = os.path.join(output_dir, "failed")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.failed_dir, exist_ok=True)

    def _upload_file(self, file: str) -> File | None:
        try:
            upfile = self.client.files.upload(file=file)
            logger.success("video uploaded successfully")
            return upfile
        except Exception as e:
            logger.error(f"error uploading the video: {e}")
            return None

    def _process(self, file: File, seconds: float = 1.0):
        while file.state is not None and file.state.name == "PROCESSING":
            time.sleep(seconds)
            if file.name:
                file = self.client.files.get(name=file.name)

    def _failed(self, file: File) -> bool:
        if file.state is not None and file.state.name == "FAILED":
            logger.error("the video processing failed in the API")
            if file.name:
                self.client.files.delete(name=file.name)
            return True

        return False

    def _generate(
        self,
        contents: ContentListUnionDict,
        response_schema: SchemaUnionDict,
        temperature: float = 0.1,
    ) -> GenerateContentResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": temperature,
            },
        )

        return response

    def annotate(self, video: str, prompt: str) -> None:
        video_name = os.path.basename(video)
        logger.info(f"processing video: {video_name}")

        logger.debug("uploading video to Gemini API...")

        if not (file := self._upload_file(file=video)):
            return

        logger.debug("waiting the video to be proccessed...")
        self._process(file=file)

        if self._failed(file=file):
            return

        logger.success("the video was successfully processed")

        try:
            logger.debug("generating annotations with the model...")
            response = self._generate(
                contents=[file, prompt], response_schema=VideoAnalysis, temperature=0.1
            )

            id: str = video_name.split(".mp4")[0]

            if response.parsed and isinstance(response.parsed, VideoAnalysis):
                if response.parsed.num_signers == 0:
                    out_path = os.path.join(self.failed_dir, f"{id}.json")
                    logger.warn(f"no signers detected → {out_path}")
                else:
                    out_path = os.path.join(self.output_dir, f"{id}.json")
                    logger.success(
                        f"annotations generated ({response.parsed.num_signers} signers, {len(response.parsed.annotations)} annotations)"
                    )

                save_json(obj=response.parsed.model_dump(), dest=out_path)

                logger.info(f"saved: {out_path}")
            else:
                logger.warn("unexpected schema returned")

        except Exception as e:
            logger.error(f"error generating: {e}")

        finally:
            if file.name:
                self.client.files.delete(name=file.name)
                logger.debug("temporal file deleted from the API")

    def fix(self):
        pass


def bacthify(split: int = 0, batch_size: int = 178, rest: bool = False) -> slice:
    return slice((split - 1) * batch_size, split * batch_size if not rest else None)


def cmd_annotate(
    metadata_file: str | None,
    output_dir: str,
    id: int | None,
    split: int | None,
    rest: bool,
):
    if metadata_file is None:
        logger.error("metadata_file is not provided")
        return

    if id is None:
        logger.error("id is not provided")
        return

    if split is None:
        logger.error("split is not provided")
        return

    if not os.path.exists(metadata_file):
        logger.error(f"metadata file not found: {metadata_file}")
        return

    logger.info(f"loaded metadata file from: {metadata_file}")

    metadata_obj: dict = load_json(src=metadata_file)
    info: list[dict] = metadata_obj["info"]

    info.sort(key=lambda entry: entry["duration"])
    videos_metadata: list[dict] = info[bacthify(split=split, rest=rest)]

    batch_size: int = len(videos_metadata)
    whole: set[str] = set(
        [entry["filename"].split(".mp4")[0] for entry in videos_metadata]
    )
    processed: set[str] = set(
        os.path.basename(file).split(".json")[0]
        for file in os.listdir(output_dir)
        if os.path.basename(file).endswith(".json")
    )

    failed: set[str] = set(
        os.path.basename(file).split(".json")[0]
        for file in os.listdir(os.path.join(output_dir, "failed"))
        if os.path.basename(file).endswith(".json")
    )

    f2p: set[str] = whole.difference(processed | failed)
    videos_metadata = [
        entry for entry in videos_metadata if entry["filename"].split(".mp4")[0] in f2p
    ]

    logger.info(f"skipping {batch_size - len(f2p)}, of {batch_size}")
    logger.info(f"bacth {id}: {batch_size} waiting to be processed")

    ann = Annotator(
        model=Models.Gemini2dot5Pro,
        api_key=ApiKeyProvider.api_key(id=None),
        output_dir=output_dir,
    )

    for idx, d in enumerate(videos_metadata, 1):
        logger.info(f"[{idx}/{len(videos_metadata)}] Starting the procedure")
        logger.info(f"Video duration in minutes {d['duration'] / 60}")
        ann.annotate(
            video=os.path.join("./datamining/CNSE/videos/", d["filename"]),
            prompt=Config.PROMPT4ANNOTATION.format(d["duration"], d["frames"]),
        )
        logger.info("-" * 80)

    ann.client.close()
    logger.success(f"Batch {id} completed. {len(videos_metadata)} processed videos")


def cmd_stats(output_dir: str):
    num_anns: int = 0
    durations: list[float] = []
    std: float = 0.0
    mean: float = 0.0
    done_annotations = [
        os.path.join(output_dir, file)
        for file in os.listdir(output_dir)
        if file.endswith(".json")
    ]

    max_ann = {}
    max_dur = -1
    d = ""

    for done in done_annotations:
        obj = load_json(src=done)
        anns = obj["annotations"]

        num_anns += len(anns)
        for ann in anns:
            durations.append(ann["duration"])

            if max_dur < ann["duration"]:
                max_ann = ann
                max_dur = ann["duration"]
                d = done

    duration: float = sum(durations)
    mean = duration / num_anns
    std = sum((mean - x) * (mean - x) for x in durations) / num_anns

    console = Console()
    table = Table(
        title="📊 Estadísticas de Anotaciones",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Métrica", style="cyan", no_wrap=True)
    table.add_column("Valor", style="green")

    table.add_row("Total de anotaciones", str(num_anns))
    table.add_row("Duración total", f"{duration:.2f}s")
    table.add_row("Duración promedio", f"{mean:.2f}s")
    table.add_row("Desviación estándar", f"{std:.2f}s")
    table.add_row("Duración mínima", f"{min(durations):.2f}s")
    table.add_row("Duración máxima", f"{max(durations):.2f}s")

    console.print(table)

    logger.debug(f"{max_ann}  {d}")


def cmd_fix_json_encoding(output_dir: str): ...


def cmd_clean(id: int | None):
    client = genai.Client(api_key=ApiKeyProvider.api_key(id=id))
    files = client.files.list()

    logger.info("starting the file cleaning...")

    for file in files:
        if file.name is not None:
            client.files.delete(name=file.name)

    client.close()

    logger.success("cleaning completed")


@click.command()
@click.option("-m", "--metadata_file", type=str)
@click.option(
    "-o",
    "--output_dir",
    type=str,
)
@click.option("-i", "--id", type=int, default=None)
@click.option("-s", "--split", type=int)
@click.option("-a", "--annotate", is_flag=True, default=False)
@click.option("-r", "--rest", is_flag=True, default=False)
@click.option("-x", "--stats", is_flag=True, default=False)
@click.option("-e", "--fix-json-encoding", is_frag=True, default=False)
@click.option("-c", "--clean", is_flag=True, default=False)
def cmd(
    metadata_file: str | None,
    output_dir: str | None,
    id: int | None,
    split: int | None,
    annotate: bool,
    rest: bool,
    stats: bool,
    clean: bool,
):
    if clean:
        cmd_clean(id=id)

    if output_dir is None:
        logger.warn("nothing to do here if no output directory provided")
        return

    if stats:
        cmd_stats(output_dir=output_dir)

    if annotate:
        cmd_annotate(
            metadata_file=metadata_file,
            output_dir=output_dir,
            id=id,
            split=split,
            rest=rest,
        )
    else:
        raise NotImplementedError("ANNOTATION FIXING")


if __name__ == "__main__":
    cmd()
