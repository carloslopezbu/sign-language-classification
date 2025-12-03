import json
import os
import sys

import dotenv
from google import genai

dotenv.load_dotenv()


class ApiKeyProvider:
    @staticmethod
    def api_key(id: int) -> str | None:
        return os.getenv(f"GEMINI_API_KEY{id}")


class Annotator:
    def __init__(self, model: str, api_key: str, out_dir: str) -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.out_dir = out_dir

    def annotate(self, video: str, prompt: str) -> None:
        file = self.client.files.upload(file=video)

        while file.state is not None and file.state.name == "PROCESSING":
            if file.name:
                file = self.client.files.get(name=file.name)

        if file.state is not None and file.state.name == "FAILED":
            print("Error fallo en la descarga del video", file=sys.stderr)
        else:
            print(f"SUCCESS video {file.name} cargado")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[file, prompt],
            config={"response_mime_type": "application/json", "temperature": 0.1},
        )

        id: str = video.split(".mp4")[0]
        with open(f"{os.path.join(self.out_dir, id)}.json") as f:
            json.dump(response.text, f)
            print(f"SAVING {video} ANNOTATIONS")
