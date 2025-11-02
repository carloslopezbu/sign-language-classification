from typing import override
from enum import Enum
import numpy as np
import torch
import cv2
import os


class VideoSrc(Enum):
    Video = 1
    VideoBatch = 2
    Dir = 3


class VideoBatcher:
    def __init__(self, dir: str, batch_size: int) -> None:
        if not os.path.isdir(dir):
            raise RuntimeError(f"{dir} no es un directorio")

        if batch_size < 0:
            raise ValueError("El tamaño de batch debe ser mayor que 0")

        self.batch_size: int = batch_size
        self.current: int = 0

        self.video_paths: list[str] = [
            os.path.join(dir, path)
            for path in os.listdir(dir)
            if os.path.exists(os.path.join(dir, path)) and RGBVideoLoader.is_mp4(path)
        ]

        if batch_size > len(self.video_paths):
            raise ValueError(
                "El tamaño del batch no debe ser mayor que el tamaño los videos"
            )

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self) -> list[str]:
        if self.current >= len(self.video_paths):
            raise StopIteration

        start = self.current
        end = min(start + self.batch_size, len(self.video_paths))
        batch = self.video_paths[start:end]
        self.current = end
        return batch


class RGBVideoLoader:
    def __init__(self, fps: int = 16, size: tuple[int, int] = (224, 224)) -> None:
        self.fps: int = fps
        self.size: tuple[int, int] = size
        self.videos: list[np.ndarray]
        self.src: VideoSrc

    @override
    def __repr__(self) -> str:
        return f"RGBVideoLoader(fps={self.fps} size={self.size} videos=({len(self.videos)} {str(self.videos[0].shape).replace('(', ',').replace(')', '')}))"

    @staticmethod
    def is_mp4(video: str) -> bool:
        return video.endswith(".mp4")

    def _load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps - self.fps > 1e-6:
            raise RuntimeError(f"El video debe estar a {self.fps} fps")

        ok, frame = cap.read()

        if ok and len(frame) == 3 and frame.shape[-1] != 3:
            cap.release()
            raise RuntimeError(f"El video {path} no es RGB")
        elif not ok:
            cap.release()
            raise RuntimeError(f"Hubo un error leyendo el video {path}")

        frames: list[np.ndarray] = [frame]

        while cap.isOpened():
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(src=frame, dsize=self.size, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def load(self, path: str):
        if not os.path.exists(path):
            raise RuntimeError(f"No exite {path}")

        self.src = VideoSrc.Dir if os.path.isdir(path) else VideoSrc.Video
        match self.src:
            case VideoSrc.Dir:
                video_paths = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if RGBVideoLoader.is_mp4(f)
                ]
                self.videos = [
                    self._load_video(video_path) for video_path in video_paths
                ]
            case VideoSrc.Video:
                self.videos = [self._load_video(path)]

    def load_batch(self, paths: list[str]) -> np.ndarray | None:
        mask: np.ndarray = np.array(
            [os.path.exists(path) and RGBVideoLoader.is_mp4(path) for path in paths],
            dtype=bool,
        )
        if not np.all(mask):
            raise RuntimeError("Algunos paths no existen o no son .mp4")

        self.src = VideoSrc.VideoBatch
        self.videos = [self._load_video(video_path) for video_path in paths]
