from typing import override
import cv2
import os
import numpy as np
import torch

# Obtenido  de https://www.kaggle.com/code/ahmedabdelfattah20/image-augmentation-using-opencv


class RGBVideoLoader:
    def __init__(self, fps: int = 16, size: tuple[int, int] = (224, 224)) -> None:
        self.fps: int = fps
        self.size: tuple[int, int] = size
        self.videos: list[np.ndarray]
        self.from_dir: bool

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
            raise ValueError("El video debe estar a 16 fps")

        ok, frame = cap.read()

        if ok and len(frame) == 3 and frame.shape[-1] != 3:
            cap.release()
            raise RuntimeError(f"El video {path} no es RGB")
        elif not ok:
            cap.release()
            raise RuntimeError(f"Hubo un error leyendo el video {path}")

        frames: list[np.ndarray] = []

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

        self.from_dir = os.path.isdir(path)
        if self.from_dir:
            video_paths = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if RGBVideoLoader.is_mp4(f)
            ]
            self.videos = [self._load_video(video_path) for video_path in video_paths]
        else:
            self.videos = [self._load_video(path)]


class DataAugmentator:
    def __init__(self) -> None:
        pass

    def __call__(self, dataloader: RGBVideoLoader) -> dict[str, list[np.ndarray]]:
        return {
            "original": dataloader.videos,
            "blurred": [
                np.array([self.Blur(frame) for frame in video])
                for video in dataloader.videos
            ],
            "random-noise": [
                np.array([self.RandomNoise(frame) for frame in video])
                for video in dataloader.videos
            ],
            "warp-affine": [
                np.array([self.WarpAffine(frame) for frame in video])
                for video in dataloader.videos
            ],
            "random-resize": [
                np.array([self.RandomResize(frame) for frame in video])
                for video in dataloader.videos
            ],
        }

    @staticmethod
    def Blur(frame: np.ndarray):
        ksize = np.random.randint(5, 15)
        return cv2.blur(frame, (ksize, ksize))

    @staticmethod
    def RandomNoise(frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        h = (h + np.random.randint(0, 100, h.shape, np.uint8)) % 180
        s = np.clip(s + np.random.randint(0, 20, s.shape, np.uint8), 0, 255)
        v = np.clip(v + np.random.randint(0, 10, v.shape, np.uint8), 0, 255)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def WarpAffine(frame: np.ndarray):
        shape: tuple[int, int, int] = frame.shape
        (cols, rows, _) = shape
        cbound: int = int(cols * 0.25)
        rbound: int = int(rows * 0.25)

        tx: int = np.random.randint(-cbound, cbound)
        ty: int = np.random.randint(-rbound, rbound)
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        return cv2.warpAffine(frame, M, (cols, rows))

    @staticmethod
    def RandomResize(frame: np.ndarray):
        shape: tuple[int, int, int] = frame.shape
        (cols, rows, _) = shape
        cbound: int = int(cols * 0.25)
        rbound: int = int(rows * 0.25)

        tx: int = np.random.randint(-cbound, cbound)
        ty: int = np.random.randint(-rbound, rbound)
        x, y = max(tx, 0), max(ty, 0)
        w, h = cols - abs(tx), rows - abs(ty)
        rsize = frame[y : y + h, x : x + w]
        rsize = cv2.resize(rsize, (cols, rows))
        return rsize
