import os
from enum import StrEnum
from typing import cast

import cv2
import numpy as np
import torch
from annotator import VideoAnalysis
from torch import Tensor
from ultralytics.engine.results import Results
from ultralytics.models.yolo import YOLO
from utils import logger, save_json


def s2ms(secs: float) -> float:
    return secs * 1000


def ms2s(secs: float) -> float:
    return secs / 1000


class Video:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.cap: cv2.VideoCapture | cv2.VideoWriter | None = None

    def start_reading(self) -> bool:
        if not os.path.exists(self.filename):
            logger.error(f"'{self.filename}' does not exist")
            return False

        self.cap = cv2.VideoCapture(self.filename)

        if not self.cap.isOpened():
            logger.error("error opening the video")
            return False

        return True

    def goto(self, instant: float) -> bool:
        assert self.cap is not None
        return self.cap.set(cv2.CAP_PROP_POS_MSEC, instant)

    def read(self) -> tuple[bool, np.ndarray]:
        assert self.cap is not None and self.cap is cv2.VideoCapture
        return self.cap.read()

    def stop(self) -> None:
        assert self.cap is not None
        self.cap.release()
        self.cap = None

    def current_instant(self) -> float:
        assert self.cap is not None
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def metadata(self) -> dict[str, float | str]:
        if not self.start_reading():
            return {}

        assert self.cap is not None

        fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        frames: float = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width: float = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height: float = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if fps > 0:
            duration: float = frames / fps
        else:
            logger.error(f"inconsistent fps in {self.filename} video, fps < 0")
            return {}

        self.stop()
        filename = self.filename.split("/")[-1]
        return {
            "filename": filename,
            "fps": fps,
            "frames": frames,
            "width": width,
            "height": height,
            "duration": duration,
        }


class VideoFileNameDataset:
    def __init__(self, dir) -> None:
        self.dir = dir
        self.video_filenames: list[str] = []

    def load(self) -> None:
        self.video_filenames = [
            os.path.join(self.dir, video)
            for video in os.listdir(self.dir)
            if video.endswith(".mp4")
        ]

    def metadata(self) -> dict[str, float | list[dict[str, float | str]]]:
        md = []
        total_duration: float = 0.0
        for filename in self.video_filenames:
            video = Video(filename=filename)
            if video.start_reading():
                video_md = video.metadata()
                total_duration += cast(float, video_md["duration"])
                md.append(video_md)
                video.stop()
        md = {"total_duration": total_duration, "info": md}
        return md


class YOLOTask(StrEnum):
    Segmentation = "seg"
    PoseEstimation = "pose"


type ImageSource = str | np.ndarray | Tensor
type VideoSource = int | str
type Sources = ImageSource | VideoSource


class YOLOModel:
    def __init__(self, model: str, task: YOLOTask):
        self.task = task
        self.estimator = YOLO(model=model, task=self.task.value)

    def __call__(self, source: Sources, stream: bool = True) -> list[Results]:
        results = self.estimator(source=source, stream=stream, classes=[0])

        return results

    def display_video(self, source: VideoSource):
        results = self(source=source, stream=True)

        for result in results:
            annotated_frame = result.plot()
            cv2.imshow(f"YOLO ({self.task.name})", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def display_image(self, source: Sources):
        result = self(source, stream=False)[0]
        annotated_frame = result.plot()
        cv2.imshow(f"YOLO ({self.task.name})", annotated_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


class VideoProcessor:
    def __init__(
        self, pose_estimator: YOLOModel, person_segmentator: YOLOModel
    ) -> None:
        self.pe = pose_estimator
        self.ps = person_segmentator

    def process(self, video: Video, va: VideoAnalysis):
        video.start_reading()
        for ann in va.annotations:
            video.goto(ann.start)

            ok: bool = True
            end: float = ann.duration * 1000 + ann.start
            while ok and video.current_instant() < end:
                ok, frame = video.read()
                pose_est = self.pe(frame, stream=False)
