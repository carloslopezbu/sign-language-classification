import os
from enum import StrEnum
from itertools import islice
from typing import Generator, Iterable, Self, cast, overload

import cv2
import numpy as np
import torch
from annotator import Annotation, VideoAnalysis
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset
from ultralytics.engine.results import Results
from ultralytics.models.yolo import YOLO
from utils import load_json, logger, save_json


def s2ms(secs: float) -> float:
    return secs * 1000


def ms2s(secs: float) -> float:
    return secs / 1000


class exceptions:
    class NoVideoCapError(Exception):
        def __init__(self):
            self.message = "self.cap is None"
            super().__init__(self.message)

    class WrongVideoHandlerError(Exception):
        def __init__(self, correct_cls):
            self.message = f"wrong video handler, should use {correct_cls}"
            super().__init__(self.message)

    class NoVideoProcessorProvidedError(Exception):
        def __init__(self, task):
            self.message = f"no video processor for {task} task"
            super().__init__(self.message)


class BoundingBox(BaseModel):
    x1: int
    x2: int
    y1: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def union(self, other: Self) -> Self:
        self.x1 = min(self.x1, other.x1)
        self.y1 = min(self.y1, other.y1)

        self.x2 = max(self.x2, other.x2)
        self.y2 = max(self.y2, other.y2)
        return self


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

    def goto_frame(self, frame: int) -> bool:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def goto_instant(self, instant: float) -> bool:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return self.cap.set(cv2.CAP_PROP_POS_MSEC, instant)

    def read(self) -> tuple[bool, np.ndarray]:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        if not isinstance(self.cap, cv2.VideoCapture):
            raise exceptions.WrongVideoHandlerError(cv2.VideoCapture.__name__)

        return self.cap.read()

    def stop(self) -> None:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        self.cap.release()
        self.cap = None

    def current_instant(self) -> float:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def current_frame(self) -> int:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def slice(
        self, start_sec: float, duration_sec: float, vid_stride: int = 1
    ) -> Generator[np.ndarray]:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        if not isinstance(self.cap, cv2.VideoCapture):
            raise exceptions.WrongVideoHandlerError(cv2.VideoCapture.__name__)

        start_ms: float = s2ms(start_sec)
        end_ms: float = start_ms + s2ms(duration_sec)

        self.goto_instant(start_ms)

        curr_frame: int = self.current_frame()

        while self.current_instant() <= end_ms:
            if vid_stride != 1:
                curr_frame += vid_stride
                if curr_frame >= self.num_frames:
                    break

                self.goto_frame(curr_frame)

            ok, frame = self.read()
            if not ok:
                break
            yield frame

    @property
    def fps(self) -> int:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return int(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def num_frames(self) -> int:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self.cap is None:
            raise exceptions.NoVideoCapError

        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def metadata(self) -> dict[str, float | str]:
        if not self.start_reading():
            return {}

        if self.cap is None:
            raise exceptions.NoVideoCapError

        fps: float = self.fps
        frames: float = self.num_frames
        width: float = self.width
        height: float = self.height

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


class VideoDataset(Dataset):
    def __init__(self, video_dir: str, video_analysis_dir: str) -> None:
        if not os.path.exists(video_analysis_dir):
            raise FileNotFoundError(f"{video_dir=} does not exist")

        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"{video_dir=} does not exist")

        self.video_analysis_filenames: list[str] = [
            os.path.join(video_analysis_dir, annotation)
            for annotation in os.listdir(video_analysis_dir)
            if annotation.endswith(".json")
        ]

        self.video_filenames: list[str] = [
            os.path.join(video_dir, f"{video.split('.')[0]}.mp4")
            for video in os.listdir(video_analysis_dir)
        ]

    def __len__(self) -> int:
        return len(self.video_filenames)

    def load_video(self, index: int) -> Video:
        return Video(self.video_filenames[index])

    def load_video_analysis(self, index: int) -> VideoAnalysis:
        obj = load_json(self.video_analysis_filenames[index])
        return VideoAnalysis.model_validate(obj)

    def __getitem__(self, index: int) -> tuple[Video, VideoAnalysis]:
        return self.load_video(index), self.load_video_analysis(index)

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


class Task(StrEnum):
    Segmentation = "seg"
    PoseEstimation = "pose"
    Detection = "dectect"


type ImageSource = str | np.ndarray | Tensor
type VideoSource = int | str | list
type Sources = ImageSource | VideoSource


class VideoProcessor:
    def __init__(self, model: str, task: Task):
        self.task = task
        self.estimator = YOLO(model=model, task=self.task.value)

    def __call__(self, source: Sources, stream: bool = False) -> list[Results]:
        results = self.estimator(source=source, stream=stream, classes=[0])

        return results

    def track(self, source: Sources, stream: bool = False) -> Results:
        return self.estimator.track(source, stream=stream, classes=[0])[0]

    def process_video(self, video: VideoSource) -> list[Results]:
        return self(source=video, stream=True)

    def process_frame(self, image: ImageSource) -> Results:
        return self(source=image, stream=False)[0]

    def display_video(self, source: VideoSource):
        results = self(source=source, stream=True)

        for result in results:
            annotated_frame = result.plot()
            cv2.imshow(f"YOLO ({self.task.name})", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def display_frame(self, source: Sources):
        result = self(source, stream=False)[0]

        annotated_frame = result.plot()
        cv2.imshow(f"YOLO ({self.task.name})", annotated_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


class SltVideoProcessor:
    def __init__(
        self,
        pose_estimator: VideoProcessor | None = None,
        person_segmentator: VideoProcessor | None = None,
        person_detector: VideoProcessor | None = None,
    ) -> None:
        self.pe = pose_estimator
        self.ps = person_segmentator
        self.pd = person_detector

    def _eficient_processing(
        self,
        model: VideoProcessor,
        video: Video,
        va: VideoAnalysis,
        vid_stride: int = 1,
    ) -> Generator[Results]:
        if not video.start_reading():
            return

        for ann in va.annotations:
            soi = video.slice(ann.start, ann.duration, vid_stride)

            for frame in soi:
                yield model.process_frame(frame)

        video.stop()

    def pose_estimation(
        self, video: Video, va: VideoAnalysis, vid_stride: int = 1
    ) -> Generator[Results]:
        if self.pe is None:
            raise exceptions.NoVideoProcessorProvidedError(Task.PoseEstimation.name)

        return self._eficient_processing(
            model=self.pe, video=video, va=va, vid_stride=vid_stride
        )

    def person_segmentation(
        self, video: Video, va: VideoAnalysis, vid_stride: int = 1
    ) -> Generator[Results]:
        if self.ps is None:
            raise exceptions.NoVideoProcessorProvidedError(Task.Segmentation.name)

        return self._eficient_processing(
            model=self.ps, video=video, va=va, vid_stride=vid_stride
        )

    def person_detection(
        self, video: Video, va: VideoAnalysis, vid_stride: int = 1
    ) -> Generator[Results]:
        if self.pd is None:
            raise exceptions.NoVideoProcessorProvidedError(Task.Detection.name)

        return self._eficient_processing(
            model=self.pd, video=video, va=va, vid_stride=vid_stride
        )

    def first_glance(self, video: Video, ann: Annotation) -> dict[int, BoundingBox]:
        if self.pd is None:
            raise exceptions.NoVideoProcessorProvidedError(Task.Detection.name)

        vid_stride: int = max(1, video.fps)
        soi = video.slice(
            start_sec=ann.start, duration_sec=ann.duration, vid_stride=vid_stride
        )
        return {}

    def slice_slt_video(self, video: Video, va: VideoAnalysis):
        if video.start_reading():
            return

        vid_stride: int = video.fps

        for ann in va.annotations:
            soi = video.slice(
                start_sec=ann.start, duration_sec=ann.duration, vid_stride=vid_stride
            )
            ...
