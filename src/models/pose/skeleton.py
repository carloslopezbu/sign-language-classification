import os
from dataclasses import dataclass

import cv2
from ultralytics import YOLO


class RealTimeEstimator:
    task: str | None = None

    def __init__(self, path: str = ""):
        if self.task is None:
            raise NotImplementedError(
                "Can't instantiate RealTimeEstimator. Use PoseEstimator or PersonSegmentator."
            )

        model_name = f"yolo11n-{self.task}.pt"
        self.estimator = YOLO(model=os.path.join(path, model_name))

    def __call__(self, *args, **kwds):
        return self.estimator(args=args, kwargs=kwds)

    def display(self, source: str) -> None:
        results = self.estimator(source=source, stream=True, classes=[0])

        for result in results:
            annotated_frame = result.plot()
            cv2.imshow(f"YOLO11 Task ({self.task})", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


class PoseEstimator(RealTimeEstimator):
    task = "pose"


class PersonSegmentator(RealTimeEstimator):
    task = "seg"
