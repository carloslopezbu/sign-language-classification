import json
import os
from dataclasses import dataclass

import cv2
import dotenv
import numpy as np
from cv2.dnn import TextDetectionModel_EAST
from ultralytics import YOLO
from ultralytics.engine.results import Results

_ = dotenv.load_dotenv()

METADATA_VIDEOS_FILE: str = os.getenv("METADATA_VIDEOS_FILE", default="no.file")
VIDEOS_PATH: str = os.getenv("VIDEOS_PATH", default="no.path")
ANNOTATIONS_PATH: str = os.getenv("ANNOTATIONS_PATH", default="no.path")
FOURCC_CODE: int = int(os.getenv("FOURCC_CODE", default=1983148141))


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class TextDetector:
    def __init__(self, east: TextDetectionModel_EAST):
        self.east: TextDetectionModel_EAST = east

    def detect_in_frame(
        self, frame: np.ndarray, padding: int = 0
    ) -> BoundingBox | None:
        """
        Detects text regions in a single frame.
        Returns a BoundingBox containing all detected text or None.
        """
        rects, _ = self.east.detectTextRectangles(frame)

        if len(rects) == 0:
            return None

        text_x1, text_y1 = float("inf"), float("inf")
        text_x2, text_y2 = 0, 0

        for rect in rects:
            vertices = cv2.boxPoints(rect).astype(int)
            x, y, w, h = cv2.boundingRect(vertices)

            text_x1 = min(text_x1, x)
            text_y1 = min(text_y1, y)
            text_x2 = max(text_x2, x + w)
            text_y2 = max(text_y2, y + h)

        h, w = tuple[int, int](frame.shape[:2])
        text_x1 = max(0, int(text_x1) - padding)
        text_y1 = max(0, int(text_y1) - padding)
        text_x2: int = min(w, int(text_x2) + padding)
        text_y2: int = min(h, int(text_y2) + padding)

        return BoundingBox(text_x1, text_y1, text_x2, text_y2)


class PersonDetector:
    PERSON_CLASS_ID: int = 0

    def __init__(self, yolo: YOLO):
        self.yolo: YOLO = yolo

    def detect_in_frame(self, cap: cv2.VideoCapture, padding: int = 0) -> BoundingBox:
        """
        Detects a person in the current frame.
        Returns a BoundingBox for the first detected person.
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, frame = cap.read()
        if not ok:
            return BoundingBox(0, 0, width, height)

        results: list[Results] = list[Results](
            self.yolo(frame, verbose=False, device="mps")
        )

        person_x1, person_y1 = float("inf"), float("inf")
        person_x2, person_y2 = 0, 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls)
                    if cls == self.PERSON_CLASS_ID:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        person_x1 = min(person_x1, x1)
                        person_y1 = min(person_y1, y1)
                        person_x2 = max(person_x2, x2)
                        person_y2 = max(person_y2, y2)

        if person_x1 == float("inf"):
            return BoundingBox(0, 0, width, height)

        person_x1 = max(0, int(person_x1) - padding)
        person_y1 = max(0, int(person_y1) - padding)
        person_x2 = min(width, int(person_x2) + padding)
        person_y2 = min(height, int(person_y2) + padding)

        return BoundingBox(person_x1, person_y1, person_x2, person_y2)


class VideoProcessor:
    def __init__(
        self,
        yolo: YOLO,
        east: TextDetectionModel_EAST,
    ):
        self.person_detector = PersonDetector(yolo)
        self.text_detector = TextDetector(east)

    def detect_text_in_segment(
        self, cap: cv2.VideoCapture, start: float, person_bbox: BoundingBox
    ) -> BoundingBox | None:
        """
        Detects text in the first frame of a segment after cropping to person.
        Returns a BoundingBox for the text region or None.
        """
        cap.set(cv2.CAP_PROP_POS_MSEC, start)
        ok, frame = cap.read()

        if not ok:
            return None

        crop = frame[person_bbox.y1 : person_bbox.y2, person_bbox.x1 : person_bbox.x2]
        return self.text_detector.detect_in_frame(crop, padding=5)

    def apply_text_blur(
        self, frame: np.ndarray, text_bbox: BoundingBox | None
    ) -> np.ndarray:
        """
        Applies Gaussian blur to the text region in the frame.
        """
        if text_bbox is None:
            return frame

        text_region = frame[text_bbox.y1 : text_bbox.y2, text_bbox.x1 : text_bbox.x2]

        if text_region.size > 0:
            blurred = cv2.GaussianBlur(text_region, (51, 51), 0)
            frame[text_bbox.y1 : text_bbox.y2, text_bbox.x1 : text_bbox.x2] = blurred

        return frame

    def process_segment(
        self,
        cap: cv2.VideoCapture,
        start: float,
        end: float,
        person_bbox: BoundingBox,
        text_bbox: BoundingBox | None,
        output_path: str,
        remove_text: bool = False,
    ) -> None:
        """
        Processes an entire video segment frame by frame using pre-computed bboxes.
        If remove_text=True, crops out text region (assumes text is at bottom).
        If remove_text=False, applies blur to text region.
        """
        cap.set(cv2.CAP_PROP_POS_MSEC, start)

        if remove_text and text_bbox is not None:
            crop_height = text_bbox.y1
            crop_width = person_bbox.width
        else:
            crop_height = person_bbox.height
            crop_width = person_bbox.width

        if crop_height <= 0 or crop_width <= 0:
            print("⚠️ Invalid crop dimensions, using full frame")
            crop_height = person_bbox.height
            crop_width = person_bbox.width
            remove_text = False
        wrt = cv2.VideoWriter(
            filename=output_path,
            fourcc=FOURCC_CODE,
            fps=cap.get(cv2.CAP_PROP_FPS),
            frameSize=(crop_width, crop_height),
        )

        ok = True

        while cap.get(cv2.CAP_PROP_POS_MSEC) < end and ok:
            ok, frame = cap.read()
            if not ok:
                break

            crop = frame[
                person_bbox.y1 : person_bbox.y2, person_bbox.x1 : person_bbox.x2
            ]

            if text_bbox is not None:
                crop = (
                    crop[0 : text_bbox.y1, :]
                    if remove_text
                    else self.apply_text_blur(crop, text_bbox)
                )

            wrt.write(crop)

        wrt.release()

    def process_video(
        self,
        video_path: str,
        annotations_path: str,
        output_dir: str,
        padding: int = 50,
        remove_text: bool = True,
    ) -> None:
        """
        Processes an entire video with annotations, creating sliced output videos.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        with open(annotations_path, "r") as f:
            annotations = json.load(f)["annotations"]

        prev_signer = -2
        prev_position = "NoName"
        person_bbox = BoundingBox(0, 0, 0, 0)
        text_bbox: BoundingBox | None = None

        for idx, ann in enumerate(annotations):
            curr_signer = ann["signer_id"]
            curr_position = ann["screen_position"]
            start = ann["start"] * 1000
            end = start + ann["duration"] * 1000

            if curr_signer != prev_signer or curr_position != prev_position:
                cap.set(cv2.CAP_PROP_POS_MSEC, start)
                person_bbox = self.person_detector.detect_in_frame(cap, padding=padding)

                cap.set(cv2.CAP_PROP_POS_MSEC, start)
                text_bbox = self.detect_text_in_segment(cap, start, person_bbox)

                prev_signer = curr_signer
                prev_position = curr_position

            output_path = os.path.join(output_dir, f"{video_name}-slice-{idx}.mp4")

            self.process_segment(
                cap,
                start,
                end,
                person_bbox,
                text_bbox,
                output_path,
                remove_text=remove_text,
            )

        cap.release()


def main():
    ann1 = "-9SqXjh8Y-I.json"
    video1 = os.path.join(VIDEOS_PATH, ann1.split(".json")[0] + ".mp4")
    out_dir = "datamining/CNSE/metadata/slices"
    os.makedirs(name=out_dir, exist_ok=True)

    if ANNOTATIONS_PATH is not None:
        try:
            yolo = YOLO("models/yolov8n.pt")

            east = TextDetectionModel_EAST("models/frozen_east_text_detection.pb")
            east.setConfidenceThreshold(0.5)
            east.setNMSThreshold(0.1)
            east.setInputParams(
                scale=1.0,
                size=(320, 320),
                mean=(123.68, 116.78, 103.94),
                swapRB=True,
                crop=False,
            )

            full_ann_path = os.path.join(ANNOTATIONS_PATH, ann1)

            processor = VideoProcessor(yolo=yolo, east=east)

            print(f"Processing video: {video1}")
            processor.process_video(video1, full_ann_path, out_dir)

            print("\n✅ Processing completed")

        except FileNotFoundError:
            print("❌ ERROR: Incorrect paths. Check your model files.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise


if __name__ == "__main__":
    main()
