import json
import os

import cv2

total_duration: float = 0.0


def video_metadata(video: str) -> dict:
    cap = cv2.VideoCapture(filename=video)
    global total_duration

    if not cap.isOpened():
        print("Error al abrir el video")
        return {}

    fps: float = cap.get(cv2.CAP_PROP_FPS)
    frames: float = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width: float = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height: float = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if fps > 0:
        duration: float = frames / fps
        total_duration += duration
    else:
        print(f"Error con el fichero de video {video}")
        return {}

    cap.release()
    filename = video.split("/")[-1]
    return {
        "filename": filename,
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


# videos = [
#     os.path.join("datamining/CNSE/videos", video)
#     for video in os.listdir("datamining/CNSE/videos")
#     if video.endswith(".mp4")
# ]
# metadata = [video_metadata(video) for video in videos]
# metadata = sorted(metadata, key=lambda x: -x["duration"])
# metadata = {"total_duration": total_duration, "info": metadata}

# print(total_duration)

# with open("datamining/CNSE/metadata/metadata.videos.json", "w") as f:
#     json.dump(metadata, f, indent=3)
