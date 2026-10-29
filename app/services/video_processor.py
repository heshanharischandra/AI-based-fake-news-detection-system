import subprocess
from pathlib import Path
import cv2


def extract_audio(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(output_path)
    ]

    subprocess.run(command, check=True)

    return str(output_path)



    import cv2


def extract_frames(video_path, output_folder, interval_seconds=5):
    video_path = str(video_path)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise ValueError("Invalid video FPS.")

    frame_interval = int(fps * interval_seconds)

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            filename = output_folder / f"frame_{saved_count:03d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    return saved_count