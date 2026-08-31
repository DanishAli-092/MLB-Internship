import os
import shutil
import subprocess
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO
from ultralytics.utils import USER_CONFIG_DIR


MODEL_PATH = "yolov8n.pt"
INPUT_DIR = "sample_videos"
OUTPUT_DIR = "output_videos"
TRACKER = "botsort_reid.yaml"
CONFIDENCE_THRESHOLD = 0.3


# finds the base tracker config file
def _find_base_tracker_config(base_name: str) -> Path:
    base_config_path = USER_CONFIG_DIR.parent / "cfg" / "trackers" / base_name
    if base_config_path.exists():
        return base_config_path
    import ultralytics
    return Path(ultralytics.__file__).parent / "cfg" / "trackers" / base_name


# makes sure tracker config exists and builds tuned version if needed
def ensure_tracker_config(tracker_name: str) -> str:
    if tracker_name == "botsort_reid.yaml":
        custom_path = Path("botsort_reid.yaml")
        if custom_path.exists():
            return str(custom_path)
        with open(_find_base_tracker_config("botsort.yaml"), "r") as f:
            config = yaml.safe_load(f)
        config["with_reid"] = True
        config["gmc_method"] = "sparseOptFlow"
        config["track_buffer"] = 60
        with open(custom_path, "w") as f:
            yaml.safe_dump(config, f)
        print(f"[INFO] Created {custom_path} (BoT-SORT + ReID, tuned) from the default config.")
        return str(custom_path)

    if tracker_name == "bytetrack.yaml":
        custom_path = Path("bytetrack_tuned.yaml")
        if custom_path.exists():
            return str(custom_path)
        with open(_find_base_tracker_config("bytetrack.yaml"), "r") as f:
            config = yaml.safe_load(f)
        config["track_buffer"] = 60
        with open(custom_path, "w") as f:
            yaml.safe_dump(config, f)
        print(f"[INFO] Created {custom_path} (ByteTrack, tuned) from the default config.")
        return str(custom_path)

    return tracker_name


# finds ffmpeg path on system or bundled
def _get_ffmpeg_path():
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# runs tracking on one video and returns unique object count
def track_video(model: YOLO, video_path: str, output_path: str, tracker: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_path = _get_ffmpeg_path()
    if ffmpeg_path is None:
        print("[SKIP] ffmpeg not found (system or bundled) - add imageio-ffmpeg to requirements.txt")
        cap.release()
        return 0

    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)

    unique_ids = set()
    is_first_frame = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                tracker=tracker,
                conf=CONFIDENCE_THRESHOLD,
                persist=not is_first_frame,
                verbose=False,
            )
            is_first_frame = False

            result = results[0]
            annotated_frame = result.plot()

            if result.boxes is not None and result.boxes.id is not None:
                ids_in_frame = result.boxes.id.int().tolist()
                for obj_id in ids_in_frame:
                    unique_ids.add(obj_id)

            writer.send(annotated_frame.tobytes())
    finally:
        cap.release()
        writer.close()

    return len(unique_ids)


# runs tracking on all videos in the input folder
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tracker_config = ensure_tracker_config(TRACKER)
    model = YOLO(MODEL_PATH)

    video_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print(f"No videos found in '{INPUT_DIR}'. Add at least 5 short videos and re-run.")
        return

    summary = {}

    for video_file in video_files:
        input_path = os.path.join(INPUT_DIR, video_file)
        output_name = f"tracked_{Path(video_file).stem}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"Processing: {video_file}")
        try:
            total_unique = track_video(model, input_path, output_path, tracker_config)
            summary[video_file] = total_unique
            print(f"  -> Unique objects tracked: {total_unique}")
            print(f"  -> Saved to: {output_path}")
        except Exception as e:
            print(f"  -> ERROR processing {video_file}: {e}")

    print("\n--- Summary ---")
    for video, count in summary.items():
        print(f"{video}: {count} unique objects")


if __name__ == "__main__":
    main()