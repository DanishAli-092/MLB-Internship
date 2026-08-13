
# Day 20 - Challenge Task (Mandatory)Processes 3 different videos, saves both original and processedversions separately, and prints a basic comparison of results(edge density, average frame brightness) for the README write-up.

import cv2
import os
import shutil
import numpy as np
from mini_project import VideoProcessor
def compute_edge_density(edges_frame: np.ndarray) -> float:
    """
    Returns the fraction of pixels detected as edges in a single
    processed (Canny) frame. Higher value = more detail/motion/noise.
    """
    gray_edges = cv2.cvtColor(edges_frame, cv2.COLOR_BGR2GRAY)
    edge_pixels = np.count_nonzero(gray_edges)
    total_pixels = gray_edges.size
    return edge_pixels / total_pixels

def process_and_compare(video_paths: list, output_dir: str) -> dict:
    """
    Runs each video through the VideoProcessor pipeline, saves the
    original (copied) and processed video separately, and collects
    basic comparison metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    processor = VideoProcessor(blur_kernel=(5, 5), canny_thresholds=(100, 200))
    results = {}
    for idx, input_path in enumerate(video_paths, start=1):
        if not os.path.exists(input_path):
            print(f"Skipping missing file: {input_path}")
            continue
        print(f"\n--- Processing Video {idx}: {input_path} ---")
        # 1. Save a copy of the original video into the output folder
        original_dest = os.path.join(output_dir, f"original_video{idx}.mp4")
        shutil.copy(input_path, original_dest)
        # 2. Process and save the processed version
        processed_dest = os.path.join(output_dir, f"processed_video{idx}.mp4")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(processed_dest, fourcc, fps, (width, height))
        edge_density_values = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = processor.process_frame(frame)
            out.write(processed_frame)
            edge_density_values.append(compute_edge_density(processed_frame))
            frame_count += 1
        cap.release()
        out.release()
        avg_edge_density = (
            sum(edge_density_values) / len(edge_density_values)
            if edge_density_values else 0.0
        )
        results[f"video{idx}"] = {
            "source": input_path,
            "frames": frame_count,
            "fps": round(fps, 2),
            "resolution": f"{width}x{height}",
            "avg_edge_density": round(avg_edge_density, 4),
        }
        print(f"  Frames: {frame_count} | FPS: {fps:.2f} | "
              f"Avg Edge Density: {avg_edge_density:.4f}")
        print(f"  Original saved to : {original_dest}")
        print(f"  Processed saved to: {processed_dest}")
    return results
def generate_comparison_text(results: dict) -> str:
    """
    Compares all processed videos against each other using the collected
    metrics and generates a human-readable comparison summary.
    """
    if not results:
        return "No videos were processed."
    lines = ["=== Challenge Task: Comparison of Results ===\n"]
    # Individual stats
    for key, data in results.items():
        lines.append(
            f"{key} ({data['source']}): "
            f"{data['frames']} frames, {data['fps']} FPS, "
            f"{data['resolution']}, avg edge density = {data['avg_edge_density']}"
        )
    # Find video with highest and lowest edge density
    sorted_by_density = sorted(
        results.items(), key=lambda x: x[1]["avg_edge_density"], reverse=True
    )
    highest = sorted_by_density[0]
    lowest = sorted_by_density[-1]
    lines.append("\n--- Comparison ---")
    lines.append(
        f"{highest[0]} had the HIGHEST average edge density "
        f"({highest[1]['avg_edge_density']}), meaning it likely contains "
        f"more texture, detail, or motion between frames."
    )
    lines.append(
        f"{lowest[0]} had the LOWEST average edge density "
        f"({lowest[1]['avg_edge_density']}), suggesting a simpler scene, "
        f"less motion, or smoother/flatter surfaces."
    )
    # Resolution/FPS comparison
    resolutions = {k: v["resolution"] for k, v in results.items()}
    fps_values = {k: v["fps"] for k, v in results.items()}
    lines.append(
        f"\nResolutions across videos: {resolutions}"
    )
    lines.append(
        f"FPS across videos: {fps_values} "
        f"(different FPS values affect how smooth the saved output looks)."
    )
    lines.append(
        "\nNote: Edge density is an automated proxy for visual complexity. "
        "Watch the actual processed videos in output/challenge/ and add your "
        "own qualitative observations below (e.g. lighting conditions, "
        "camera shake, background clutter) in the README."
    )
    return "\n".join(lines)
if __name__ == "__main__":
    challenge_videos = [
        "data/challenge_video1.mp4",
        "data/challenge_video2.mp4",
        "data/challenge_video3.mp4",
    ]
    output_directory = "output/challenge"
    comparison_results = process_and_compare(challenge_videos, output_directory)
    comparison_text = generate_comparison_text(comparison_results)
    print("\n" + comparison_text)
    # Save comparison text to a file so it can be pasted into README
    with open(os.path.join(output_directory, "comparison_summary.txt"), "w") as f:
        f.write(comparison_text)
    print(f"\nComparison summary saved to: {output_directory}/comparison_summary.txt")
