
#Day 20 - Coding PracticeReads a video, prints its properties, applies grayscale + Canny edge detection to each frame, and saves the processed output.


import cv2
import os


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Extract and return basic properties of a video capture object."""
    properties = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return properties


    # Calculate the cv2.waitKey delay needed to play the video back at its natural speed. Falls back to a fixed delay if FPS is missing/invalid (some corrupted files or codecs report 0 or NaN).
    

def get_playback_delay(fps: float, fallback_ms: int = 25) -> int:
    
    if fps and fps > 0:
        return int(1000 / fps)
    return fallback_ms
     
      
        #  Reads a video frame by frame, converts to grayscale, applies Canny edge detection, displays it live at natural playback speed, and saves the processed output as a new video file.
         

def process_video(input_path: str, output_path: str) -> None:
   
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video file: {input_path}")

    props = get_video_properties(cap)
    print("Video Properties:")
    print(f"  FPS           : {props['fps']:.2f}")
    print(f"  Width         : {props['width']}")
    print(f"  Height        : {props['height']}")
    print(f"  Total Frames  : {props['total_frames']}")

    # Dynamic delay so cv2.imshow plays at the video's actual speed,
    # with a safe fallback if FPS metadata is missing or invalid
    
    delay = get_playback_delay(props["fps"])
    print(f"  Playback delay: {delay} ms")

    # VideoWriter setup - note the codec choice for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, props["fps"], (props["width"], props["height"])
    )

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        # Convert single-channel edges back to 3-channel so VideoWriter
        # (and imshow alongside color frames) works correctly
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Display original and processed frame side by side
        cv2.imshow("Original", frame)
        cv2.imshow("Processed (Canny Edges)", edges_bgr)

        out.write(edges_bgr)

        # Press 'q' to quit early
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("Processing interrupted by user.")
            break

    print(f"Total frames processed: {frame_count}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")


if __name__ == "__main__":
    INPUT_VIDEO = "data/sample_video.mp4"
    OUTPUT_VIDEO = "output/processed_video.mp4"

    process_video(INPUT_VIDEO, OUTPUT_VIDEO)