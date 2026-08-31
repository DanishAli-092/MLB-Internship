# Day 29 - Object Tracking with YOLO

**MLB Summer Internship | Danish Ali**

**Streamlit App URL:** [View Live Application Here](https://mlb-internship-danish-ali-day-29-object-tracking.streamlit.app/)

## What is Object Tracking?

Object tracking is the process of following the same object across multiple frames of a video and assigning it a persistent identity (an ID) that stays the same as long as that object is visible.

## Detection vs Tracking

- Detection looks at a single frame in isolation and answers "what objects are here, and where?" It has no memory of previous frames.
- Tracking builds on detection by linking detections across frames over time, so the same physical object keeps the same ID even as it moves, gets partially occluded, or changes size/angle.

## Tracking Algorithm Used

This project uses ByteTrack (with BoT-SORT + ReID available as an alternate option in the app) via Ultralytics' built-in `model.track()` API. Both configs are tuned beyond their defaults - see Challenges below for why.

## Challenges Faced

**Missing tracker config.** The BoT-SORT + ReID variant needed a config file (`botsort_reid.yaml`) that Ultralytics doesn't ship by default - only `bytetrack.yaml` and a plain `botsort.yaml` exist out of the box. Referencing the ReID variant directly crashed `model.track()` with a `FileNotFoundError`. Fixed by generating the config at runtime from the base `botsort.yaml`, with `with_reid` enabled.

**IDs switching even without overlap.** Two things caused this: the default `track_buffer` (30 frames) expired a track too quickly on any brief miss - motion blur, a partially occluded frame, a couple of low-confidence detections - so the object got a brand new ID the moment it reappeared. And ByteTrack only matches by motion/IoU, not appearance, so it has nothing to fall back on when a match is ambiguous. Fixed by raising `track_buffer` to 60 frames on both trackers, and additionally enabling `with_reid` (appearance matching) and `gmc_method: sparseOptFlow` (camera-motion compensation) on the BoT-SORT config.

**Tracked output video wouldn't play.** Processing completed with no errors, correct object counts, and a correct confidence table - but the saved video showed 0:00 duration and wouldn't play in the browser. Cause: `cv2.VideoWriter` with the `mp4v` codec silently produced an empty/corrupt file on this setup, with no exception raised, so the failure was invisible until playback. Fixed by removing `cv2.VideoWriter` entirely and piping annotated frames directly to `ffmpeg` (H.264) via `imageio-ffmpeg`, which also removed the need for a separate re-encode pass.

**Confidence threshold trade-off.** Too low a threshold introduces flicker and false detections that create spurious new IDs; too high a threshold causes missed detections and dropped tracks. Settled on 0.6 as the default after testing.

## Files

- `tracking_script.py` - batch script that runs tracking on all videos in `sample_videos/` and saves annotated output to `output_videos/`.
- `app.py` - Streamlit app for interactive tracking; supports both uploading a video and picking one from `sample_videos/`.
- `requirements.txt` - dependencies.

## How to Run

```bash
pip install -r requirements.txt
python tracking_script.py        # batch processing
streamlit run app.py             # interactive app
```