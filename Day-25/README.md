# Day 25 - Feature Detection & Feature Matching

**MLB Summer Internship | Danish Ali**

## What are Image Features?

An image feature is a distinctive, locally identifiable region of an image — like a corner, blob,
or edge junction — that can be reliably detected again even if the image is rotated, scaled, or
lit differently. A feature has two parts:

- **Keypoint**: the (x, y) location, plus scale and orientation.
- **Descriptor**: a numerical vector describing the local appearance around that keypoint, used to
  compare and match it against keypoints in another image.

Corners are preferred over flat regions or edges because they show intensity change in two
directions at once, which makes their location uniquely identifiable.

## Harris Corner Detection vs ORB

| | Harris Corner Detection | ORB |
|---|---|---|
| What it detects | Corner locations only | Keypoints + binary descriptors |
| Scale invariant | No | Limited (uses image pyramid) |
| Rotation invariant | No | Yes (oriented FAST + rotated BRIEF) |
| Descriptor for matching | None | 256-bit binary descriptor |
| Speed | Fast | Very fast |
| Best use case | Simple corner detection, calibration | Real-time matching, AR, mobile apps |

Harris works by checking how much the image intensity changes when a small window is shifted in
every direction — corners show a large change in all directions, edges only in one direction, and
flat regions show almost no change. It's great for finding corners but gives no descriptor, so it
can't be used to match keypoints between two images on its own.

ORB solves this by combining FAST (fast corner detection) with BRIEF (a binary descriptor built
from pixel intensity comparisons), and adds orientation so it stays reliable even when the image
is rotated. That's why ORB is used for the matching task in this project, not Harris.

## How Feature Matching Works

1. ORB detects keypoints and computes a binary descriptor for each one, in both images.
2. A **Brute-Force Matcher** compares every descriptor in image 1 against every descriptor in
   image 2 using Hamming distance (since ORB descriptors are binary), and keeps the closest match.
3. For better quality, a **KNN Matcher** (k=2) is used instead, returning the two closest matches
   per keypoint.
4. **Lowe's ratio test** filters out unreliable matches: a match is kept only if the best match's
   distance is significantly smaller than the second-best match's distance
   (`distance1 < 0.75 * distance2`). This removes ambiguous matches where the correct
   correspondence isn't clear.

Feature matching is used in image stitching/panoramas, object recognition, augmented reality
overlays (homography estimation), visual localization/SLAM, and tracking objects across video
frames.

## Best Performing Image Pair

Pair **4** (MacBook Pro — straight front-facing view vs. a slightly rotated 3/4 angle view)
produced the best matching results with **297 good matches** (KNN + Lowe's ratio test), far
ahead of the next closest pair (190). The laptop's screen bezel, keyboard keys, trackpad edges,
and menu bar icons gave a huge number of distinct, well-textured corner regions that stayed
consistent across the two viewpoints, and the rotation angle between the shots was moderate —
large enough to test ORB's rotation invariance, small enough that most of the object still
overlapped between both images.

Full results across all 10 pairs (good matches count, KNN + ratio test):

| Pair | Good Matches |
|------|--------------|
| pair4 (laptop, front vs angled) | 297 |
| pair1 (Coca-Cola bottle, front vs tilted) | 190 |
| pair2 | 117 |
| pair9 | 69 |
| pair6 | 20 |
| pair3 | 14 |
| pair8 | 5 |
| pair5 | 4 |
| pair7 | 1 |

Worst performing pairs (pair7, pair5, pair8) had very few good matches — likely due to large
viewpoint/lighting changes, low texture/repetitive patterns, or insufficient overlap between the
two images.

## Folder Structure

```
Day-25/
├── __pycache__/            # Python bytecode cache (auto-generated, ignore)
├── images/                 # Sample image pairs used for testing
│   ├── pair1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── pair2/
│   ├── ...
│   └── pair10/
├── output_images/          # Result images saved automatically when running the Streamlit app
├── outputs/                # Result images saved when running feature_detection.py /
│                            # feature_matching.py directly
├── screen_recording/
│   └── link.md             # Link to the demo screen recording
├── app.py                  # Streamlit app - upload two images, run detection & matching,
│                            # view results, download outputs
├── feature_detection.py    # Harris Corner Detection + ORB keypoint detection & visualization,
│                            # plus Harris vs ORB performance comparison (count + speed)
├── feature_matching.py     # ORB feature extraction with two matching modes: Brute-Force
│                            # (crossCheck) and KNN + Lowe's ratio test
├── requirements.txt        # Python dependencies
└── README.md                # This file
```

## Files

- `feature_detection.py` — Harris Corner Detection + ORB keypoint detection & visualization, plus a
  Harris vs ORB performance comparison (count + speed).
- `feature_matching.py` — ORB feature extraction with two matching modes: Brute-Force (crossCheck)
  and KNN + Lowe's ratio test, saves the matched visualization.
- `app.py` — Streamlit app: upload two images, choose matching method (KNN or Brute-Force), view
  matched features, keypoint/corner counts, a full Harris vs ORB comparison table, and download
  buttons for every result image.
- `images/pair1 ... pair10/` — sample image pairs used for testing.
- `outputs/` — result images saved when running the scripts directly.
- `output_images/` — result images saved automatically when running the Streamlit app (created on
  first run, not included in this submission until you run the app).
- `screen_recording/link.md` — link to the demo screen recording of the app in action.
- `requirements.txt` — Python dependencies required to run this project.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the standalone scripts (saves results to outputs/)
python feature_detection.py
python feature_matching.py

# Run the Streamlit app
streamlit run app.py
```

