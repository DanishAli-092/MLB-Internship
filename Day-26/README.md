# Day 26 - Introduction to Image Segmentation

**MLB Summer Internship | Danish Ali**

## What is Image Segmentation?

Image segmentation is the process of dividing an image into meaningful regions
(sets of pixels) so that each pixel can be assigned to a specific object or
category. Unlike object detection, which only draws a bounding box around an
object, segmentation identifies the exact pixels that belong to that object -
giving a precise shape/boundary instead of a rough rectangle.

## Features

- Binary, Adaptive, and Otsu thresholding
- Foreground/background removal (contour-based)
- Watershed segmentation for touching/overlapping objects
- Side-by-side comparison grid with auto-picked "best" result
- Batch processing script for running on multiple images at once
- Streamlit web app for interactive use with live preview and downloads

## Binary vs Adaptive vs Otsu Thresholding

| Method | How it works | Best used when |
|---|---|---|
| Binary | Uses one fixed threshold value for the whole image | Lighting is uniform and consistent |
| Adaptive | Calculates a local threshold for small regions of the image | Lighting is uneven or shadows are present |
| Otsu | Automatically calculates the optimal global threshold from the image histogram | Image has a clear bimodal histogram (distinct foreground/background) |

## Which method worked best for this dataset and why

Running `scripts/run_batch.py` on the 15-image dataset gave the following
best-method breakdown:

| Image | Otsu threshold | Best method | Score |
|---|---|---|---|
| img1.jpg | 123.0 | binary | 0.58 |
| img2.jpg | 154.0 | otsu | 0.95 |
| img3.png | 151.0 | adaptive | 0.16 |
| img4.jpg | 97.0 | binary | 0.47 |
| img5.jpg | 31.0 | binary | 0.00 |
| img6.png | 165.0 | adaptive | 0.90 |
| img7.png | 104.0 | binary | 0.55 |
| img8.jpg | 180.0 | otsu | 0.92 |
| img9.jpg | 215.0 | otsu | 0.45 |
| img10.jpg | 181.0 | otsu | 0.55 |
| img11.jpg | 171.0 | otsu | 0.70 |
| img12.jpg | 161.0 | adaptive | 0.05 |
| img13.jpg | 147.0 | otsu | 0.21 |
| img14.jpg | 150.0 | adaptive | 0.13 |
| img15.jpg | 115.0 | adaptive | 0.18 |

Three clear patterns emerged:

**1. Otsu wins on plain white-background product photos.** Images 8-11
(apple, banana, car, tree - all shot on a plain white background) were
dominated by Otsu, with the apple scoring the highest overall (0.92). White
backgrounds create a clean bimodal histogram (one peak for the background,
one for the object), which is exactly the condition Otsu's automatic
threshold-picking is built for.

**2. Binary wins on document/paper photos with uneven lighting**, but only
when there's still enough of a light/dark split to work with. Images 1, 4,
and 7 (photographed documents with shadows or fabric-like backgrounds) scored
best with binary. Image 5 (a near-blank page) scored 0.00 regardless of
method - the page had almost no dark content, so its foreground ratio fell
outside the 2%-98% range the scoring function treats as a valid
segmentation.

**3. Colored (non-white/non-gray) backgrounds were the hardest case across
all three methods.** Images 12-15 (plate on green, bulb on green, calculator
on blue, orange on blue) all scored low (0.05-0.21) no matter which method
won. This is expected: binary, adaptive, and Otsu all threshold on grayscale
*brightness*, and a colored background can be nearly as bright as the
object sitting on it - so brightness alone can't separate them cleanly. This
is exactly why `segmentation.py`'s `remove_background()` also tries a
saturation-channel mask (colored objects have far higher saturation than a
white/gray background) and keeps whichever mask - grayscale or
saturation-based - scores better.

**Overall**: Otsu was the best-performing method most often (6/15 images) and
by the widest margin on suitable images, adaptive thresholding was the
fallback for shadowed or colored-background images, and binary only won on
document-style photos with a clear light/dark split.

## Challenges faced during implementation

- **Colored backgrounds broke plain grayscale thresholding.** As the batch
  results confirm (img12-15 all scored under 0.21), a colored background can
  match the object's brightness closely enough that no grayscale threshold
  separates them well. Adding a saturation-channel mask as a second
  candidate in `remove_background()`, and picking whichever mask scored
  higher, was needed to handle these cases.
- **Near-blank/low-content images broke the scoring function**, not the
  thresholding itself - img5 scored 0.00 across the board because its
  foreground ratio fell outside the 2%-98% "valid segmentation" range the
  scorer expects, even though the threshold output itself wasn't
  necessarily wrong.
- **Choosing a fixed block size for adaptive thresholding** was tricky - a
  small block size picked up too much noise, while a large block size
  smoothed over fine details of smaller objects. Settled on a middle-ground
  default that worked reasonably across most images.
- **Watershed over-segmentation** - on images with textured surfaces or busy
  backgrounds, the distance transform sometimes created more "sure
  foreground" markers than actual objects, splitting a single object into
  multiple regions. Cleaning up noise with morphological opening before
  computing markers helped reduce this.
- **Contour selection on noisy thresholded images** - when the threshold
  picked up background texture as foreground, `findContours` sometimes
  returned the background's contour instead of the actual object. Adding a
  corner-based auto-invert check (checking whether image corners came out
  white) helped catch and fix this automatically.
- **Keeping the Streamlit app responsive** with large uploaded images -
  resizing images to a max side length before processing avoided slow
  reruns on every UI interaction (e.g. moving the threshold slider).

## Project Structure

```
Day-26/
├── scripts/
│   ├── thresholding.py   # binary, adaptive, otsu thresholding functions
│   ├── segmentation.py   # background removal + watershed algorithm
│   ├── utils.py          # comparison grid + output saving helpers
│   └── run_batch.py      # runs all methods across sample_images/ dataset
├── sample_images/        # add at least 15 input images here
├── output_images/        # generated comparison outputs land here
├── app.py                # Streamlit deployment app
├── requirements.txt
└── README.md
```

## Requirements

```
opencv-python
numpy
matplotlib
streamlit
Pillow
```

Install with:
```
pip install -r requirements.txt
```

## How to run

1. Add at least 15 images to `sample_images/` (documents, plain-background
   objects, uneven lighting, and shadowed images).
2. Run the batch processing script:
   ```
   python scripts/run_batch.py
   ```
   This generates a comparison grid, individual outputs, and the best-scoring
   result for every image, saved to `output_images/`.
3. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```
   Upload an image, pick a segmentation method from the sidebar, and download
   the result.

## Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib
- Streamlit

