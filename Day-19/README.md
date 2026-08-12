# Day 19 - Contours in OpenCV

This is my submission for Day 19 of the ML Bench Summer Internship. Today's topic was **Contours** - how OpenCV finds the outline/boundary of objects in an image, and how we can use that to detect and measure simple shapes.

---

## 📁 Folder Structure

```
Day-19/
├── input_images/            # 10 images I created myself (dataset)
├── output_images/
│   ├── original/             # copy of originals (used in challenge task)
│   ├── contours/              # contour + bounding box + min enclosing circle results
│   └── shapes_labeled/        # final labeled shape detection results
├── challenge_output/
│   ├── image1/  ... image10/  # each has original.jpg, contours.jpg, shapes_labeled.jpg
├── screen_recording/
│   └── drive_link.md
├── contour_detection.py       # Task 1-6: grayscale, threshold, contours, area/perimeter, bounding box, min enclosing circle
├── shape_detection.py         # Task 7-8 + Mini Project: shape classification system
├── challenge.py                # Challenge Task: runs pipeline on 10 images, 3-way output
├── app.py                      # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 🔍 What are Contours?

In simple words, a contour is just the **outline of a shape** - like if you trace the edge of an object with a pencil, that traced line is the contour.

Technically, a contour is a curve that joins all the continuous points along a boundary that share the same color/intensity. OpenCV can only find contours properly on a **binary image** (pure black and white) - that's why the pipeline always goes: grayscale → threshold → find contours. If you skip straight from a normal color image to `findContours()`, it won't give you clean results.

Contours are useful because once you have the outline of an object, you can measure it (area, perimeter), draw boxes around it, or figure out what shape it roughly is - which is exactly what today's task needed.

---

## ⚙️ How Contour Detection Works (My Pipeline)

1. **Grayscale conversion** - `cv2.cvtColor()` to drop the color info, we only need brightness for thresholding.
2. **Gaussian Blur** - `cv2.GaussianBlur()` smooths out small noise/texture on the paper so it doesn't create tiny fake contours.
3. **Thresholding (Otsu's method)** - `cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)`. I didn't use a fixed value like 127 because my dataset has different backgrounds and lighting (a real photo, a gray background image, colored shapes on white, etc). Otsu automatically calculates the best black/white split based on each image's own histogram instead of me guessing one number for every image.
4. **`cv2.findContours()`** - finds the boundaries on the thresholded (binary) image.
5. **Filtering** - any contour with area smaller than `MIN_CONTOUR_AREA` gets ignored, since those are usually just leftover noise from thresholding, not real shapes.
6. **Measuring** - `cv2.contourArea()` for area, `cv2.arcLength()` for perimeter, `cv2.boundingRect()` for the bounding box, `cv2.minEnclosingCircle()` for the smallest circle that covers the shape.
7. **Classifying the shape** - `cv2.approxPolyDP()` reduces the contour down to its main corner points. Counting those corners tells me the shape:
   - 3 corners → Triangle
   - 4 corners → Square or Rectangle (checked using the width/height aspect ratio of the bounding box)
   - 5 corners → Pentagon
   - many corners / no clear corner count → checked using the **circularity formula** `(4 * π * Area) / Perimeter²`, which is close to 1.0 for a real circle and drops for anything angular or elongated

---

## 🔺 Shapes My Program Can Detect

- Circle
- Square
- Rectangle
- Triangle
- Pentagon
- General Polygon (fallback label for anything with more corners that isn't a circle)

---

## 🚧 Challenges I Faced

### 1. Building a proper self-made dataset
The task wanted 10-15 images made by me, not downloaded, so I made mine using a mix of MS Paint, PowerPoint, and one real hand-drawn photograph (on paper, photographed with my phone).

- **First attempt:** I drew a few images with thin, light-colored outlines (thin pink/yellow lines on a white background). When I ran my script on them, the contours weren't detecting properly - the outlines were too faint compared to the background for a clean threshold split.
- **Something I considered:** lowering `MIN_CONTOUR_AREA` or adding extra blur to try and "catch" the faint lines in code. I decided against this because it would make the threshold less reliable on my other, properly-drawn images, and it doesn't fix the actual problem (the outline genuinely has too little contrast against the background).
- **Final fix:** I went back and re-drew those specific shapes with **bold outlines and more saturated colors**. This solved it completely without touching a single line of code - the input image itself just needed to be clearer.

### 2. Square vs Rectangle getting misclassified
I was checking `if 0.95 <= aspect_ratio <= 1.05: Square`. One of my Paint-drawn squares kept getting labeled "Rectangle" even though it looked like a perfect square.

- **First thing I tried:** redrawing the square in Paint more carefully, holding the mouse steady, trying to get it pixel-perfect. It helped a little but still occasionally misfired, since a mouse-drawn shape is never going to be a mathematically exact 1:1 ratio.
- **Second thing I considered:** rounding the width and height to the nearest 10 pixels before computing the ratio, to "absorb" small mouse imprecision. I dropped this idea because it felt like a hacky patch rather than a real fix, and it could backfire on genuinely thin rectangles.
- **Final fix:** I loosened the aspect ratio range to **0.90–1.10**. This gives enough tolerance for normal hand-drawn imprecision, while my actual rectangles (which have a much bigger width/height gap) are nowhere close to that range, so they never get misclassified.

### 3. A rotated square looking like a "diamond"
One of my polygon-focus shapes was a square rotated 45° (visually looks like a diamond/rhombus).

- **What I noticed:** my program labels it "Square," not "Diamond" - which surprised me at first since visually it looks like a completely different shape.
- **What I considered doing:** adding a rotation-angle check using `cv2.minAreaRect()`'s angle output, so I could add a separate "Diamond" label for 4-corner shapes rotated close to 45°.
- **Why I left it as is:** mathematically, a rotated square is still a square - 4 equal sides, 4 right angles, just spun around. Today's requirement only asked to detect shape type (corners/geometry), not orientation. So I decided this is correct, expected behavior rather than a bug, and just noted it here instead of adding extra logic that wasn't actually asked for.

### 4. Real photograph noise
My real photograph (circle + rectangle, originally also had a pentagon) had a stray pen mark and a slight paper crease near the circle.

- **First attempt:** I tried increasing the Gaussian blur kernel size in code (from `(5,5)` to `(9,9)`) to smooth away the noise before thresholding. It reduced the noise a bit but also softened my bold shape outlines more than I liked, and the noise still occasionally reappeared as a small extra contour.
- **Second thing I considered:** manually filtering out contours with an unusually high perimeter-to-area ratio (since the noise line was long and thin). I decided this was too fragile - it depends on tuning yet another threshold value that might not generalize to other photos.
- **Final fix:** since this was a physical dataset issue and not something code should have to compensate for, I fixed it at the source - cleaned up the paper, drew the shapes boldly, used a flat surface with even lighting, and retook the photo. The pentagon didn't survive re-drawing clearly enough in that pass, so my final real photo has 2 clean shapes (circle, rectangle) instead of 3 - I'd rather have fewer shapes detected perfectly than more shapes with unreliable noise.

### 5. Colored shapes disappearing on a gray background (the trickiest one)
This was the toughest bug. My "Challenging/Mixed" image (`10_Mixed.jpg`) had 8 shapes with different colors on a **gray** background. With plain grayscale + Otsu thresholding, some shapes were getting completely missed, and in one run the entire background got picked up as a single giant "shape" instead of the actual 8 individual ones.

The reason: a couple of my shape colors, once converted to grayscale, ended up almost the same brightness as the gray background - so there wasn't enough contrast left for thresholding to separate them (this is called an **iso-luminance** problem - two different colors that map to the same grayscale intensity). When Otsu's automatic split picked a cutoff where the whole gray background counted as "foreground," everything merged into one blob.

- **Attempt 1 - HSV Saturation channel:** since gray has very low saturation and colored shapes have high saturation, I tried thresholding on the Saturation channel instead of grayscale. This actually fixed the colored shapes perfectly, but it broke something else - my black and dark-outlined shapes disappeared, because pure black also has zero saturation, so the algorithm was treating them as "background" too.
- **Attempt 2 - combining Saturation + Value channels:** I tried thresholding Saturation and Value (brightness) separately and merging the two results with `bitwise_or`, hoping to catch both colored shapes (via saturation) and dark shapes (via brightness) at once. This actually made things worse on some images - depending on how bright the gray background itself was, Otsu on the Value channel sometimes flipped and classified the whole background as foreground, so the combined mask became one giant blob again.
- **Attempt 3 - multi-channel Canny edge detection:** I skipped grayscale entirely, split the image into Blue/Green/Red channels, ran Canny edge detection on each channel separately, merged them with `bitwise_or`, and dilated the result to close small gaps. This technically caught every color correctly, but it introduced a brand new bug - circles started getting misclassified as polygons, because the edges from Canny + dilation came out slightly jagged/stair-stepped instead of smooth. That extra jaggedness inflated the calculated perimeter, which dropped my circularity score `(4 * π * Area) / Perimeter²` below the 0.8 cutoff, so a perfect circle was no longer being recognized as one.
- **Fixing the jagged edges:** to fix that specific circularity bug, I applied `cv2.convexHull(contour)` on every contour before measuring it. A convex hull basically wraps a tight, smooth boundary around the contour like stretching a rubber band around it, which removed the small stair-step jaggedness from the Canny + dilate result. Once the boundary was smooth again, the perimeter calculation normalized and my circles were correctly classified as circles again.
- **Where this ended up:** the multi-channel Canny + convexHull combination is what I actually used in `app.py` - I liked that it could handle any color/background combination automatically without me having to fix the input image every time, so I kept it there as a more "automatic" version of the pipeline. But for `contour_detection.py`, `shape_detection.py`, and `challenge.py`, I went with the simpler fix instead (see below), since those scripts are meant to closely follow the actual thresholding steps asked for in the task.
- **Final fix (used in the other 3 scripts):** I stepped back and fixed the gray-background problem the simplest possible way - directly in the dataset. I went back into `10_Mixed.jpg` and made the low-contrast shape colors **bolder/darker** so their grayscale brightness became clearly different from the gray background. That let the original, simple grayscale + Otsu pipeline detect all 8 shapes correctly - no extra channels, no edge detection, no dilation, no convex hull. The real fix was in the input image, not in extra code complexity.

---


## ▶️ How to Run

```powershell

python contour_detection.py
python shape_detection.py
python challenge.py

# for the Streamlit app:
streamlit run app.py
```

Output images get saved automatically into `output_images/` and `challenge_output/`.

## 💡 Lessons I Learned

- **Fix the data before you fix the code.** Almost every "detection bug" I hit today wasn't really a code bug  it was a low-contrast outline, a stray pen mark, or a color too close to the background. My first instinct was always to patch the code (more blur, a new color space, edge detection), but most of the time going back and improving the actual input image was the simpler and more reliable fix.
- **A more "clever" solution isn't automatically a better one.** The multi-channel Canny + convexHull pipeline technically handles more edge cases automatically, but it's a lot more moving parts than a simple grayscale + Otsu threshold, and it introduces its own new bugs (like the circle-to-polygon issue) that then need their own fixes. Simpler code that I fully understand and can explain beats a more powerful pipeline I can't fully reason about.
- **Otsu's method is genuinely worth knowing over a fixed threshold value.** Hardcoding something like `127` only works if you assume every image has the same lighting and background - which almost never holds true once you test on a real photograph or a colored background instead of clean digital shapes.
- **Tolerance ranges need to match how the data was actually created.** My square-vs-rectangle check kept failing until I remembered that hand-drawn/mouse-drawn shapes are never pixel-perfect. Tight thresholds work great on synthetic/mathematically generated data, but real (or hand-drawn) data needs some breathing room built in.
- **Understanding *why* something works matters more than just getting it to work.** I could have kept whichever approach passed the most test images without knowing why the others failed. Digging into the actual cause each time (iso-luminance, jagged edges inflating perimeter, non-perfect pixel ratios) is what actually helped me pick the right fix instead of randomly trying things until something stuck.


---

