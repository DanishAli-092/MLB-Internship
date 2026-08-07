"""
Day 16 - Mini Project: Image Processing Toolkit
Menu-driven OpenCV application.
"""

import cv2
import os
import numpy as np
from datetime import date


class ImageToolkit:
    """A simple menu-driven image processing toolkit built on OpenCV."""

    def __init__(self, output_dir="../output_images/toolkit"):
        self.image = None
        self.original_image = None
        self.image_path = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------- Core Operations ----------

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            print(f"Error: could not load image from '{path}'. Check the path.")
            return False

        self.image = img
        self.original_image = img.copy()
        self.image_path = path
        print(f"Loaded image: {path} | shape: {img.shape}")
        return True

    def to_grayscale(self):
        if self._no_image_loaded():
            return
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print("Converted to grayscale.")

    def resize(self, width, height):
        if self._no_image_loaded():
            return
        self.image = cv2.resize(self.image, (width, height))
        print(f"Resized to {width}x{height}.")

    def rotate(self, angle):
        if self._no_image_loaded():
            return

        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        if angle not in rotation_map:
            print("Invalid angle. Choose from 90, 180, 270.")
            return

        self.image = cv2.rotate(self.image, rotation_map[angle])
        print(f"Rotated by {angle} degrees.")

    def flip(self, direction):
        if self._no_image_loaded():
            return

        flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
        if direction not in flip_codes:
            print("Invalid direction. Choose: horizontal, vertical, both.")
            return

        self.image = cv2.flip(self.image, flip_codes[direction])
        print(f"Flipped: {direction}.")

    def crop(self, x1, y1, x2, y2):
        if self._no_image_loaded():
            return

        h, w = self.image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x1 >= x2 or y1 >= y2:
            print("Invalid crop coordinates.")
            return

        self.image = self.image[y1:y2, x1:x2]
        print(f"Cropped region: ({x1},{y1}) to ({x2},{y2}).")

    def draw_shape(self, shape_type, **kwargs):
        if self._no_image_loaded():
            return

        color = kwargs.get("color", (0, 255, 0))
        thickness = kwargs.get("thickness", 2)

        if shape_type == "rectangle":
            cv2.rectangle(self.image, kwargs["pt1"], kwargs["pt2"], color, thickness)
        elif shape_type == "circle":
            cv2.circle(self.image, kwargs["center"], kwargs["radius"], color, thickness)
        elif shape_type == "line":
            cv2.line(self.image, kwargs["pt1"], kwargs["pt2"], color, thickness)
        elif shape_type == "polygon":
            pts = np.array(kwargs["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.image, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            print("Unknown shape type.")
            return

        print(f"Drew {shape_type}.")

    def add_text(self, text, position, color=(255, 255, 255), scale=0.8, thickness=2):
        if self._no_image_loaded():
            return
        cv2.putText(self.image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness)
        print(f"Added text: '{text}'.")

    def adjust_brightness_contrast(self, brightness=0, contrast=0):
        """Bonus: brightness/contrast adjustment using cv2.convertScaleAbs."""
        if self._no_image_loaded():
            return

        alpha = 1 + (contrast / 100.0)   # contrast factor
        beta = brightness                 # brightness offset
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        print(f"Adjusted brightness={brightness}, contrast={contrast}.")

    def compare_bgr_rgb(self):
        """Bonus: convert to RGB and show side by side with BGR."""
        if self._no_image_loaded():
            return None

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        combined = np.hstack((self.image, rgb_image))
        return combined

    def show_side_by_side(self):
        """Bonus: display original vs processed image."""
        if self.original_image is None or self.image is None:
            print("No image loaded to compare.")
            return

        orig_resized = cv2.resize(self.original_image, (400, 400))

        # Handle grayscale processed image (1 channel) for hstack
        if len(self.image.shape) == 2:
            processed = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            processed = self.image

        processed_resized = cv2.resize(processed, (400, 400))
        combined = np.hstack((orig_resized, processed_resized))

        cv2.imshow("Original (Left) vs Processed (Right)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def reset(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            print("Image reset to original.")

    def save(self, filename=None):
        if self._no_image_loaded():
            return

        if filename is None:
            base = os.path.basename(self.image_path)
            filename = f"processed_{base}"

        out_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(out_path, self.image)
        print(f"Saved processed image: {out_path}")

    def display(self):
        if self._no_image_loaded():
            return
        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _no_image_loaded(self):
        if self.image is None:
            print("No image loaded yet. Please load an image first (option 1).")
            return True
        return False


def print_menu():
    print("\n===== IMAGE PROCESSING TOOLKIT =====")
    print("1.  Load image")
    print("2.  Convert to grayscale")
    print("3.  Resize image")
    print("4.  Rotate image")
    print("5.  Flip image")
    print("6.  Crop image")
    print("7.  Draw shape")
    print("8.  Add custom text")
    print("9.  Adjust brightness/contrast (bonus)")
    print("10. Compare BGR vs RGB (bonus)")
    print("11. Show original vs processed side by side (bonus)")
    print("12. Display current image")
    print("13. Reset to original")
    print("14. Save processed image")
    print("0.  Exit")


def main():
    toolkit = ImageToolkit()

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        try:
            if choice == "1":
                path = input("Enter image path: ").strip()
                toolkit.load_image(path)

            elif choice == "2":
                toolkit.to_grayscale()

            elif choice == "3":
                w = int(input("New width: "))
                h = int(input("New height: "))
                toolkit.resize(w, h)

            elif choice == "4":
                angle = int(input("Angle (90/180/270): "))
                toolkit.rotate(angle)

            elif choice == "5":
                direction = input("Direction (horizontal/vertical/both): ").strip()
                toolkit.flip(direction)

            elif choice == "6":
                x1 = int(input("x1: ")); y1 = int(input("y1: "))
                x2 = int(input("x2: ")); y2 = int(input("y2: "))
                toolkit.crop(x1, y1, x2, y2)

            elif choice == "7":
                shape = input("Shape (rectangle/circle/line/polygon): ").strip()
                if shape == "rectangle":
                    x1 = int(input("x1: ")); y1 = int(input("y1: "))
                    x2 = int(input("x2: ")); y2 = int(input("y2: "))
                    toolkit.draw_shape("rectangle", pt1=(x1, y1), pt2=(x2, y2))
                elif shape == "circle":
                    cx = int(input("center x: ")); cy = int(input("center y: "))
                    r = int(input("radius: "))
                    toolkit.draw_shape("circle", center=(cx, cy), radius=r)
                elif shape == "line":
                    x1 = int(input("x1: ")); y1 = int(input("y1: "))
                    x2 = int(input("x2: ")); y2 = int(input("y2: "))
                    toolkit.draw_shape("line", pt1=(x1, y1), pt2=(x2, y2))
                elif shape == "polygon":
                    n = int(input("Number of points: "))
                    points = []
                    for i in range(n):
                        px = int(input(f"Point {i+1} x: "))
                        py = int(input(f"Point {i+1} y: "))
                        points.append((px, py))
                    toolkit.draw_shape("polygon", points=points)
                else:
                    print("Unknown shape.")

            elif choice == "8":
                text = input("Enter text: ")
                x = int(input("x position: ")); y = int(input("y position: "))
                toolkit.add_text(text, (x, y))

            elif choice == "9":
                b = int(input("Brightness (-100 to 100): "))
                c = int(input("Contrast (-100 to 100): "))
                toolkit.adjust_brightness_contrast(b, c)

            elif choice == "10":
                combined = toolkit.compare_bgr_rgb()
                if combined is not None:
                    cv2.imshow("BGR (Left) vs RGB (Right)", combined)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            elif choice == "11":
                toolkit.show_side_by_side()

            elif choice == "12":
                toolkit.display()

            elif choice == "13":
                toolkit.reset()

            elif choice == "14":
                fname = input("Filename to save as (Enter for default): ").strip()
                toolkit.save(fname if fname else None)

            elif choice == "0":
                print("Exiting toolkit. Goodbye!")
                break

            else:
                print("Invalid choice. Try again.")

        except ValueError:
            print("Invalid input — please enter numbers where required.")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
    # Default name/date text example on save:
    # today = date.today().strftime("%Y-%m-%d")