# Day 28 - Custom PPE Detection System

**MLB Summer Internship | Danish Ali**

## Dataset

**Selected dataset:** [PPE Detection](https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb) (Roboflow Universe, workspace `sdp-lfigk`, version 13) 

**Classes (6):** Gloves, Hard_hat, Mask, Person, Safety_boots, Vest

**Why this dataset:** It was the dataset assigned for Day 29 (PPE Detection). It provides a clean YOLOv8-format export with train/valid/test splits already prepared, and covers the core PPE items relevant to a construction-site safety monitoring use case.

**Dataset split:**
- Train / Valid / Test splits as provided by the Roboflow export
- 425 validation images, 1794 validation instances across all 6 classes

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | YOLOv8s (`yolov8s.pt`) |
| Epochs | 60 |
| Batch size | 16 |
| Image size | 640 |
| Patience (early stopping) | 15 |
| Optimizer | Default (Ultralytics auto) |

## Final Evaluation Metrics

**Final deployed model: YOLOv8s baseline (no augmentation)**

| Metric | Value |
|---|---|
| mAP@50 | **89.44%** |
| mAP@50-95 | 62.55% |
| Precision | 86.4% |
| Recall | 87.3% |

**Per-class mAP@50:**

| Class | mAP@50 |
|---|---|
| Hard_hat | 97.7% |
| Vest | 95.8% |
| Person | 94.7% |
| Safety_boots | 86.1% |
| Gloves | 82.3% |
| Mask | 80.0% |

> **Performance Target: mAP@50 ≥ 80% — ACHIEVED.**
> Final mAP@50 = **89.44%**, exceeding the required 80% target by 9.44 percentage points.
>
> Note: mAP@50-95 (62.55%) is a separate, much stricter metric (averaged across IoU thresholds 0.5–0.95) and is reported for completeness only — it was not part of the assignment's performance target.

## Challenges Faced and Improvement Experiments

While overall accuracy comfortably met the target, per-class results showed **Mask** and **Gloves** were the weakest-performing classes. Both are small, frequently occluded objects (face masks partially covered by other gear, gloves at varying hand angles), which made them harder to detect consistently — especially at different angles or in crouching/occluded poses.

To investigate whether this was a model-capacity issue or a data issue, three configurations were trained and compared:

| Run | Model | Augmentation | Overall mAP@50 | Overall mAP@50-95 | Gloves mAP@50 | Mask mAP@50 | Training Time |
|---|---|---|---|---|---|---|---|
| **v1** | YOLOv8s | None (baseline) | **89.44%** | 62.55% | 82.3% | **80.0%** | ~30 min |
| v2 | YOLOv8s | hsv, mosaic, mixup, copy_paste | 89.72% | 61.88% | 83.6% | 79.6% | ~30 min |
| v3 | YOLOv8m | hsv, mosaic, mixup, copy_paste | 89.78% | 62.18% | **85.5%** | 78.6% | ~63 min |

**Observations:**
- Augmentation (v2) gave only a marginal overall improvement (+0.28% mAP@50) over baseline, and did not improve the Mask class — it slightly declined (80.0% → 79.6%).
- Switching to a larger model, YOLOv8m (v3), gave the largest gain on Gloves (82.3% → 85.5%), suggesting that class benefited from extra model capacity. However, the Mask class got *worse* (80.0% → 78.6%), and training time roughly doubled (30 min → 63 min) with model size more than doubling (22.5MB → 52MB).
- This pattern suggests the Mask class's weakness is **not** primarily a model-capacity or augmentation problem, but more likely a **data-level issue** — limited variety in mask angles/occlusion scenarios within the training set.

**Decision:** The baseline YOLOv8s model (v1) was selected for final deployment. The accuracy difference across all three runs was small (within ~0.3%), while v1 offers the smallest file size and fastest inference — a better fit for a Streamlit deployment than the marginal gains from augmentation or a larger model.

**Further testing** on real-world images also showed:
- Reduced detection confidence on occluded or unusually angled poses (e.g., crouching workers), consistent with the training data having limited pose variety.
- The model correctly does not detect out-of-domain PPE types (e.g., medical/healthcare gear) that weren't part of the training classes — expected behavior given the trained class scope, not a defect.

**Possible future improvements:**
- Add more Mask/Gloves-specific training images covering varied angles and occlusion levels.
- Increase `imgsz` to 832/960 to give small objects more resolution.
- Review label quality on existing Mask annotations for possible mislabeling.

## Deployment

A Streamlit app (`app.py`) allows uploading an image or video and running inference with the trained model:
- Adjustable confidence threshold (default 0.30)
- Custom bounding-box/label rendering that avoids overlapping labels
- Interactive detection summary chart (count and average confidence per class)
- Video support for `.mp4`, `.avi`, and `.mov`, with automatic re-encoding for browser playback
- Download buttons for both processed images and videos

## Deliverables

- Training Notebook: `train.ipynb`
- Trained Model: `models/best.pt`
- Streamlit Application: `app.py`
- Requirements: `requirements.txt`
- Sample Test Images: `Sample Test Images/`
- Prediction Results: `Prediction Results/`


