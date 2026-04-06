# ⚽ Football Match Tracker & Highlights Processor

An advanced computer vision project to process football match videos, track players, classify teams based on jersey colors, and smartly track the football using a custom-trained YOLOv8m model.

## Key Features
* **Custom Object Detection:** Trained a custom YOLOv8m model specifically for detecting football players and the ball.
* **Team Classification:** Uses **K-Means Clustering** on player bounding boxes (filtering out the green grass using HSV masking) to identify the dominant jersey color and classify players into two teams (Red vs. Sky Blue).
* **Smart Ball Tracking:** Implemented distance-based heuristics (Pixel Jump & Lost Patience logic) to prevent the tracker from jumping to false positives, ensuring smooth ball tracking.
* **Custom Visualizations:** Instead of standard bounding boxes, the script draws clean 3D-style ellipses at the players' feet and a highlighted triangle marker over the ball for a professional broadcast look.
* **Modern Package Management:** Uses `uv` for lightning-fast dependency management.
* **Automated Sourcing:** Pipeline optimized to process videos downloaded directly via `yt-dlp`.

---

## Dataset & Model Performance
The core detection model is fine-tuned on **YOLOv8m** using a custom online dataset consisting of **663 annotated images**. 

**Model Evaluation Metrics:**
* **Precision:** `96.95%`
* **Recall:** `77.32%`
* **mAP@50 (Mean Average Precision):** `85.64%`

---

## Tech Stack & Requirements
* **Language:** Python
* **Package Manager:** [uv](https://github.com/astral-sh/uv)
* **Libraries:** `ultralytics` (YOLOv8), `opencv-python` (cv2), `scikit-learn` (KMeans), `numpy`, `yt-dlp`

---

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast and automated dependency management.

**1. Clone the repository:**
```bash
git clone [https://github.com/starvipin/Multi-Object-Detection-and-Persistent-ID-Tracking-in-Public-SportsEvent-Footage.git]
cd Multi-Object-Detection-and-Persistent-ID-Tracking-in-Public-SportsEvent-Footage
```
**2. Install uv (Fast Package Manager):**
```bash
pip install uv
```
**3. Sync Dependencies: Automatically install all required packages using uv:**
```base
uv sync
```

**4. Now, navigate to the **src folder**, select the "Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage" kernel, and run the .ipynb notebook.**