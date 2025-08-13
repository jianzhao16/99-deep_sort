import os
import cv2
import numpy as np
import pandas as pd

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import visualization
from application_util import preprocessing

# === Configuration ===
#dataset_name = 'Fluo-N2DH-SIM+'
# dataset_name = 'Fluo-C2DL-Huh7'
dataset_name = 'Fluo-C2DL-MSC'
seq_num = '02'
local_dir = 'data/CTC'
#gt_json_path = f"{local_dir}/{dataset_name}/{seq_num}-gt_dict.json"
OUTPUT_DIR = f"{local_dir}/{dataset_name}/{seq_num}-tracking_results/"
#frame_diff = 1
#include_track_ids = True

IMAGE_DIR = f"{local_dir}/{dataset_name}/{seq_num}"                # Raw CTC images (optional, for visualization)
DETECTION_CSV =  f"{local_dir}/{dataset_name}/{seq_num}-detections.csv"
# Converted CSV with frame, x, y, w, h, conf
#OUTPUT_DIR = "tracking_results"      # Folder for annotated frames
RESULTS_FILE = "tracking_output.txt" # Final tracking output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Deep SORT Parameters ===
max_cosine_distance = 0.2
nn_budget = 100
min_confidence = 0.3
nms_max_overlap = 1.0
min_detection_height = 0
display = True

# === Load detection CSV ===
df = pd.read_csv(DETECTION_CSV)

# === Initialize Deep SORT ===
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

results = []

# === Optional: Map of image paths for visualization ===
image_filenames = {
    int(''.join(filter(str.isdigit, os.path.splitext(f)[0]))): os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")
}


# === Main tracking loop ===
for frame_idx in sorted(df["frame"].unique()):
    print(f"Processing frame {frame_idx}")

    # === Load detections for frame ===
    frame_df = df[df["frame"] == frame_idx]
    bboxes = frame_df[["x", "y", "w", "h"]].values
    scores = frame_df["conf"].values

    # Wrap as Detection objects
    detections = [
        Detection(tlwh, score, feature=np.zeros((128,), dtype=np.float32))  # dummy feature
        for tlwh, score in zip(bboxes, scores)
        if tlwh[3] >= min_detection_height and score >= min_confidence
    ]

    boxes = np.array([d.tlwh for d in detections])
    confs = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, confs)
    detections = [detections[i] for i in indices]

    # === Predict + Update Tracker ===
    tracker.predict()
    tracker.update(detections)

    # === Optional: Load frame for visualization ===
    if display and frame_idx in image_filenames:
        image = cv2.imread(image_filenames[frame_idx])
        vis = visualization.NoVisualization({})
        vis.set_image(image.copy())
        vis.draw_detections(detections)
        vis.draw_trackers(tracker.tracks)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{frame_idx:03d}.png"), vis.image)

    # === Save results for this frame ===
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        results.append([
            frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]
        ])

# === Write output like MOT format ===
with open(RESULTS_FILE, 'w') as f:
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

print(f"\n✅ Tracking complete. Results saved to: {RESULTS_FILE}")
