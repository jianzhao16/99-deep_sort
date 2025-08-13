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

IMAGE_DIR = f"{local_dir}/{dataset_name}/{seq_num}_Msk_CSTQ"                # Raw CTC images (optional, for visualization)
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
#metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)

tracker = Tracker(metric)

results = []

# === Optional: Map of image paths for visualization ===
image_filenames = {
    int(''.join(filter(str.isdigit, os.path.splitext(f)[0]))): os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")
}




def _read_detections_any(detection_file):
    """
    Supports:
      - .npy in MOT format: [frame, id, x, y, w, h, conf, a1, a2, a3, feat...]
      - .csv in CTC format: frame,x,y,w,h,conf,(class),(track)  -> converted to MOT-like (no features)
    Returns ndarray or None.
    """
    import os
    import numpy as np
    import pandas as pd

    if detection_file is None:
        return None

    ext = os.path.splitext(detection_file)[1].lower()
    if ext == ".npy":
        return np.load(detection_file)
    elif ext == ".csv":
        df = pd.read_csv(detection_file)
        # Ensure required columns exist
        for c in ["frame", "x", "y", "w", "h"]:
            if c not in df.columns:
                raise ValueError(f"CSV missing column '{c}'")

        # Fallbacks
        if "conf" not in df.columns:
            df["conf"] = 1.0

        # Build MOT-like 10-column matrix (no features):
        # [frame, id=-1, x, y, w, h, conf, -1, -1, -1]
        mot = np.zeros((len(df), 10), dtype=np.float32)
        mot[:, 0] = df["frame"].astype(np.int32).to_numpy()
        mot[:, 1] = -1
        mot[:, 2] = df["x"].to_numpy()
        mot[:, 3] = df["y"].to_numpy()
        mot[:, 4] = df["w"].to_numpy()
        mot[:, 5] = df["h"].to_numpy()
        mot[:, 6] = df["conf"].to_numpy()
        mot[:, 7:] = -1
        return mot
    else:
        raise ValueError(f"Unsupported detections file type: {ext}")


def _build_image_map(sequence_dir):
    """
    Works for:
      - MOT: images in sequence_dir/img1 with numeric names (000001.jpg)
      - CTC: images directly in sequence_dir (*.tif, *.png, *.jpg)
    Returns: (image_filenames_dict, image_size_tuple_or_None)
    """
    import os, re, cv2

    # Prefer MOT-style if present
    mot_img_dir = os.path.join(sequence_dir, "img1")
    if os.path.isdir(mot_img_dir):
        image_dir = mot_img_dir
        numeric_key = True
    else:
        # CTC-style: images are directly under sequence_dir
        image_dir = sequence_dir
        numeric_key = False  # need to extract numbers robustly

    if not os.path.isdir(image_dir):
        return {}, None

    # Gather images
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    if not files:
        return {}, None

    def _frame_index_from_name(name):
        # Try entire stem as int first (MOT case), else extract last digit-run
        stem = os.path.splitext(name)[0]
        try:
            return int(stem)
        except ValueError:
            m = re.findall(r"(\d+)", stem)
            return int(m[-1]) if m else 0

    frames = {}
    for f in files:
        idx = _frame_index_from_name(f) if not numeric_key else int(os.path.splitext(f)[0])
        frames[idx] = os.path.join(image_dir, f)

    # Infer image size from one sample
    sample = cv2.imread(next(iter(frames.values())), cv2.IMREAD_GRAYSCALE)
    image_size = sample.shape if sample is not None else None
    return frames, image_size


def gather_sequence_info(sequence_dir, detection_file):
    """
    Universal: supports MOTChallenge layout OR a plain CTC-style folder.
    - Builds image_filenames map
    - Loads detections from .npy (MOT) or .csv (CTC) if provided
    - Fills MOT-like keys required by visualization.* classes
    """
    import os, numpy as np, cv2

    image_filenames, image_size = _build_image_map(sequence_dir)

    # ground truth (only if classic MOT layout)
    gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
    groundtruth = None
    if os.path.exists(gt_file):
        groundtruth = np.loadtxt(gt_file, delimiter=',')

    # detections (supports .npy or .csv)
    detections = _read_detections_any(detection_file)

    if len(image_filenames) > 0:
        min_frame_idx = int(min(image_filenames.keys()))
        max_frame_idx = int(max(image_filenames.keys()))
    else:
        # fall back to detection frames if images missing
        if detections is not None and detections.size > 0:
            min_frame_idx = int(detections[:, 0].min())
            max_frame_idx = int(detections[:, 0].max())
        else:
            min_frame_idx = 0
            max_frame_idx = 0

    # seqinfo.ini only exists for MOT; otherwise default to ~50 FPS -> 20 ms/frame
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)
        try:
            update_ms = 1000 / int(info_dict.get("frameRate", 50))
        except Exception:
            update_ms = 20
    else:
        update_ms = 20  # good default for our visualizer

    feature_dim = int(detections.shape[1] - 10) if detections is not None else 0

    return {
        "sequence_name": os.path.basename(sequence_dir.rstrip("/")),
        "image_filenames": image_filenames,   # {frame_idx: path}
        "detections": detections,             # ndarray or None
        "groundtruth": groundtruth,           # ndarray or None
        "image_size": image_size,             # (H, W) or None
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,           # 0 if no features
        "update_ms": update_ms
    }


# === Main tracking loop ===
for frame_idx in sorted(df["frame"].unique()):
    print(f"Processing frame {frame_idx}")

    # === Load detections for frame ===
    frame_df = df[df["frame"] == frame_idx]
    print(frame_idx, frame_df.shape)

    bboxes = frame_df[["x", "y", "w", "h"]].values
    scores = frame_df["conf"].values

    print('detecting', len(bboxes), 'bboxes')
    print('----------')
    print('begin detection')
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
    print(detections)
    print('----------')
    print('begin tracking')
    # === Predict + Update Tracker ===
    tracker.predict()
    tracker.update(detections)
    print('end tracking')
    print('----------')
    # === Optional: Load frame for visualization ===
    def _draw_frame(image, detections, tracks):
        out = image.copy()

        # draw detections (green)
        for d in detections:
            x, y, w, h = map(int, d.tlwh)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # draw confirmed tracks (blue + ID)
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            x, y, w, h = map(int, t.to_tlwh())
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(out, f"ID {t.track_id}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        return out


    # --- inside your loop ---
    if display and frame_idx in image_filenames:
        image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_UNCHANGED)
        vis_img = _draw_frame(image, detections, tracker.tracks)
        out_path = os.path.join(OUTPUT_DIR, f"{frame_idx:03d}.png")
        ok = cv2.imwrite(out_path, vis_img)
        if not ok:
            print(f"[WARN] Failed to write {out_path}")

    # if display and frame_idx in image_filenames:
    #     image = cv2.imread(image_filenames[frame_idx])
    #     seq_info = gather_sequence_info(IMAGE_DIR, DETECTION_CSV)
    #     vis = visualization.NoVisualization(seq_info)
    #     vis.set_image(image.copy())
    #     vis.draw_detections(detections)
    #     vis.draw_trackers(tracker.tracks)
    #     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{frame_idx:03d}.png"), vis.image)

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
