# --- Add these imports near the top of deep_sort_app_v2.py ---
import os, re
import numpy as np
import pandas as pd
import cv2

from application_util import preprocessing, visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


# ---------------------------
# Helpers: images + detections
# ---------------------------
def _build_image_map(sequence_dir):
    """
    Supports:
      - MOT: images in <sequence_dir>/img1 (000001.jpg, etc.)
      - CTC: images live directly in <sequence_dir> (*.tif/*.png/*.jpg)
    Returns (image_filenames_dict, image_size_tuple_or_None).
    Keys are integer frame indices.
    """
    mot_img_dir = os.path.join(sequence_dir, "img1")
    if os.path.isdir(mot_img_dir):
        image_dir = mot_img_dir
        numeric_stem = True
    else:
        image_dir = sequence_dir
        numeric_stem = False

    if not os.path.isdir(image_dir):
        return {}, None

    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    if not files:
        return {}, None

    def frame_from_name(name):
        stem = os.path.splitext(name)[0]
        if numeric_stem:
            return int(stem)
        m = re.findall(r"(\d+)", stem)
        return int(m[-1]) if m else 0

    image_filenames = {frame_from_name(f): os.path.join(image_dir, f) for f in files}
    sample = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
    image_size = sample.shape if sample is not None else None
    return image_filenames, image_size


def _read_detections_any(detection_file):
    """
    - .npy (MOT-style): returns ndarray with features
    - .csv (CTC-style): returns pandas.DataFrame (frame,x,y,w,h,conf[,...])
    """
    if detection_file is None:
        return None
    ext = os.path.splitext(detection_file)[1].lower()
    if ext == ".npy":
        return np.load(detection_file)
    elif ext == ".csv":
        df = pd.read_csv(detection_file)
        # Required columns
        for c in ["frame", "x", "y", "w", "h"]:
            if c not in df.columns:
                raise ValueError(f"CSV missing column '{c}'")
        if "conf" not in df.columns:
            df["conf"] = 1.0
        df["frame"] = df["frame"].astype(int)
        return df
    else:
        raise ValueError(f"Unsupported detection file extension: {ext}")


def _create_detections_from_npy(detection_mat, frame_idx, min_height=0):
    """Original Deep SORT path (features included)."""
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx
    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def _create_detections_from_csv(df_by_frame, frame_idx, min_height=0, use_cosine=True):
    """
    CSV path: no appearance features available.
    If cosine metric is used, feed a unit dummy vector (avoid NaNs).
    If euclidean metric is used, zeros are fine.
    """
    frame_df = df_by_frame.get(int(frame_idx), None)
    if frame_df is None or len(frame_df) == 0:
        return []
    if use_cosine:
        dummy = np.zeros((128,), np.float32); dummy[0] = 1.0  # unit one-hot
        feat_fn = lambda: dummy
    else:
        feat_fn = lambda: np.zeros((128,), np.float32)

    dets = []
    for _, r in frame_df.iterrows():
        x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
        conf = float(r.get("conf", 1.0))
        if h < min_height:
            continue
        dets.append(Detection(np.array([x, y, w, h], dtype=np.float32), conf, feat_fn()))
    return dets


# ---------------------------
# Public API: run(...) like deep_sort_app.py
# ---------------------------
def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display,update_ms_interval):
    """
    Universal runner:
      - MOT mode: <sequence_dir>/img1 + .npy detections (with features)
      - CTC mode: <sequence_dir> images + .csv detections (no features)
    """
    # Build sequence info
    image_filenames, image_size = _build_image_map(sequence_dir)
    detections_any = _read_detections_any(detection_file)

    # Decide metric & feature handling
    use_cosine = True
    if isinstance(detections_any, pd.DataFrame):
        # CSV has no real appearance features; cosine would normalize zeros.
        # Either switch to euclidean OR keep cosine with unit dummy features.
        use_cosine = True  # keep cosine for compatibility (dummy unit features)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine" if use_cosine else "euclidean", max_cosine_distance, nn_budget
    )

    tracker = Tracker(metric)
    results = []

    # Prepare access by frame
    if isinstance(detections_any, np.ndarray):
        # MOT .npy
        min_frame = int(min(image_filenames.keys())) if image_filenames else int(detections_any[:, 0].min())
        max_frame = int(max(image_filenames.keys())) if image_filenames else int(detections_any[:, 0].max())
        frame_range = range(min_frame, max_frame + 1)
        det_source = "npy"
    elif isinstance(detections_any, pd.DataFrame):
        # CTC .csv
        groups = {k: v for k, v in detections_any.groupby("frame")}
        min_frame = int(min(image_filenames.keys())) if image_filenames else int(min(groups.keys()))
        max_frame = int(max(image_filenames.keys())) if image_filenames else int(max(groups.keys()))
        frame_range = range(min_frame, max_frame + 1)
        det_source = "csv"
    else:
        raise ValueError("detection_file must be a .npy (MOT) or .csv (CTC)")

    # Build a minimal seq_info dict for visualization.* (so .Visualization/.NoVisualization work)
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir.rstrip("/")),
        "image_filenames": image_filenames,
        "detections": detections_any if det_source == "npy" else None,
        "groundtruth": None,
        "image_size": image_size,
        "min_frame_idx": min_frame,
        "max_frame_idx": max_frame,
        "feature_dim": (detections_any.shape[1] - 10) if det_source == "npy" else 0,
        "update_ms": update_ms_interval,
    }

    # Choose visualizer
    visualizer = visualization.Visualization(seq_info, update_ms=update_ms_interval) if display else visualization.NoVisualization(seq_info)

    def frame_callback(vis, frame_idx):
        # Create detections for this frame
        if det_source == "npy":
            detections = _create_detections_from_npy(detections_any, frame_idx, min_detection_height)
        else:
            detections = _create_detections_from_csv(groups, frame_idx, min_detection_height, use_cosine=use_cosine)

        # Confidence filter
        detections = [d for d in detections if d.confidence >= min_confidence]

        # NMS
        if detections:
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            keep = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections[:] = [detections[i] for i in keep]

        # Predict + Update
        tracker.predict()
        tracker.update(detections)

        # Draw (if we have images for this frame)
        if display and frame_idx in image_filenames:
            img = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            if img is not None:
                vis.set_image(img.copy())
                vis.draw_detections(detections)
                vis.draw_trackers(tracker.tracks)

        # Save MOT-style row(s)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            results.append([frame_idx, track.track_id, x, y, w, h])

    # Run the visualization loop (also drives frame_callback)
    visualizer.run(frame_callback)

    # Write output file (MOT format)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for row in results:
            print("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" %
                  (row[0], row[1], row[2], row[3], row[4], row[5]), file=f)


if __name__ == "__main__":
    dataset_name = 'Fluo-C2DL-MSC'
    seq_num = '02'
    local_dir = 'data/CTC'
    # gt_json_path = f"{local_dir}/{dataset_name}/{seq_num}-gt_dict.json"
    OUTPUT_DIR = f"{local_dir}/{dataset_name}/{seq_num}-tracking_results/"
    # frame_diff = 1
    # include_track_ids = True

    IMAGE_DIR = f"{local_dir}/{dataset_name}/{seq_num}_Msk_CSTQ"  # Raw CTC images (optional, for visualization)
    DETECTION_CSV = f"{local_dir}/{dataset_name}/{seq_num}-detections.csv"

    run(
        sequence_dir=IMAGE_DIR,          # images live here (e.g., 000.tif, 001.tif, ...)
        detection_file=DETECTION_CSV,
        output_file=f"{local_dir}/{dataset_name}/{seq_num}-deepsort-track.txt",
        min_confidence=0.3,
        nms_max_overlap=1.0,
        min_detection_height=0,
        max_cosine_distance=0.2,
        nn_budget=100,
        display=True,
        update_ms_interval=200
    )
