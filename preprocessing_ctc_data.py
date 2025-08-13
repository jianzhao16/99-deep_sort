import os
import numpy as np
from skimage.measure import regionprops, label
import pandas as pd
from tifffile import imread

def convert_ctc_to_detections(seg_dir, output_csv):
    """
    Convert CTC segmentation masks to Deep SORT-style detection CSV.
    Format: frame, x1, y1, width, height, score (default=1.0), class_id, track_id (optional)
    """
    records = []
    file_list = sorted([f for f in os.listdir(seg_dir) if f.endswith(".tif")])
    print(f"[INFO] Found {len(file_list)} .tif files in {seg_dir}")

    for idx, filename in enumerate(file_list):
        frame_number = int(''.join(filter(str.isdigit, os.path.splitext(filename)[0])))
        file_path = os.path.join(seg_dir, filename)
        try:
            seg = imread(file_path)
            labeled = label(seg)
            props = regionprops(labeled)

            print(f"[Frame {frame_number:03d}] Processing '{filename}' → {len(props)} detections")

            for region in props:
                y1, x1, y2, x2 = region.bbox
                width = x2 - x1
                height = y2 - y1
                record = [frame_number, x1, y1, width, height, 1.0, 1, -1]
                records.append(record)
                print(record)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    df = pd.DataFrame(records, columns=["frame", "x", "y", "w", "h", "conf", "class", "track"])
    df.to_csv(output_csv, index=False)
    print(f"[DONE] Saved {len(df)} total detections to: {output_csv}")


# === Config ===
#dataset_name = 'Fluo-N2DH-SIM+'
# dataset_name = 'Fluo-C2DL-Huh7'
dataset_name = 'Fluo-C2DL-MSC'

seq_num = '02'
local_dir = 'data/CTC'
seg_dir_set = f"{local_dir}/{dataset_name}/{seq_num}/"
output_csv_set = f"{local_dir}/{dataset_name}/{seq_num}-detections.csv"

# === Run ===
convert_ctc_to_detections(seg_dir_set, output_csv_set)
