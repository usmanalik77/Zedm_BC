#!/usr/bin/env python3
import os
import csv
import numpy as np

LOG_DIR = '/home/usman/duplo_data'           
OUTPUT_FILE = '/home/usman/duplo_combined/collected_data.npy'

def combine_data():
    images = []
    labels = []


    image_files = sorted(
        f for f in os.listdir(LOG_DIR)
        if f.startswith('image_') and f.endswith('.npy')
    )

    for fname in image_files:
        img_path = os.path.join(LOG_DIR, fname)
        img = np.load(img_path)          # shape (H, W, 3) 
        images.append(img)

    # 2) Read CSV
    csv_path = os.path.join(LOG_DIR, 'data_logs.csv')
    label_map = {"square": 1, "not-square": 0}  # convert text to numeric classes
    with open(csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)  # skip header => ["image_index","label","state"]
        for row in reader:

            if len(row) >= 2:
                label_str = row[1]   # e.g. "square" or "not-square"
                # convert to 1 or 0
                numeric_label = label_map.get(label_str, 0)
                labels.append(numeric_label)

    print(f"Loaded {len(images)} images and {len(labels)} labels from CSV.")
    if len(images) != len(labels):
        print(f"Data mismatch: {len(images)} images vs {len(labels)} labels!")
        return

    images = np.array(images, dtype=np.float32)  # shape  (N, H, W, 3)
    # simple normalization from 0..255 => 0..1 if needed
    if images.max() > 1.0:
        images /= 255.0

    labels = np.array(labels, dtype=np.int64)    # shape(N,)

    out_dict = {
        "images": images,     # shape (N,H,W,3)
        "labels": labels,     # shape (N,)
    }
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, out_dict, allow_pickle=True)
    print(f"Combined data saved to {OUTPUT_FILE}")

def main():
    combine_data()

if __name__ == "__main__":
    main()
