import os
import numpy as np
import csv

LOG_DIR = '/home/usman/behavior_logs'
OUTPUT_FILE = '/home/usman/collected_data.npy'

def combine_data():
    images = []
    controls = []

    # Load images
    image_files = sorted([file for file in os.listdir(LOG_DIR) if file.endswith('.npy') and file.startswith('image_')])
    for file in image_files:
        image_path = os.path.join(LOG_DIR, file)
        images.append(np.load(image_path))

    # Load controls
    control_logs_path = os.path.join(LOG_DIR, 'control_logs.csv')
    with open(control_logs_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 4:
                # Attempt to unpack only the first four elements
                _, _, steering, throttle = row[:4]
                controls.append([float(steering), float(throttle)])
            else:
                print(f"Unexpected row format: {row}")

    print(f"Loaded {len(images)} images and {len(controls)} control entries.")

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize images
    controls = np.array(controls, dtype=np.float32)

    # Ensure data consistency
    if len(images) != len(controls):
        print(f"Data mismatch: {len(images)} images and {len(controls)} controls. Please verify your data.")
        return

    # Combine and save data
    np.save(OUTPUT_FILE, {'images': images, 'controls': controls})
    print(f"Combined data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    combine_data()
