import numpy as np
import matplotlib.pyplot as plt

# Path to your combined data file
DATA_FILE = '/home/usman/duplo_combined/collected_data.npy'

def visualize_data(data_file, num_samples=8):
    # Load the data as a dictionary
    data = np.load(data_file, allow_pickle=True).item()
    print("Keys found in data file:", list(data.keys()))
    
    if 'images' not in data:
        raise KeyError("Key 'images' not found in data file!")
    images = data['images']
    
    # Use 'controls' if available, otherwise try 'labels'
    if 'controls' in data:
        labels = data['controls']
    elif 'labels' in data:
        labels = data['labels']
    else:
        labels = ["N/A"] * len(images)
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        ax = axes[i]
        # Assume images are normalized; multiply by 255 if needed
        ax.imshow(images[i])
        ax.set_title(f"Index {i}\nLabel: {labels[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_data(DATA_FILE)
