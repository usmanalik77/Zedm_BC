import numpy as np

data = np.load('/home/usman/collected_data/collected_data.npy', allow_pickle=True).item()
images = data['images']
controls = data['controls']

print(f"Number of images: {images.shape[0]}")
print(f"Number of control entries: {controls.shape[0]}")

# Optionally, visualize an image and its controls
import matplotlib.pyplot as plt

plt.imshow(images[0])
plt.title(f"Steering: {controls[0][0]}, Throttle: {controls[0][1]}")
plt.show()
