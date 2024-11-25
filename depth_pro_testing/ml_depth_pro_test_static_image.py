from PIL import Image
import depth_pro
import numpy as np
import matplotlib.cm as cm

import mediapipe as mp


import cv2
import torch

image_path = r"C:\Users\aaron\Documents\HumonLab\jsm_for_depth_testing.PNG"

img = cv2.imread(image_path)

mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
) as pose:
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_height, img_width, _ = img.shape

mediapipe_array = []

for landmark_data in results.pose_landmarks.landmark:
    # mediapipe_array.append([landmark_data.x*img.shape[0], landmark_data.y*img.shape[1]])
    x = landmark_data.x * img_width
    y = landmark_data.y * img_height
    z = landmark_data.z * img_width  # Z is relative to the width
    mediapipe_array.append([x, y, z])

mediapipe_array = np.array(mediapipe_array)




f = 2
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"))
model.eval()

# Load and preprocess an image
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference
prediction = model.infer(image, f_px=f_px)
depth_array = prediction['depth'].cpu().numpy()

z_values = []
rgd_array = []
for landmark in mediapipe_array:
    x, y, _ = landmark
    x_int = int(x)
    y_int = int(y)
    z = depth_array[y_int, x_int]*1000
    z_values.append(z)
    rgd_array.append([x, y, z])
    print(f"Landmark: {x}, {y}, {z}")
    # print(f"Depth: {depth_array[y, x]*1000}")

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
max_range = 1600
x_limits = [mean_x - max_range, mean_x + max_range]
y_limits = [mean_y - max_range, mean_y + max_range]
z_limits = [mean_z - max_range, mean_z + max_range]

# Plot the 3D scatter points
rgd_array = np.array(rgd_array)
ax.scatter(rgd_array[:, 0], rgd_array[:, 1], rgd_array[:, 2], c='b', marker='o')

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
ax.set_zlim(z_limits)
plt.show()

f = 2
# # Normalize the depth data to [0, 1] for colormap
# depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

# # Apply a colormap (e.g., plasma)
# colormap = cm.plasma(depth_normalized)  # Returns an RGBA array

# # Convert the colormap output to 8-bit RGB
# depth_colored = (colormap[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel

# # Create a PIL image in RGB mode
# depth_image_colored = Image.fromarray(depth_colored, mode='RGB')

# # Display the colored depth image
# depth_image_colored.show()


