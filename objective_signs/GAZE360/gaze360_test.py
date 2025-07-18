import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from gaze360.code.model import GazeLSTM

# Load model
model = GazeLSTM()
state_dict = torch.load("gaze360_model.pth.tar")["state_dict"]

# Remove "module." and rename "fc1" to "fc" if needed
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("module.", "")
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset_path = r"D:\DATASETS\artifact_dataset\train\\"
image_path = dataset_path + "image_00042_1.png"
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img)

# Repeat image to form a sequence: [1, 7, 3, 224, 224]
img_sequence = img_tensor.unsqueeze(0).repeat(7, 1, 1, 1)
img_sequence = img_sequence.unsqueeze(0)

with torch.no_grad():
    angular_output, var = model(img_sequence)
    theta, phi = angular_output[0].cpu().numpy()
    sigma = var[0, 0].cpu().item()


def interpret_gaze(theta_rad, phi_rad, sigma):
    theta_deg = np.degrees(theta_rad)
    phi_deg = np.degrees(phi_rad)

    if sigma < 0.3:
        confidence = "high confidence"
    elif sigma < 0.6:
        confidence = "moderate confidence"
    else:
        confidence = "low confidence"

    if theta_deg < -10:
        rel_person_horizontal = "looking to their LEFT"
    elif theta_deg > 10:
        rel_person_horizontal = "looking to their RIGHT"
    else:
        rel_person_horizontal = "looking straight (horizontally)"

    if phi_deg < -10:
        rel_person_vertical = "looking down"
    elif phi_deg > 10:
        rel_person_vertical = "looking up"
    else:
        rel_person_vertical = "looking straight (vertically)"

    if theta_deg < -10:
        rel_camera_horizontal = "looking right (from the camera's perspective)"
    elif theta_deg > 10:
        rel_camera_horizontal = "looking left (from the camera's perspective)"
    else:
        rel_camera_horizontal = "looking directly at the camera (horizontally)"

    if phi_deg < -10:
        rel_camera_vertical = "looking down (from the camera's perspective)"
    elif phi_deg > 10:
        rel_camera_vertical = "looking up (from the camera's perspective)"
    else:
        rel_camera_vertical = "looking directly at the camera (vertically)"

    print(f"Gaze angles:")
    print(f"  θ (left/right): {theta_rad:.2f} rad / {theta_deg:.1f}°")
    print(f"  φ (up/down): {phi_rad:.2f} rad / {phi_deg:.1f}°")
    print(f"  Uncertainty (sigma): {sigma:.2f} — {confidence}")
    print()
    print(f"Relative to person: {rel_person_horizontal}, {rel_person_vertical}")
    print(f"Relative to camera: {rel_camera_horizontal}, {rel_camera_vertical}")


interpret_gaze(theta, phi, sigma)
