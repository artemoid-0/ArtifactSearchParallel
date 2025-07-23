import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from gaze360.code.model import GazeLSTM
import os
from tqdm import tqdm

# === Parameters ===
dataset_path = r"D:\DATASETS\artifact_dataset\train\\"
bbox_csv_path = "../../face_focus/GOOD_RESERVE/mtcnn_crop_log_(train).csv"  # ← Path to CSV with face coordinates
output_dir = os.path.join(os.path.dirname(__file__), "results_(train)")
results_csv_path = os.path.join(os.path.dirname(__file__), "gaze360_results_(train).csv")
os.makedirs(output_dir, exist_ok=True)

# === Gaze Model ===
model = GazeLSTM()
state_dict = torch.load("gaze360_model.pth.tar")["state_dict"]
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
model.eval()

# === Gaze360 Transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Function to visualize gaze + bbox ===
def draw_gaze_arrow(original_img: np.ndarray, theta: float, phi: float, bbox=None):
    img = original_img.copy()
    h, w = img.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    center = ((x1 + x2) // 2, (y1 + y2) // 2) if bbox else (w // 2, h // 2)

    arrow_len = int(w * 0.4)
    dx = -arrow_len * np.sin(theta) * np.cos(phi)
    dy = -arrow_len * np.sin(phi)
    end_point = (int(center[0] + dx), int(center[1] + dy))

    cv2.arrowedLine(img, center, end_point, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)

    # Text with background
    yaw_deg = np.degrees(theta)
    pitch_deg = np.degrees(phi)
    label = f"yaw: {yaw_deg:.1f} deg, pitch: {pitch_deg:.1f} deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
    cv2.putText(img, label, (15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

# === Load face coordinates ===
bbox_df = pd.read_csv(bbox_csv_path)
bbox_df["file"] = bbox_df["file"].apply(lambda x: os.path.basename(x))

# === CSV for saving results ===
results = []

# === Image processing ===
for _, row in tqdm(bbox_df.iterrows(), total=len(bbox_df), desc="Processing images"):
    image_name = row["file"]
    status = row["status"]

    if status != "success":
        print(f"⛔ Skipping {image_name}: status {status}")
        continue

    try:
        image_path = os.path.join(dataset_path, image_name)
        img_pil = Image.open(image_path).convert("RGB")

        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        face_crop = img_pil.crop((x1, y1, x2, y2))

        # Prepare input
        img_tensor = transform(face_crop)
        img_sequence = img_tensor.unsqueeze(0).repeat(7, 1, 1, 1).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            angular_output, _ = model(img_sequence)
            theta, phi = angular_output[0].cpu().numpy()

        # Visualization
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        viz_img = draw_gaze_arrow(img_cv2, theta, phi, bbox=(x1, y1, x2, y2))

        # Save image
        output_path = os.path.join(output_dir, f"viz_{image_name}")
        cv2.imwrite(output_path, viz_img)

        # Save to CSV
        results.append({
            "file": image_name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "yaw_rad": float(theta),
            "pitch_rad": float(phi),
            "yaw_deg": float(np.degrees(theta)),
            "pitch_deg": float(np.degrees(phi)),
        })

    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")

# === Save final CSV ===
pd.DataFrame(results).to_csv(results_csv_path, index=False)
print(f"\n✅ Done! Results saved to: {output_dir}")
