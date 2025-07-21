# inference server start

import os
import cv2
import numpy as np
import pandas as pd
from inference_sdk import InferenceHTTPClient
from PIL import Image
from tqdm import tqdm

# === Параметры ===
dataset_path = r"D:\DATASETS\artifact_dataset\train\\"
output_dir = os.path.join(os.path.dirname(__file__), "inference_results_(train)")
output_filename = "inference_results_(train).csv"
tolerance_deg = 20

API_URL = "http://localhost:9001"
API_KEY = os.getenv("ROBOFLOW_API_KEY")

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
os.makedirs(output_dir, exist_ok=True)

# === Функция отрисовки взгляда ===
def draw_gaze_arrow(original_img: np.ndarray, yaw: float, pitch: float, bbox=None):
    img = original_img.copy()
    h, w = img.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
    else:
        center = (w // 2, h // 2)

    arrow_len = int(w * 0.4)
    dx = -arrow_len * np.sin(yaw) * np.cos(pitch)
    dy = -arrow_len * np.sin(pitch)
    end_point = (int(center[0] + dx), int(center[1] + dy))

    cv2.arrowedLine(img, center, end_point, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)

    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    label = f"yaw: {yaw_deg:.1f} deg, pitch: {pitch_deg:.1f} deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
    cv2.putText(img, label, (15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

# === CSV для записи результатов ===
results = []

# === Получение всех файлов ===
image_names = [f for f in os.listdir(dataset_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# === Обработка изображений ===
for image_name in tqdm(image_names, desc="Processing images"):
    try:
        image_path = os.path.join(dataset_path, image_name)
        img_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(img_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Предсказание
        gaze_result = client.detect_gazes(image_np)

        if not gaze_result or not gaze_result[0]["predictions"]:
            raise ValueError("Face not found")

        prediction = gaze_result[0]["predictions"][0]
        yaw_rad = prediction["yaw"]
        pitch_rad = prediction["pitch"]

        face = prediction.get("face")
        if face:
            x = face.get("x")
            y = face.get("y")
            width = face.get("width")
            height = face.get("height")
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
            bbox = (x1, y1, x2, y2)
        else:
            bbox = None
            x1 = y1 = x2 = y2 = np.nan

        # Визуализация
        viz_img = draw_gaze_arrow(image_bgr, yaw_rad, pitch_rad, bbox=bbox)
        cv2.imwrite(os.path.join(output_dir, f"viz_{image_name}"), viz_img)

        results.append({
            "file": image_name,
            "status": "success",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "yaw_rad": float(yaw_rad),
            "pitch_rad": float(pitch_rad),
            "yaw_deg": float(np.degrees(yaw_rad)),
            "pitch_deg": float(np.degrees(pitch_rad)),
        })

    except Exception as e:
        print(f"⚠️ Ошибка при обработке {image_name}: {e}")
        results.append({
            "file": image_name,
            "status": "failure",
            "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
            "yaw_rad": np.nan, "pitch_rad": np.nan,
            "yaw_deg": np.nan, "pitch_deg": np.nan,
        })

# === Сохранение итогового CSV ===
results_csv_path = os.path.join(os.path.dirname(__file__), output_filename)
pd.DataFrame(results).to_csv(results_csv_path, index=False)
print(f"\n✅ Готово! Результаты в: {output_dir}")
