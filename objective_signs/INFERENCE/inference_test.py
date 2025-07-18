import os
import cv2
import numpy as np
import pandas as pd
from inference_sdk import InferenceHTTPClient

# === Parameters ===
API_URL = "http://localhost:9001"
API_KEY = os.getenv("ROBOFLOW_API_KEY")
IMAGE_FOLDER = r"D:\DATASETS\artifact_dataset\train"
TOLERANCE_DEGREES = 20
CSV_PATH = f"results/gaze_results_tol{TOLERANCE_DEGREES}.csv"
VIZ_FOLDER = f"gaze_viz_tol{TOLERANCE_DEGREES}"

# === Initialize client and output folder ===
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
os.makedirs(VIZ_FOLDER, exist_ok=True)

# === Gaze visualization ===
def draw_gaze(img: np.ndarray, gaze: dict):
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    _, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
    dy = -arrow_length * np.sin(gaze["pitch"])
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )

    for keypoint in face["landmarks"]:
        x, y = int(keypoint["x"]), int(keypoint["y"])
        cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

    label = f"yaw {gaze['yaw']:.2f}  pitch {gaze['pitch']:.2f}"
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

# === Results list ===
results = []

# === Process images ===
for root, _, files in os.walk(IMAGE_FOLDER):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        full_path = os.path.join(root, file)
        try:
            image = cv2.imread(full_path)
            if image is None:
                raise ValueError("Image load error")

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gaze_result = client.detect_gazes(inference_input=rgb_image)

            if not gaze_result or not gaze_result[0]["predictions"]:
                yaw_deg, pitch_deg, verdict = None, None, "❌ Face not found"
            else:
                prediction = gaze_result[0]["predictions"][0]
                yaw_rad = prediction["yaw"]
                pitch_rad = prediction["pitch"]

                yaw_deg = np.degrees(yaw_rad)
                pitch_deg = np.degrees(pitch_rad)
                is_forward = abs(yaw_deg) < TOLERANCE_DEGREES and abs(pitch_deg) < TOLERANCE_DEGREES
                verdict = "✔️ Looking at camera" if is_forward else "❌ Not looking at camera"

                # Visualization and saving
                draw_gaze(image, prediction)
                save_path = os.path.join(VIZ_FOLDER, os.path.relpath(full_path, IMAGE_FOLDER))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, image)

            results.append({
                "image_path": os.path.relpath(full_path, IMAGE_FOLDER),
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "verdict": verdict
            })

            print(f"{file}: {verdict} ({yaw_deg=:.1f}°, {pitch_deg=:.1f}°)" if yaw_deg is not None else f"{file}: {verdict}")

        except Exception as e:
            print(f"[ERROR] {file}: {e}")
            results.append({
                "image_path": os.path.relpath(full_path, IMAGE_FOLDER),
                "yaw_deg": None,
                "pitch_deg": None,
                "verdict": f"❌ Error: {str(e)}"
            })

# === Save CSV ===
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Results saved to: {CSV_PATH}")
print(f"✅ Visualizations saved to: {VIZ_FOLDER}")
