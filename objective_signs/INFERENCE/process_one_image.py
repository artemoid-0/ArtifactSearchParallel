import os
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

API_URL = "http://localhost:9001"
API_KEY = os.getenv("ROBOFLOW_API_KEY")
IMAGE_PATH = "results/image_00140_0.png"
TOLERANCE_DEGREES = 20  # yaw/pitch tolerance for "looking at the camera"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"File not found: {IMAGE_PATH}")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gaze_result = client.detect_gazes(inference_input=rgb_image)
if not gaze_result or not gaze_result[0]["predictions"]:
    raise ValueError("Face not detected")

prediction = gaze_result[0]["predictions"][0]
yaw_rad = prediction["yaw"]
pitch_rad = prediction["pitch"]

yaw_deg = np.degrees(yaw_rad)
pitch_deg = np.degrees(pitch_rad)

is_forward = abs(yaw_deg) < TOLERANCE_DEGREES and abs(pitch_deg) < TOLERANCE_DEGREES
print(f"Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°")
print("✔️ Looking at camera" if is_forward else "❌ Not looking at camera")

face = prediction["face"]
xc, yc = int(face["x"]), int(face["y"])
imgH, imgW = image.shape[:2]
L = imgW // 2

dx = -L * np.sin(yaw_rad) * np.cos(pitch_rad)
dy = -L * np.sin(pitch_rad)

cv2.arrowedLine(image, (xc, yc), (int(xc + dx), int(yc + dy)), (0, 0, 255), 3)

cv2.rectangle(
    image,
    (int(face["x"] - face["width"] / 2), int(face["y"] - face["height"] / 2)),
    (int(face["x"] + face["width"] / 2), int(face["y"] + face["height"] / 2)),
    (255, 0, 0), 2
)

cv2.putText(
    image,
    f"Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°",
    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
)

cv2.imshow("Gaze Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
