import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import torch

# === Settings ===
filename = "image_01422_1.png"
image_path = r"D:\DATASETS\artifact_dataset\train\\" + filename
output_path = f"problematic_images/{filename}"
margin = 0.25  # Default margin (if individual ones are not specified)

# === Individual margins (None = use default margin) ===
top_margin = 0.1
bottom_margin = 0.0  # For example, reduce the bottom margin
left_margin = None
right_margin = None

# === MTCNN parameters ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(
    keep_all=True,
    thresholds=[0.4, 0.5, 0.5],
    min_face_size=100,
    factor=0.6,
    device=device
)

print(f"Using device: {device}")

# === Load and process image ===
img = Image.open(image_path).convert('RGB')
boxes, probs = mtcnn.detect(img)

if boxes is None:
    print("❌ No face detected.")
    exit()

# Visualize detected faces
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for box, prob in zip(boxes, probs):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    draw.text((x1, y1 - 10), f"{prob:.2f}", fill='red')

plt.imshow(img_draw)
plt.axis('off')
plt.title("Detected faces")
plt.show()

# === Cropping with margins ===
def crop_with_margin(
        img,
        box,
        margin=0.25,
        top=None,
        bottom=None,
        left=None,
        right=None
):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    top = top if top is not None else margin
    bottom = bottom if bottom is not None else margin
    left = left if left is not None else margin
    right = right if right is not None else margin

    x1 = max(0, int(x1 - left * w))
    x2 = min(img.width, int(x2 + right * w))
    y1 = max(0, int(y1 - top * h))
    y2 = min(img.height, int(y2 + bottom * h))

    return img.crop((x1, y1, x2, y2))

# === Make square crop with padding ===
def make_square(img):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))  # black background
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

# === Crop, square, and save ===
first_box = boxes[0]
cropped = crop_with_margin(
    img, first_box,
    margin=margin,
    top=top_margin,
    bottom=bottom_margin,
    left=left_margin,
    right=right_margin
)
squared = make_square(cropped)

plt.imshow(squared)
plt.axis('off')
plt.title("Square cropped face")
plt.show()

squared.save(output_path)
print(f"✅ Saved: {output_path}")
