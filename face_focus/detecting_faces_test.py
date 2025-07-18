import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# === Paths ===
input_dir = r"D:\DATASETS\artifact_dataset\train"
alternative = ""

output_dir = r"faces_cropped_mtcnn_(train)"
os.makedirs(output_dir, exist_ok=True)

# === Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# === MTCNN initialization ===
mtcnn = MTCNN(keep_all=False, thresholds=[0.4, 0.5, 0.5], min_face_size=150, factor=0.6, device=device)

# === Global margin settings (can be adjusted) ===
default_margin = 0.25
top_margin = 0.0
bottom_margin = 0.075
left_margin = None
right_margin = None

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

def make_square(img):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

# === Process all images ===
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        input_path = os.path.join(root, file)
        img = Image.open(input_path).convert('RGB')

        boxes, _ = mtcnn.detect(img)

        if boxes is None:
            print(f"❌ Face not found: {file}")
            continue

        box = boxes[0]
        cropped = crop_with_margin(
            img,
            box,
            margin=default_margin,
            top=top_margin,
            bottom=bottom_margin,
            left=left_margin,
            right=right_margin
        )
        squared = make_square(cropped)

        rel_path = os.path.relpath(root, input_dir)
        save_folder = os.path.join(output_dir, rel_path)
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, file)
        squared.save(save_path)
        print(f"✅ Saved: {save_path}")
