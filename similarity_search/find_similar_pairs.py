import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch.nn.functional as F

# === Paths ===
train_dir = Path(__file__).resolve().parent.parent / "face_focus" / "faces_cropped_mtcnn_(train)"
output_dir = Path("similar_pairs_vis")
output_csv = Path("similar_pairs.csv")
output_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_shape = (224, 224)

# === Transformation ===
transform = transforms.Compose([
    transforms.Resize(resize_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Model without classifier ===
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

@torch.no_grad()
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB").resize(resize_shape[::-1])  # (W, H)
    tensor = transform(img).unsqueeze(0).to(device)
    embedding = model(tensor).squeeze()
    return F.normalize(embedding, p=2, dim=0).cpu()

# === Collect files ===
artifact_images = sorted([f for f in os.listdir(train_dir) if f.endswith("_0.png")])
clean_images = sorted([f for f in os.listdir(train_dir) if f.endswith("_1.png")])
print(f"Found {len(artifact_images)} artifact and {len(clean_images)} clean images.")

# === Clean embeddings ===
print("Extracting embeddings for clean images...")
clean_embeddings = {}
for fname in tqdm(clean_images):
    try:
        clean_embeddings[fname] = get_embedding(train_dir / fname)
    except Exception as e:
        print(f"⚠️ {fname}: {e}")

# === Find similar pairs ===
pairs = []
print("Finding pairs...")
for artifact_fname in tqdm(artifact_images):
    try:
        art_emb = get_embedding(train_dir / artifact_fname)
    except Exception as e:
        print(f"⚠️ {artifact_fname}: {e}")
        continue

    best_match = None
    best_sim = -1
    for clean_fname, clean_emb in clean_embeddings.items():
        sim = torch.dot(art_emb, clean_emb).item()
        if sim > best_sim:
            best_sim = sim
            best_match = clean_fname

    pairs.append({
        "artifact": artifact_fname,
        "matched_clean": best_match,
        "similarity": round(best_sim, 4)
    })

df = pd.DataFrame(pairs)
df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv.resolve()}")

# === Visualization ===
print("Visualizing pairs...")
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for idx, row in tqdm(df.iterrows(), total=len(df)):
    a, b, sim = row["artifact"], row["matched_clean"], row["similarity"]
    if not a or not b:
        continue

    path_a, path_b = train_dir / a, train_dir / b
    try:
        img_a = Image.open(path_a).convert("RGB").resize((512, 512))
        img_b = Image.open(path_b).convert("RGB").resize((512, 512))
    except:
        continue

    canvas = Image.new("RGB", (2 * 512, 512 + 40), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (512, 0))

    # Labels
    def get_text_width(text, font):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    text_a = a
    text_b = b
    text_sim = f"sim: {sim:.4f}"

    w_a = get_text_width(text_a, font)
    w_b = get_text_width(text_b, font)
    w_sim = get_text_width(text_sim, font)

    draw.text((10, 512 + 10), text_a, font=font, fill=(0, 0, 0))
    draw.text((1024 - w_b - 10, 512 + 10), text_b, font=font, fill=(0, 0, 0))
    draw.text(((1024 - w_sim) // 2, 512 + 10), text_sim, font=font, fill=(255, 0, 0))

    canvas.save(output_dir / f"pair_{idx:04d}.jpg")

print(f"✅ Visualization saved in {output_dir.resolve()}")
