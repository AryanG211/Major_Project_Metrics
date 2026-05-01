from insightface.app import FaceAnalysis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import torch

# ---------- CONFIG ----------
real_dir = r"C:\Users\hp\OneDrive\Desktop\dataset_test\faces"
fake_dir = r"C:\Users\hp\OneDrive\Desktop\dataset_test\cycle_output"

# ---------- INIT FACE ANALYSIS ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# ---------- COSINE SIMILARITY ----------
def get_embedding(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding  # use first detected face

sims = []
for file in os.listdir(real_dir):
    if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    real_path = os.path.join(real_dir, file)
    fake_path = os.path.join(fake_dir, f"colored_{file}")
    if not os.path.exists(fake_path):
        continue

    emb_real = get_embedding(real_path)
    emb_fake = get_embedding(fake_path)
    if emb_real is not None and emb_fake is not None:
        sim = cosine_similarity([emb_real], [emb_fake])[0][0]
        sims.append(sim)

if sims:
    avg_sim = np.mean(sims)
    print(f"🧩(CycleGAN) Average Identity Similarity (Cosine): {avg_sim:.4f}")
else:
    print("⚠️ No faces detected in one or more images.")
