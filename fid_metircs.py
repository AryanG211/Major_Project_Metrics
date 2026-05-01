import torch
from torchvision import transforms, models
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import os

# ---------- CONFIG ----------
real_dir = r"C:\Users\hp\OneDrive\Desktop\dataset_test\faces"
fake_dir = r"C:\Users\hp\OneDrive\Desktop\dataset_test\cycle_output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL ----------
inception = models.inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device)
inception.fc = torch.nn.Identity()  # remove classification layer
inception.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

def get_features(img_dir):
    features = []
    with torch.no_grad():
        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                feat = inception(img).cpu().numpy().squeeze()
                features.append(feat)
    return np.array(features)

# ---------- COMPUTE MEAN & COV ----------
def calculate_fid(real_dir, fake_dir):
    act1 = get_features(real_dir)
    act2 = get_features(fake_dir)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

fid_value = calculate_fid(real_dir, fake_dir)
print(f" (CycleGan)FID Score: {fid_value:.4f}")
