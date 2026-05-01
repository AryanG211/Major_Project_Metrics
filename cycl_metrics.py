import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colored_folder = r"C:\Users\hp\OneDrive\Desktop\dataset_test\cycle_output"
real_folder = r"C:\Users\hp\OneDrive\Desktop\dataset_test\sketch"  # ground truth

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------- LOAD IMAGES ----------
colored_images = sorted([f for f in os.listdir(colored_folder) if f.lower().endswith(('.jpg','.png'))])
real_images = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.jpg','.png'))])

mse_list = []
psnr_list = []
ssim_list = []
identity_sim_list = []

# Optional: Use pretrained identity model (ResNet/InsightFace)
# Here, we'll use a simple feature extractor from torchvision for demo
from torchvision.models import resnet18
from torch.nn import functional as F

identity_model = resnet18(pretrained=True).to(device)
identity_model.eval()

def extract_features(img_tensor):
    # img_tensor: (3,H,W) normalized [0,1]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feat = identity_model(img_tensor)
    return feat.cpu().numpy()

# ---------- LOOP THROUGH IMAGES ----------
for c_img_name, r_img_name in tqdm(zip(colored_images, real_images), total=len(colored_images)):
    c_path = os.path.join(colored_folder, c_img_name)
    r_path = os.path.join(real_folder, r_img_name)
    
    c_img = transform(Image.open(c_path).convert("RGB")).numpy().transpose(1,2,0)
    r_img = transform(Image.open(r_path).convert("RGB")).numpy().transpose(1,2,0)
    
    # MSE
    mse = np.mean((c_img - r_img) ** 2)
    mse_list.append(mse)
    
    # PSNR
    psnr = compare_psnr(r_img, c_img, data_range=1.0)
    psnr_list.append(psnr)
    
    # SSIM
    ssim = compare_ssim(r_img, c_img, multichannel=True, data_range=1.0)
    ssim_list.append(ssim)
    
    # Identity similarity (cosine)
    c_tensor = torch.tensor(c_img.transpose(2,0,1)).float()
    r_tensor = torch.tensor(r_img.transpose(2,0,1)).float()
    f1 = extract_features(c_tensor)
    f2 = extract_features(r_tensor)
    identity_sim = cosine_similarity(f1, f2)[0][0]
    identity_sim_list.append(identity_sim)

# ---------- AVERAGE METRICS ----------
print(f"MSE: {np.mean(mse_list):.4f}")
print(f"PSNR: {np.mean(psnr_list):.4f}")
print(f"SSIM: {np.mean(ssim_list):.4f}")
print(f"Identity similarity (cosine): {np.mean(identity_sim_list):.4f}")

# ---------- FID ----------
# Using pytorch-fid: install with `pip install pytorch-fid`
fid_value = calculate_fid_given_paths([colored_folder, real_folder], batch_size=8, device=device, dims=2048)
print(f"FID Score: {fid_value:.4f}")
