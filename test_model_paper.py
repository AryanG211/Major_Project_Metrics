# import torch
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# from pix2pix_model import Generator  # Assuming your code above is in model.py

# # ---------- CONFIG ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = "SKetch_new.pth"  # path to your saved model
# input_image_path = r"C:\Users\hp\OneDrive\Desktop\dataset_test\sketch\4055.jpg"  # path to your sketch image
# output_path = "output_colored.jpg"

# # ---------- LOAD MODEL ----------
# model = Generator(in_channels=3, out_channels=3).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # ---------- PREPROCESS IMAGE ----------
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# image = Image.open(input_image_path).convert("RGB")
# input_tensor = transform(image).unsqueeze(0).to(device)

# # ---------- INFERENCE ----------
# with torch.no_grad():
#     fake_image = model(input_tensor)
# fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)  # Denormalize from [-1,1] → [0,1]

# # ---------- SAVE & SHOW ----------
# save_image = transforms.ToPILImage()(fake_image.squeeze().cpu())
# save_image.save(output_path)

# plt.imshow(save_image)
# plt.axis("off")
# plt.title("Generated Colorized Image")
# plt.show()

# print(f"✅ Output saved at: {os.path.abspath(output_path)}")

import torch
from torchvision import transforms
from PIL import Image
import os
from pix2pix_model import Generator  # your model file

# ---------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "SKetch_new.pth"  # path to your saved model
input_folder = r"C:\Users\hp\OneDrive\Desktop\dataset_test\sketch"  # folder with sketches
output_folder = r"c:\Users\hp\OneDrive\Desktop\dataset_test\Output_coloured"  # folder to save colorized images

os.makedirs(output_folder, exist_ok=True)

# ---------- LOAD MODEL ----------
model = Generator(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- PREPROCESS ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

to_pil = transforms.ToPILImage()

# ---------- INFERENCE LOOP ----------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for i, file_name in enumerate(image_files, start=1):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, f"colored_{file_name}")

    # Load and preprocess
    image = Image.open(input_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        fake_image = model(input_tensor)
    fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)

    # Save
    save_image = to_pil(fake_image.squeeze().cpu())
    save_image.save(output_path)

    print(f"[{i}/{len(image_files)}] ✅ Saved: {output_path}")

print("\n🎨 All images colorized and saved successfully!")
