from pathlib import Path

import cn_clip.clip as clip
import torch
import yaml
from PIL import Image

# Set the paths
with open("config.yaml", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    features_path = Path(data['features_path'])
    photos_path = Path(data['photos_path'])
    batch_size = data['batch_size']
    ext_list = data['ext_list']
    display_num = data['display_num']
    model_name = data['model']

# Load the open CLIP model
print("Available models:", clip.available_models())  
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load_from_name(model_name, device=device, download_root='./')


def compute_text_feature(text):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(text).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded.cpu().numpy()

# Function that computes the feature vectors for a batch of images
def compute_image_features(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    return photos_features.cpu().numpy()






