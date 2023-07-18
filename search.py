import numpy as np
import pandas as pd

from utils import compute_text_feature, display_num, features_path

# Read the photos table
photo_features = np.load(features_path / "features.npy")
photo_ids = pd.read_csv(features_path / "photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])
print(f"Photos found: {len(photo_ids)}")
print("press Enter to quit")

search_query = input("> ")
while search_query:
    text_features = compute_text_feature(search_query)
    similarities = list((text_features @ photo_features.T).squeeze(0))
    best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)

    for i in range(display_num):
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        print(photo_id)
    
    search_query = input("> ")