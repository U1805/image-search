import math

import numpy as np
import pandas as pd

from utils import *

# List all JPGs in the folder
photos_files = [item for e in ext_list for item in photos_path.glob(e)]
print(f"Photos found: {len(photos_files)}")


batches = math.ceil(len(photos_files) / batch_size)

# Process each batch
for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    batch_ids_path = features_path / f"{i:010d}.csv"
    batch_features_path = features_path / f"{i:010d}.npy"
    
    if not batch_features_path.exists():
        try:
            batch_files = photos_files[i*batch_size : (i+1)*batch_size]

            batch_features = compute_image_features(batch_files)
            np.save(batch_features_path, batch_features)

            photo_ids = [str(photo_file.resolve()) for photo_file in batch_files]
            photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
            photo_ids_data.to_csv(batch_ids_path, index=False)
        except:
            print(f'Problem with batch {i}')


# Load all numpy files
features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

# Concatenate the features and store in a merged file
features = np.concatenate(features_list)
np.save(features_path / "features.npy", features)

# Load all the photo IDs
photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
photo_ids.to_csv(features_path / "photo_ids.csv", index=False)