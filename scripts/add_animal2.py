#!/usr/bin/env python
"""pick 300 random art images from art kaggle dataset"""
import os
import random
import shutil

source_dir = "../data/dogs"
dest_dir = "../data/animal"
os.makedirs(dest_dir, exist_ok=True)

all_images = os.listdir(source_dir)

# pick 6 random images
selected_images = random.sample(all_images, 6)

# copy them
for img_name in selected_images:
    shutil.copy(
        os.path.join(
            source_dir, img_name), os.path.join(
            dest_dir, img_name))

print(f"Copied 6 random art images to {dest_dir}")
