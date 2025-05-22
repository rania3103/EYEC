#!/usr/bin/env python
"""pick 300 random places images from places kaggle dataset"""
import os
import random
import shutil

source_dir = "../data/places"
dest_dir = "../data/place"
os.makedirs(dest_dir, exist_ok=True)

all_images = os.listdir(source_dir)

# pick 300 random images
selected_images = random.sample(all_images, 300)

# copy them
for img_name in selected_images:
    shutil.copy(
        os.path.join(
            source_dir, img_name), os.path.join(
            dest_dir, img_name))

print(f"Copied 300 random place images to {dest_dir}")
