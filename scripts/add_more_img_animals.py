#!/usr/bin/env python
"""pick 300 random art images from art kaggle dataset"""
import os
import random
import shutil

source_dir = "../data/cats"
dest_dir = "../data/animal"
os.makedirs(dest_dir, exist_ok=True)

all_images = os.listdir(source_dir)

# pick 300 random images
selected_images = random.sample(all_images, 150)

# copy them
for img_name in selected_images:
    shutil.copy(
        os.path.join(
            source_dir, img_name), os.path.join(
            dest_dir, img_name))

print(f"Copied 300 random art images to {dest_dir}")

source_dir_2 = "../data/dogss"
os.makedirs(dest_dir, exist_ok=True)

all_images2 = os.listdir(source_dir)

# pick 300 random images
selected_images2 = random.sample(all_images, 150)

# copy them
for img_name in selected_images2:
    shutil.copy(
        os.path.join(
            source_dir, img_name), os.path.join(
            dest_dir, img_name))

print(f"Copied 300 random art images to {dest_dir}")
