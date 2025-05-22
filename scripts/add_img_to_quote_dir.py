#!/usr/bin/env python
"""add more images to quote folder to attain 300 images"""
import os
import random
import shutil

dest_dir = "../data/quote"

q_dir = "../data/quotes"

# calculate how many more to reach 300
remaining_needed = 300 - 51
print(f"need {remaining_needed} more random quotes from backup.")

# get random images from backup quotes dir
if remaining_needed > 0:
    backup_images = os.listdir(q_dir)
    selected_images = random.sample(backup_images, remaining_needed)

    for img_name in selected_images:
        src_path = os.path.join(q_dir, img_name)
        dest_path = os.path.join(dest_dir, img_name)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)

    print(f"copied {remaining_needed} random quote images from backup.")
