#!/usr/bin/env python3
"""detecting corrupted images"""
import os
from PIL import Image


def detect_corrupted_images(data_dir, log_file_path, subdirs_to_check):
    """detect corrupted images inside dataset and log their paths to a log file"""
    corrupted_files = []
    # walk through every file in directory and its subdirectories
    for subdir in subdirs_to_check:
        for file in os.listdir(subdir):
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                corrupted_files.append(file_path)
    with open(log_file_path, "w") as f:
        f.write(f"Detected corrupated images:\n\n")
        for cor_file in corrupted_files:
            f.write(f"{cor_file}\n")
        f.write(f"\n\nTotal corrupted images: {len(corrupted_files)} images")
    print("logging done!")


# call function
subdirs_to_check = [
    "../data/person",
    "../data/quote",
    "../data/animal",
    "../data/meme",
    "../data/food",
    "../data/car",
    "../data/art",
    "../data/place"]
detect_corrupted_images(
    "data/",
    "../logs/corrupted_images_log.txt",
    subdirs_to_check)
