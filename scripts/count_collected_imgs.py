#!/usr/bin/env python3
""" log the number of collected images (number of images in each folder and the total ) in a text file inside logs folder"""
import os


def count_img_in_folder(img_folder, subdirs_to_check):
    """ count images in folder and return a dictionary with folder names as keys and counts as values"""
    total_img = 0
    folder_img_count = {}
    for subdir in subdirs_to_check:
        folder_path = os.path.join(img_folder, subdir)
        num_images = len([file for file in os.listdir(folder_path)])
        folder_img_count[subdir] = num_images
        total_img += num_images
    return folder_img_count, total_img


def log_image_counts(img_folder, log_file):
    """ log the image count in each folder and total images in a log file before cleaning"""
    folder_img_count, total_img = count_img_in_folder(
        "../data/", subdirs_to_check)
    with open(log_file, "w") as f:
        f.write(f"Log of images in {img_folder}\n")
        f.write("=" * 40 + "\n")
        for folder, count in folder_img_count.items():
            f.write(f"Folder [{folder}]: {count} images \n")
        f.write("=" * 40 + "\n")
        f.write(
            f"Total number of images across all folders': {total_img} images \n")


# log image counts
subdirs_to_check = ["../data/non_meme"]
log_image_counts("../data/", "../logs/image_count_log.txt")
print("logging done")
