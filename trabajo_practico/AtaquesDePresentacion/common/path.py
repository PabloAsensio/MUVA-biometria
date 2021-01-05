import os
import sys


def get_absolute_path(relative_path):
    return os.path.realpath(relative_path)


def check_path(path):
    return os.path.exists(path)


def get_all_images_paths(root, ext=".jpg"):
    if not check_path(root):
        print("Path doesn't exist.")
        sys.exit(1)

    print("Getting all images paths...")
    images_list = []
    for path, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(ext):
                images_list.append(get_absolute_path(os.path.join(path, name)))
    print("Done! (Getting all images paths).")
    return images_list