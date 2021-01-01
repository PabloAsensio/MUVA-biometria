import numpy as np
import cv2 as cv
import os
import sys
from classifiers import calculate_histogram, calculate_label
import matplotlib.pyplot as plt
import pandas as pd

# img = cv2.imread("./FravAttack/FLIR/Rainbow/front/users/USER_024.JPG")


# cv2.imshow("Main Face", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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


if __name__ == "__main__":
    raw_img_list = get_all_images_paths(
        "./FravAttack/FLIR/Rainbow/front", ext=".jpg"
    )  # Change to your dataset path

    # create pandas dataframe
    features_df = pd.DataFrame()

    # select output path
    output_path = "histogram_features.csv"

    for _, individual in enumerate(raw_img_list):
        # calulate features and label of picture
        hist = calculate_histogram(individual)
        label = calculate_label(individual)

        # append labels and path to features
        hist = np.append(hist, label)
        hist = np.append(hist, individual)

        # data to df
        features_df = features_df.append(
            pd.DataFrame(data=hist).transpose(), ignore_index=True
        )

    # # DECOMMENT if you wanna store df to .csv
    # features_df.to_csv(output_path, index=False)
    print(features_df[769][149])
