import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classifiers import do_classification, histogram_features
from common import get_all_images_paths
from metrics import metric

if __name__ == "__main__":
    img_list = get_all_images_paths(
        "./FravAttack/FLIR/Rainbow/front", ext=".jpg"
    )  # Change to your dataset path

    # features_df = histogram_features(img_list)

    # # DECOMMENT if you wanna store df to .csv
    # features_df.to_csv("histogram_features.csv", index=False)

    features_df = pd.read_csv("histogram_features.csv", header=0, delimiter=",")

    # Train and test soft voting classifier
    (y_prob, y_predic, y_real) = do_classification(features_df)

    # Generate APCER-BPCER-curve
    metric(y_prob, y_predic, y_real, 1)
