import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classifiers import do_classification, histogram_features, Conv2Dclassification
from FaceDetector import getSegmentedFaces
from metrics import metric
from common import get_all_images_paths


if __name__ == "__main__":
    y_prob_metric = []
    y_predic_metric = []
    y_real_metric = []
    labels = []
    img_list = get_all_images_paths(
        'D:/Descargas/FravAttack_con_bonafide/FravAttack/FLIR/Rainbow/front', ext=".jpg"
    )  # Change to your dataset path

    features_df = histogram_features(img_list)

    # # DECOMMENT if you wanna store df to .csv
    # features_df.to_csv("histogram_features.csv", index=False)

    # # DECOMMENT if you wanna read df from .csv
    # features_df = pd.read_csv("histogram_features.csv", header=0, delimiter=",")

    # Train and test soft voting classifier
    (y_prob, y_predic, y_real) = do_classification(features_df)
    y_prob_metric.append(y_prob)
    y_predic_metric.append(y_predic)
    y_real_metric.append(y_real)
    labels.append('Histograms - Termicas')

    print('********************* LOADING DATASET IR IMAGES *********************')
    path = 'D:/Descargas/REAL_SENSE/REAL_SENSE/REAL_SENSE_IR/' # Change to your dataset path
    ir_segmented_images, ir_segmented_tags = getSegmentedFaces(path)
    ir_segmented_images = np.asarray(ir_segmented_images)
    print('********************* DATASET LOADED  *********************')
    y_prob, y_predic, y_real = Conv2Dclassification(ir_segmented_images, ir_segmented_tags)

    y_prob_metric.append(y_prob)
    y_predic_metric.append(y_predic)
    y_real_metric.append(y_real)
    labels.append('Conv2D - IR')

    metric(y_prob_metric, y_predic_metric, y_real_metric, labels, 1)


    # Generate APCER-BPCER-curve
    #metric(y_prob, y_predic, y_real, 1)
