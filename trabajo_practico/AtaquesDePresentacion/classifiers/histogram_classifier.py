import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.utils import shuffle


def do_classification(features, seed=1234):
    # randomize labels and features
    features = shuffle(features, random_state=seed)

    # img_path = features.iloc[:, -1]

    # Extractind data
    labels = features.iloc[:, -2]
    features = features.drop(features.columns[[-1, -2]], axis=1)

    # Separate train and test data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.33, random_state=42
    )

    # Data training processing
    scaler = MinMaxScaler()
    pca = PCA(
        n_components=0.99
    )  # Conserv 99% init information (only 85 features to work with) [768 before]

    X_ = scaler.fit_transform(features_train)
    Y_train = labels_train.values.ravel()
    pca.fit(X_, Y_train)
    X_train = pca.transform(X_)

    # Data testing processing
    X_test = pca.transform(scaler.transform(features_test))
    Y_test = labels_test.values.ravel()

    # Calculate soft VotingClassifier
    clf = classifier(X_train, Y_train)
    y_prob = np.max(clf.predict_proba(X_test), axis=1)
    y_predic = clf.predict(X_test)

    return (y_prob, y_predic, Y_test)


def classifier(X, Y, seed=1234):
    # Classifier
    svm_clf = SVC(
        C=0.001,
        degree=4,
        gamma=0.01,
        kernel="poly",
        random_state=seed,
        probability=True,
    )
    svm_clf_1 = SVC(
        C=0.05, gamma=0.03, kernel="rbf", random_state=seed, probability=True
    )
    svm_clf_2 = SVC(
        C=0.1, gamma=0.05, kernel="rbf", random_state=seed, probability=True
    )
    svm_clf_3 = SVC(C=1, gamma=0.075, kernel="rbf", random_state=seed, probability=True)
    tree_clf = DT(max_depth=3, random_state=seed)

    clf = VotingClassifier(
        estimators=[
            ("svc", svm_clf),
            ("svc1", svm_clf_1),
            ("svc2", svm_clf_2),
            ("svc3", svm_clf_3),
            ("tree", tree_clf),
        ],
        voting="soft",
    )

    clf.fit(X, Y)
    return clf


def histogram_features(img_list):
    # create pandas dataframe
    features_df = pd.DataFrame()

    print("Extracting histogram features...")
    for _, individual in enumerate(img_list):
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
    print("Done! (Extracting histogram features)")

    return features_df


def calculate_histogram(img):
    hist = []
    img = cv.imread(img)
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        hist.append(histr)
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    # plt.show()
    hist = np.asarray(hist, dtype=np.float32)
    return np.ravel(hist)


def calculate_label(img):
    """
    1 is bonafide, otherwise 0
    """
    if "users" in img:
        return 0
    return 1
