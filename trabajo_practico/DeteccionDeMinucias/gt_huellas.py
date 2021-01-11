import os

import cv2
import numpy as np
import pandas as pd


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, filename


# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
    # Clicks botón izquierdo: TERMINACIÓN
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        cv2.circle(imagen, (x, y), 4, (255, 0, 0), -1)
        print("terminación: ", x, " ", y)

        minucias.append([x, y, 1])
        # bifurcacion_x.append(0)
        # bifurcacion_y.append(0)
        cv2.imshow("image", imagen)

        # Cliks botón derecho: BIFURCACIÓN
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        cv2.circle(imagen, (x, y), 4, (0, 0, 255), -1)
        print("bifurcación: ", x, " ", y)

        minucias.append([x, y, 3])
        # terminacion_x.append(0)
        # terminacion_y.append(0)
        cv2.imshow("image", imagen)


if __name__ == "__main__":

    folder = "huellas_test"
    imagenes, fil = load_images_from_folder(folder)
    n_imagen = 0

    contenido = os.listdir(folder)

    dirName = "GT-huellas"
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    for imagen in imagenes:
        n_imagen += 1
        minucias = []

        # almacenamos el nombre del archivo actual
        filenames = contenido[n_imagen - 1]

        print(dirName, n_imagen)
        print(filenames)

        # displaying the image
        cv2.imshow("image", imagen)

        # setting mouse hadler for the image and calling the click_event() function
        cv2.setMouseCallback("image", click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

        # nombre de los archivos de salida
        datafile_path1 = "GT-huellas/"
        datafile_path2 = filenames  # .remplace(".tif", "")
        datafile_path3 = ".txt"
        datafile_path = (
            datafile_path1 + str(datafile_path2).replace(".tif", "") + datafile_path3
        )

        minucias = pd.DataFrame.from_records(minucias)
        minucias.columns = ["col", "row", "label"]
        minucias.to_csv(datafile_path, sep=";")

        print("Se ha guardado en: ", datafile_path)
