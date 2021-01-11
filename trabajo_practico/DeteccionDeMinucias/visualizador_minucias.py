import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Modificar las rutas
    img = "./huellasFVC2004/101_3.tif"
    minucias = pd.read_csv("./GT-huellas/101_3.txt", sep=";", index_col=0, header=0)

    terminaciones = minucias[minucias["label"] == 1].drop(columns="label")
    bifurcaciones = minucias[minucias["label"] == 3].drop(columns="label")

    if True:
        img = plt.imread(img)

        _ = plt.figure(5)

        plt.plot(
            terminaciones["col"], terminaciones["row"], "b.", label="terminaciones"
        )
        plt.plot(
            bifurcaciones["col"], bifurcaciones["row"], "r.", label="bifurcaciones"
        )

        plt.imshow(img, cmap="gray")
        plt.legend(loc="best")
        plt.title("minutaes GT")

        plt.show()