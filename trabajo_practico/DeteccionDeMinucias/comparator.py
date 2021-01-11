import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    gt = "./GT-huellas/110_7.txt"
    extracted = "./Minutae_Extraction/110_7.txt"

    minuciasGT = pd.read_csv(gt, sep=";", index_col=0, header=0)
    minuciasEX = pd.read_csv(extracted, sep=";", index_col=0, header=0)

    minutaesDetected = []
    minutaesOK = []

    for index, minutae in minuciasGT.iterrows():
        # Es minucia si está a 10 pixel o menos
        isMinutae = minuciasEX[
            (abs(minuciasEX["row"] - minutae["row"]) < 10)
            & (abs(minuciasEX["col"] - minutae["col"]) < 10)
        ]

        # Es minucia del mismo tipo si coincide la etiqueta
        isOk = isMinutae[isMinutae["label"] == minutae["label"]]

        # Se guardan los datos de las minucias
        minutaesDetected.append(len(isMinutae))
        minutaesOK.append(len(isOk))

    print(
        "Se ven {} minucias, de las cuales son detectadas {}, y bien detectadas {}. Se extraen {} minucias. Un {}\% de diferencia.".format(
            len(minuciasGT),
            len(minutaesDetected) - minutaesDetected.count(0),
            len(minutaesOK) - minutaesOK.count(0),
            len(minuciasEX),
            len(minuciasEX) * 100 / len(minuciasGT),
        )
    )


# if __name__ == "__main__":
#     img = "./huellasFVC2004/101_3.tif"

#     minucias = pd.read_csv("./GT-huellas/101_3.txt", sep=";", index_col=0, header=0)
#     # minucias.columns = ["col", "row", "label"]

#     terminaciones = minucias[minucias["label"] == 1].drop(columns="label")
#     bifurcaciones = minucias[minucias["label"] == 3].drop(columns="label")

#     if True:
#         img = plt.imread(img)

#         _ = plt.figure(5)

#         plt.plot(
#             terminaciones["col"], terminaciones["row"], "b.", label="terminaciones"
#         )
#         plt.plot(
#             bifurcaciones["col"], bifurcaciones["row"], "r.", label="bifurcaciones"
#         )

#         plt.imshow(img, cmap="gray")
#         plt.legend(loc="best")
#         plt.title("minutaes GT")

#         plt.show()
