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
