from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from main import KMeans, KMeansPP


def get_data(
    name: Literal["banknote", "vehicle", "ecoli", "wineqr", "EEG Eye State"]
) -> np.ndarray:
    """
    Get real-world data

    source: https://github.com/milaan9/Clustering-Datasets/tree/master/01.%20UCI
    """
    if name == "banknote":
        df = pd.read_csv("data/banknote.csv")
        df = df.drop(columns=["class"])
    elif name == "vehicle":
        df = pd.read_csv("data/vehicle.csv")
        df = df.drop(columns=["Class"])
    elif name == "ecoli":
        df = pd.read_csv("data/ecoli.csv")
        df = df.drop(columns=["class"])
    elif name == "wineqr":
        df = pd.read_csv("data/wineqr.csv")
    elif name == "EEG Eye State":
        df = pd.read_csv("data/EEG Eye State.csv")
        df = df.drop(columns=["eyeDetection"])
    else:
        raise ValueError("Invalid dataset name")

    return df.to_numpy()


def plot_elbow_curve(k_min: int, k_max: int, X: np.ndarray):
    k_list = np.array([])
    k_list_pp = np.array([])

    for k in range(k_min, k_max):
        kmeans = KMeans(k)
        kmeans.fit(X)

        kmeanspp = KMeansPP(k)
        kmeanspp.fit(X)

        k_list = np.append(k_list, kmeans.potential[-1])
        k_list_pp = np.append(k_list_pp, kmeanspp.potential[-1])

    x = np.arange(k_min, k_max)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x, k_list)
    ax[0].set_title("KMeans")
    ax[1].plot(x, k_list_pp)
    ax[1].set_title("KMeans++")
    plt.show()
