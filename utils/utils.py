#Utilidades de la solucion

import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split

# Tipo de datos para llevar las variables independientes y la dependiente
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def revolver(X: np.ndarray, y: np.ndarray) -> XY:
    """Revolver el dataset"""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def particionar(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Primera idea de particionamiento.
    Se deja para posteriores usos"""
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )


def cargar_datos(ruta_archivo: str, div_ratio=0.8) -> Dataset:
    """Cargar los datos de la ruta de procesado
    """
    df_datos = pd.read_csv(ruta_archivo)
    X = df_datos.iloc[:, :-1]  # the last column contains labels
    y = df_datos.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=div_ratio)
    return (x_train, y_train), (x_test, y_test)


def preparar_datos(df_datos) -> pd.DataFrame:
    """Limpieza de los datos, fiebre y comuna se remueven. 
    Sexo para a ser categorica con valores numericos"""
    df_datos.drop("fiebre", axis=1, inplace=True)
    df_datos.drop("comuna", axis=1, inplace=True)
    df_datos["sexo_"] = df_datos["sexo_"].astype("category")
    df_datos["sexo_"] = df_datos["sexo_"].cat.codes
    return df_datos.infer_objects()
