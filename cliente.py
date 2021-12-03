import flwr as fl
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import warnings
from utils.Imodelos import Modelo
from utils.LogisticRegression import RegresionLogistica

from utils.utils import XY, particionar, preparar_datos


def procesar_argumentos():
    descripcion = "Programa cliente del modelo de aprendizaje federado"
    argumentos = argparse.ArgumentParser(description=descripcion)
    argumentos.add_argument("--modelo", dest="modelo_a_usar",
                            default="regresion_logistica", type=str)
    argumentos.add_argument("--archivo", dest="archivo", required=True,
                            help="Ruta hasta el archivo e incluyendolo", type=str)
    argumentos.add_argument("--particiones", dest="particiones",
                            help="cantidad de particiones", type=int, default=10)
    argv = argumentos.parse_args()
    modelo = argv.modelo_a_usar
    archivo = argv.archivo
    particiones = argv.particiones
    return modelo, archivo, particiones


def escoger_modelo(tipo_modelo: str) -> Modelo:
    if tipo_modelo == "regresion_logistica":
        modelo = RegresionLogistica(
            20, 3, {"penalty": "l2", "max_iter": 1, "warm_start": True})
        return modelo
    else:
        raise Exception(
            f"Modelo: {tipo_modelo} no ha sido implementado en el cliente")


class FederadoCliente(fl.client.NumPyClient):

    def __init__(self, modelo: Modelo, xy_entrenamiento: XY, xy_prueba: XY) -> None:
        super().__init__()
        self.modelo = modelo
        self.xy_entrenamiento = xy_entrenamiento
        self.xy_prueba = xy_prueba

    def get_parameters(self):  # type: ignore
        return self.modelo.obtener_parametros_modelo()

    def fit(self, parameters, config):  # type: ignore
        self.modelo.asignar_parametros_modelo(parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.modelo.entrenar_modelo(
                self.xy_entrenamiento)
        print(f"Entrenamiento terminado para la ronda {config['rnd']}")
        return self.modelo.obtener_parametros_modelo(), len(self.xy_entrenamiento[0]), {}
        # return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        self.modelo.asignar_parametros_modelo(parameters)
        #utils.set_model_params(model, parameters)
        return self.modelo.evaluar_modelo(self.xy_prueba)
        #loss = log_loss(y_test, model.predict_proba(X_test))
        #accuracy = model.score(X_test, y_test)
        # return loss, len(X_test), {"accuracy": accuracy}


def main():
    archivo, modelo_str, particiones = procesar_argumentos()
    ruta_archivo = Path(archivo)
    df_datos = pd.read_csv(ruta_archivo)
    df_datos = preparar_datos(df_datos)
    X = df_datos.drop("clas_dengue", axis=1).to_numpy()
    Y = df_datos["clas_dengue"].to_numpy()
    particion_id = np.random.choice(particiones)
    particiones_datos = particionar(
        X, Y, particiones)
    xy_entrenamiento = particiones_datos[particion_id]
    particion_id = (particion_id + 1) % particiones
    xy_pruebas = particiones_datos[particion_id]
    modelo = escoger_modelo(modelo_str)
    fl.client.start_numpy_client("0.0.0.0:8080", FederadoCliente(
        modelo, xy_entrenamiento, xy_pruebas))


if __name__ == "__main__":
    main()
