import flwr as fl
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import warnings
from utils.Imodelos import Modelo
from utils.LogisticRegression import RegresionLogistica
from utils.SupportVectorMachine import SupportVectorMachine
from utils.utils import XY, preparar_datos
from sklearn.model_selection import train_test_split


def procesar_argumentos():
    """Procesamiento de argumentos que se pasan por consola"""
    descripcion = "Programa cliente del modelo de aprendizaje federado"
    argumentos = argparse.ArgumentParser(description=descripcion)
    argumentos.add_argument("--modelo", dest="modelo_a_usar",
                            default="regresion_logistica", type=str)
    argumentos.add_argument("--archivo", dest="archivo", required=True,
                            help="Ruta hasta el archivo e incluyendolo", type=str)
    argumentos.add_argument("--porcentaje-entrenamiento", dest="particiones",
                            help="Porcentaje del dataset para emplear como entrenamiento", type=float, default=0.8)
    argv = argumentos.parse_args()
    modelo = argv.modelo_a_usar
    archivo = argv.archivo
    particiones = argv.particiones
    return modelo, archivo, particiones


def escoger_modelo(tipo_modelo: str, pesos_clases = None) -> Modelo:
    """Escoger uno de los modelos implementados para correr en aprendizaje federado.
    Los parametros para cada modelo están fijos"""
    if tipo_modelo == "regresion_logistica":
        modelo = RegresionLogistica(
            21, 3, {"penalty": "l2", "max_iter": 1, "warm_start": True, "class_weight": pesos_clases})
        return modelo
    if tipo_modelo == "support_vector":
        modelo = SupportVectorMachine(21,3,{  "penalty": "elasticnet", "shuffle": True, "n_jobs": -1, "loss": "log", "class_weight": pesos_clases})
        return modelo
    if tipo_modelo == "red_neuronal_desbalanceo":
        from tensorflow import keras
        from utils.RedNeuronal import RedNeuronal
        modelo = RedNeuronal(21,3,{"red_neuronal": [keras.layers.Dense(
            10, activation="relu", input_shape=(21,)
        ),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax")], "pesos_clases": pesos_clases})
        return modelo
    if tipo_modelo == "bosque_aleatorio":
        raise Exception("No se ha implementado este metodo por dificultades tecnicas")
        #modelo = BosqueAleatorio(20,3,{"class_weight": pesos_clases})
    else:
        raise Exception(
            f"Modelo: {tipo_modelo} no ha sido implementado en el cliente")


class FederadoCliente(fl.client.NumPyClient):
    """Clase del cliente para paso de datos en formato numpy
    el cliente NumpyClient exige mantener los nombres aqui mencionados para los metodos"""
    def __init__(self, modelo: Modelo, xy_entrenamiento: XY, xy_prueba: XY) -> None:
        """Iniciar el cliente con los datos respectivos"""
        super().__init__()
        self.modelo = modelo
        self.xy_entrenamiento = xy_entrenamiento
        self.xy_prueba = xy_prueba

    def get_parameters(self):
        """Obtención de parametros o pesos del modelo"""
        return self.modelo.obtener_parametros_modelo()

    def fit(self, parameters, config):  # type: ignore
        """Entrenamiento del modelo. Primero se asignn los pesos
        Se devuelven los nuevos pesos"""
        self.modelo.asignar_parametros_modelo(parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.modelo.entrenar_modelo(
                self.xy_entrenamiento)
        print(f"Entrenamiento terminado para la ronda {config['rnd']}")
        return self.modelo.obtener_parametros_modelo(), len(self.xy_entrenamiento[0]), {}

    def evaluate(self, parameters, config):
        """Evaluacion del modelo para tomar metricas"""
        self.modelo.asignar_parametros_modelo(parameters)
        #utils.set_model_params(model, parameters)
        salida = self.modelo.evaluar_modelo(self.xy_prueba)
        print(salida)# Metricas del cliente
        return salida


def main():
    """Ejecución del cliente como archivo o proceso independiente"""
    modelo_str, archivo, particiones = procesar_argumentos()
    ruta_archivo = Path(archivo)
    #Preparacion de datos
    df_datos = pd.read_csv(ruta_archivo)
    df_datos = preparar_datos(df_datos)
    X = df_datos.drop("clas_dengue", axis=1).to_numpy()
    df_datos.loc[(df_datos.clas_dengue == 3),'clas_dengue']=2 # Grave etiqueta 2 a 3
    Y = df_datos["clas_dengue"].to_numpy()
    conteos = np.bincount(Y)
    #pesos para las clases
    weight_for_0 = 1.0 / conteos[0]
    weight_for_1 = 1.0 / conteos[1]
    weight_for_3 = 1.0 / conteos[2]
    clases_pesos = {0:weight_for_0,1:weight_for_1, 2:weight_for_3}
    X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=float(particiones))
    xy_entrenamiento = (X_train, y_train)
    xy_pruebas = (X_test, y_test)
    #Modelo para evaluar
    modelo = escoger_modelo(modelo_str,clases_pesos)
    #Inicio del cliente luego de configurar
    fl.client.start_numpy_client("localhost:8090", FederadoCliente(
        modelo, xy_entrenamiento, xy_pruebas))


if __name__ == "__main__":
    main()
