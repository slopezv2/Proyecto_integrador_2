from typing import List, Optional, Tuple
import flwr as fl
import argparse
from pathlib import Path
import numpy as np
from utils.Imodelos import Modelo


def enviar_ronda(rnd: int) -> dict:
    """Enviar la ronda al cliente"""
    return {"rnd": rnd}


def procesar_argumentos():
    """Funcion para leer los argumentos y obtener los parametros por consola para ejecutar  el programa servidor"""
    descripcion = "Programa servidor del modelo de aprendizaje federado"
    argumentos = argparse.ArgumentParser(description=descripcion)
    argumentos.add_argument("--ruta_modelos", dest="ruta_modelos",
                            required=True, type=str, help="Ruta para guardar los pesos de los modelos")
    argumentos.add_argument("--rondas", dest="rondas", required=True,
                            help="Cantidad de rondas, por defecto son 5", type=str, default=5)
    argv = argumentos.parse_args()
    ruta_modelo = argv.ruta_modelos
    rondas = argv.rondas
    return ruta_modelo, rondas


def obtener_funcion_evaluacion(modelo: Modelo):
    """En caso de necesitas implementar una evaluación centralizada del modelo"""
    pass


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Clase especial para implementar la estrategia de evaluación del metodo.
    Permite guardar los pesos de cada modelo para un posterior uso"""
    def __init__(self, *args, **kwargs) -> None:
        self.ruta_pesos_modelo = kwargs.get("ruta_modelo") # ruta para guardar los modelos
        kwargs.pop("ruta_modelo")
        super(SaveModelStrategy, self).__init__(*args, **kwargs) # seguir lo mismo de la estrategia FedAVG

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        """Metodo sobrecargado para guardar el modelo"""
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Guardando la ronda {rnd}")
            np.savez(Path(self.ruta_pesos_modelo).joinpath(
                f"ronda-{rnd}-pesos.npz"), *aggregated_weights) #Guardar pesos modelos
        return aggregated_weights


def main():
    ruta_modelo, rondas = procesar_argumentos()
    # Create strategy and run server
    strategy = SaveModelStrategy(
        ruta_modelo=ruta_modelo,
        min_available_clients=10,
        min_fit_clients=10,
        on_fit_config_fn=enviar_ronda
        # (same arguments as FedAvg here)
    )
    fl.server.start_server(server_address="localhost:8090",
                           strategy=strategy, config={"num_rounds": int(rondas)})


if __name__ == "__main__":
    main()
