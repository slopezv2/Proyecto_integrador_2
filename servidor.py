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
    pass


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs) -> None:
        super(SaveModelStrategy, self).__init__(*args, **kwargs)
        self.ruta_pesos_modelo = kwargs.get("ruta_modelo")

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(Path(self.ruta_pesos_modelo).joinpath(
                f"ronda-{rnd}-pesos.npz"), *aggregated_weights)
        return aggregated_weights


def main():
    ruta_modelo, rondas = procesar_argumentos()
    # Create strategy and run server
    strategy = SaveModelStrategy(
        ruta_modelo=ruta_modelo,

        # (same arguments as FedAvg here)
    )
    fl.server.start_server(strategy=strategy, config={"num_rounds": rondas})
