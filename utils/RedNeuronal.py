# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import class_weight
from utils.Imodelos import Modelo
from tensorflow import keras
from utils.utils import XY
from tensorflow.keras.utils import to_categorical

class RedNeuronal(Modelo):
    """Implementacion de red neuronal empleando tensorflow
    capas: 4: 10(relu), 5(relu), 5(relu), 3 (softmax)"""
    def __init__(self, caracteristicas: int, clases: int, configuracion: dict) -> None:
        capas = configuracion["red_neuronal"]
        self.pesos_clases = configuracion["pesos_clases"]
        self.modelo: keras.Sequential = keras.Sequential(capas)
        self.clases = clases
        self.caracteristicas = caracteristicas
        self.modelo.summary()
        # F1 score no disponible por defecto
        metrics = [ keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.Precision(name="precision")
        ]
        self.modelo.compile("adam", "categorical_crossentropy", metrics=metrics)
    def obtener_parametros_modelo(self) :
        """Obtener los pesos de la red"""
        return self.modelo.get_weights()


    def asignar_parametros_modelo(self, parametros):
        self.modelo.set_weights(parametros)

    def entrenar_modelo(self, datos_xy: XY):
        y_prueba = to_categorical(datos_xy[1])
        self.modelo.fit(datos_xy[0], y_prueba,epochs=10, batch_size=32, class_weight=self.pesos_clases)

    def evaluar_modelo(self, datos_pruebas: XY):
        y_prueba = to_categorical(datos_pruebas[1])
        dic_medidas = self.modelo.evaluate(datos_pruebas[0],y_prueba,return_dict=True)
        return dic_medidas["loss"], len(datos_pruebas[1]), dic_medidas
