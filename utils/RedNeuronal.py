# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:56:21 2021

@author: Merqueo
"""
import numpy as np
from utils.Imodelos import Modelo
from tensorflow import keras
from utils.utils import XY

class RedNeuronal(Modelo):
    def __init__(self, caracteristicas: int, clases: int, configuracion: dict) -> None:
        capas = configuracion["red_neuronal"]
        self.pesos_clases = configuracion["pesos_clases"]
        self.modelo: keras.Sequential = keras.Sequential(capas)
        self.clases = clases
        self.caracteristicas = caracteristicas
        self.modelo.summary()
        metrics = [ keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc")
        ]
        self.modelo.compile("adam", "sparse_categorical_crossentropy", metrics=metrics)
    def obtener_parametros_modelo(self) :
        return self.modelo.get_weights()


    def asignar_parametros_modelo(self, parametros):
        self.modelo.set_weights(parametros)

    def entrenar_modelo(self, datos_xy: XY):
        self.modelo.fit(datos_xy[0], datos_xy[1],epochs=20, batch_size=32, class_weight= self.pesos_clases)

    def evaluar_modelo(self, datos_pruebas: XY):
        self.xy_prueba
        perdida, accuracy = self.modelo.evaluate(datos_pruebas[0], datos_pruebas[1])
        dic_medidas = self.modelo.evaluate(datos_pruebas[0],datos_pruebas[1],return_dict=True)
        return perdida, len(datos_pruebas[1]), dic_medidas
