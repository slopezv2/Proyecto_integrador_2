# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:56:21 2021

@author: Merqueo
"""
import numpy as np


from utils.Imodelos import Modelo
from sklearn.neural_network import MLPClassifier

class RedNeural(Modelo):
    def __init__(self, caracteristicas: int, clases: int, configuracion: dict) -> None:
        self.modelo: MLPClassifier = MLPClassifier(**configuracion)
        self.clases = clases
        self.caracteristicas = caracteristicas
        self.modelo.classes_ = np.array(range(clases))
        self.modelo.coef_ = np.zeros((clases, caracteristicas))
        if self.modelo.fit_intercept:
            self.modelo.intercept_ = np.zeros((clases,))

    def obtener_parametros_modelo(self) :
        if self.modelo.fit_intercept:
            return (self.modelo.coef_, self.modelo.intercept_)
        else:
            return (self.modelo.coef_,)


    def asignar_parametros_modelo(self, parametros):
        self.modelo.coef_ = parametros[0]
        if self.modelo.fit_intercept:
            self.modelo.intercept_ = parametros[1]

    def entrenar_modelo(self, datos_xy: XY):
        self.modelo.fit(datos_xy[0], datos_xy[1])

    def evaluar_modelo(self, datos_pruebas: XY):
        y_pred = self.modelo.predict(datos_pruebas[0])
        y_pred_proba = self.modelo.predict_proba(datos_pruebas[0])
        perdida = log_loss(datos_pruebas[1], y_pred_proba)
        accuracy = self.modelo.score(datos_pruebas[0], datos_pruebas[1])
        f1 = f1_score(datos_pruebas[1], y_pred, average="macro")
        #matriz_confusion = confusion_matrix(datos_pruebas[1], y_pred)
        roc_auc = roc_auc_score(
            datos_pruebas[1], y_pred_proba, average="macro", multi_class="ovo")
        recall = recall_score(datos_pruebas[1], y_pred, average="macro")
        dic_medidas = {"accuracy": accuracy, "f1_score": f1,
                       "recall": recall, "roc_auc": roc_auc}
        return perdida, len(y_pred), dic_medidas
