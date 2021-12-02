import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, recall_score, confusion_matrix, roc_auc_score
import Imodelos


class RegresionLogistica(Imodelos.Modelo):
    def __init__(self, caracteristicas: int, clases: int, configuracion: dict) -> None:
        self.modelo: LogisticRegression = LogisticRegression(**configuracion)
        self.clases = clases
        self.caracteristicas = caracteristicas
        self.modelo.classes_ = np.array(range(clases))
        self.modelo.coef_ = np.zeros((clases, caracteristicas))
        if self.modelo.fit_intercept:
            self.modelo.intercept_ = np.zeros((clases,))

    def obtener_parametros_modelo(self):
        if self.modelo.fit_intercept:
            return (self.modelo.coef_, self.modelo.intercept_)
        else:
            return (self.modelo.coef_,)

    def asignar_parametros_modelo(self, parametros):
        self.modelo.coef_ = parametros[0]
        if self.modelo.fit_intercept:
            self.modelo.intercept_ = parametros[1]

    def entrenar_modelo(self, datos_xy: utiles.XY):
        self.modelo.fit(datos_xy[0], datos_xy[1])

    def evaluar_modelo(self, datos_pruebas: utiles.XY):
        y_pred = self.modelo.predict_proba(datos_xy[0])
        perdida = log_loss(datos_xy[1], y_pred)
        accuracy = self.modelo.score(datos_xy[0], datos_xy[1])
        f1 = f1_score(datos_xy[1], y_pred)
        matriz_confusion = confusion_matrix(datos_xy[1], y_pred)
        roc_auc = roc_auc_score(datos_xy[1], y_pred)
        recall = recall_score(datos_xy[1], y_pred)
        dic_medidas = {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": matriz_confusion,
                       "recall": recall, "roc_auc": roc_auc}
        return perdida, len(y_pred), dic_medidas
