from sklearn.metrics import log_loss, f1_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from utils.utils import  XY
from utils.Imodelos import Modelo
import numpy as np

class BosqueAleatorio(Modelo):
    def __init__(self, caracteristicas: int, clases: int, configuracion: dict) -> None:
        self.modelo: RandomForestClassifier = RandomForestClassifier(**configuracion)
        self.clases = clases
        self.caracteristicas = caracteristicas
        self.modelo.classes_ = np.array(range(clases))

    def obtener_parametros_modelo(self) :
        parametros = self.modelo.get_params()
        print(parametros)
        return parametros

    def asignar_parametros_modelo(self, parametros):
        self.modelo.set_params(parametros)

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