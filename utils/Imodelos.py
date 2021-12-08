import abc
from utils.utils import XY


class Modelo(abc.ABC):

    @abc.abstractmethod
    def __init__(self, caracteristicas: int, clases: int, configuracion: dict) -> None:
        """Constructor del modelo, se espera poder inicializar el modelo aqui"""
        pass

    @property
    def modelo(self):
        """Propiedad get para retornar el modelo"""
        return self._modelo

    @modelo.setter
    def modelo(self, value):
        """Propiedad set para asignar un nuevo modelo dentro de la clase"""
        self._modelo = value

    @abc.abstractmethod
    def obtener_parametros_modelo(self):
        """Implementacion para obtener los parametros o pesos del modelo"""
        pass

    @abc.abstractmethod
    def asignar_parametros_modelo(self, parametros):
        """Asignar los pesos del modelo o parametros, no son los de configuracion"""
        pass

    @abc.abstractmethod
    def entrenar_modelo(self, datos_xy: XY):
        """Metodo para entrenar el modelo, se caracteriza por el uso del metodo fit"""
        pass

    @abc.abstractmethod
    def evaluar_modelo(self, datos_pruebas: XY):
        """Metodo de evaluacion del modelo, esto podria emplear diferentes metricas en su interior"""
        pass
