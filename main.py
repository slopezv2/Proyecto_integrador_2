import argparse
import glob
import subprocess
from sys import stdout


def procesar_argumentos():
    descripcion = "Programa Controlador, maneja el Servidor y Cliente, solo sirve con modelos scikit-learn"
    argumentos = argparse.ArgumentParser(description=descripcion)
    argumentos.add_argument("--ruta_modelos", dest="ruta_modelos",
                            required=True, type=str, help="Ruta para guardar los pesos de los modelos")
    argumentos.add_argument("--rondas", dest="rondas", required=True,
                            help="Cantidad de rondas, por defecto son 5", type=str, default=5)
    argumentos.add_argument("--particiones", dest="particiones",
                            help="cantidad de particiones para dividir los datos, una se una para entrenar y otra para evaluar, las demas no se usan en esta iteracion", type=int, default=5)
    argumentos.add_argument("--ruta_carpeta", dest="ruta_carpeta",
                            required=True, type=str, help="Ruta para leer los archivos y simular un cliente independiente por cada archivo")
    argumentos.add_argument("--modelo", dest="modelo_a_usar",
                            default="regresion_logistica", type=str, help="El modelo empleado, por defecto es regresion_logistica")
    argv = argumentos.parse_args()
    ruta_modelo = argv.ruta_modelos
    rondas = argv.rondas
    particiones = argv.particiones
    ruta_carpeta = argv.ruta_carpeta
    modelo = argv.modelo_a_usar
    return ruta_modelo, rondas, particiones, ruta_carpeta, modelo


def main():
    procesos_cliente = []
    ruta_modelo, rondas, particiones, ruta_carpeta, modelo = procesar_argumentos()
    archivos = glob.glob(f"{ruta_carpeta}/*")
    ejecutar_servidor = ["python", "servidor.py",
                         "--ruta_modelos", ruta_modelo, "--rondas", rondas]
    ejecutar_clientes = ["python", "cliente.py",
                         "--modelo", modelo, "--particiones", str(particiones), "--archivo"]
    with open("salida/servidor_log.txt", "w+") as log:
        p_servidor = subprocess.Popen(
            ejecutar_servidor, stdout=log, stderr=log)
        for archivo in archivos:
            comando = ejecutar_clientes + [archivo]
            print(comando)
            proceso = subprocess.Popen(comando)
            procesos_cliente.append(proceso)
        p_servidor.wait()


if __name__ == "__main__":
    main()
