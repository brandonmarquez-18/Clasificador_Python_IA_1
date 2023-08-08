from operator import eq
import csv  # Importa el módulo csv para trabajar con archivos CSV.
import pandas as pd  # Importa el módulo pandas y se le asigna el alias 'pd'.
import numpy as np  # Importa el módulo numpy y se le asigna el alias 'np'.

# Lee el archivo CSV "Data2.csv" y crea un DataFrame 'df' con las columnas 0, 1 y 2.
df = pd.read_csv("Data2.csv", usecols=[0, 1, 2])

# Lee el archivo CSV "Data2.csv" y crea un DataFrame 'ultimacolumna' con la columna 3.
ultimacolumna = pd.read_csv("Data2.csv", usecols=[3])

# Convierte el DataFrame 'df' a una matriz NumPy y la guarda en la variable 'x'.
x = df.to_numpy()

# Convierte el DataFrame 'ultimacolumna' a una matriz NumPy y la guarda en la variable 'etiquetas'.
etiquetas = ultimacolumna.to_numpy()

# Define tres listas 'w1', 'w2' y 'w3' con valores numéricos para utilizar como pesos.
w1 = [-0.41, 1, -0.53]
w2 = [0.61, 1, -0.73]
w3 = [-0.27, 1, 0]

# Abre el archivo CSV "Data2.csv" nuevamente para leer sus contenidos utilizando el lector csv.
with open('Data2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        # Imprime cada fila del archivo CSV separada por comas y espacios.
        print(', '.join(row))

# Definición de una matriz 'x' que contiene cuatro vectores de tres elementos cada uno.
x = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]

# Definición de la función 'Clasificador' que toma como entrada una matriz 'x' y tres vectores 'w1', 'w2' y 'w3'.


def Clasificador(x, w1, w2, w3):
    # Inicializa una lista vacía 'clase' para almacenar los resultados de clasificación.
    clase = []
    for i in range(len(x)):  # Itera sobre los índices de la matriz 'x'.
        # Calcula el producto escalar entre 'x[i]' y 'w1'.
        h1 = np.dot(x[i], w1)
        # Calcula el producto escalar entre 'x[i]' y 'w2'.
        h2 = np.dot(x[i], w2)
        # Calcula el producto escalar entre 'x[i]' y 'w3'.
        h3 = np.dot(x[i], w3)
        # Imprime el valor del resultado del producto escalar 'h1'.
        print('h1: ', h1)
        # Imprime el valor del resultado del producto escalar 'h2'.
        print('h2: ', h2)
        # Imprime el valor del resultado del producto escalar 'h3'.
        print('h3: ', h3)

        # Clasificación de los vectores 'x[i]' según los valores de los productos escalares 'h1', 'h2' y 'h3'.
        if h1 >= 0 and h2 >= 0 and h3 >= 0:  # Condición 1
            clase.append(0)  # Agrega el valor 0 a la lista 'clase'.
        elif h1 < 0 and h2 < 0 and h3 >= 0:  # Condición 2
            clase.append(0)  # Agrega el valor 0 a la lista 'clase'.
        elif h1 < 0 and h2 >= 0 and h3 >= 0:  # Condición 3
            clase.append(1)  # Agrega el valor 1 a la lista 'clase'.
        elif h1 < 0 and h2 >= 0 and h3 < 0:  # Condición 5
            clase.append(1)  # Agrega el valor 1 a la lista 'clase'.
        elif h1 >= 0 and h2 < 0 and h3 >= 0:  # Condición 6
            clase.append(0)  # Agrega el valor 0 a la lista 'clase'.
        else:  # Condición 4
            clase.append(1)  # Agrega el valor 1 a la lista 'clase'.
    # Devuelve la lista 'clase' con los resultados de clasificación.
    return clase


# Convierte nuevamente el DataFrame 'df' a una matriz NumPy y lo guarda en la variable 'x'.
x = df.to_numpy()

# Llama a la función 'Clasificador' con los argumentos 'x', 'w1', 'w2' y 'w3' y guarda los resultados en 'prediccion'.
prediccion = Clasificador(x, w1, w2, w3)

# Imprime los resultados de la clasificación 'prediccion'.
print(prediccion)

# Importa la función 'eq' del módulo 'operator' para comparar las etiquetas y las predicciones.

# Calcula el rendimiento del clasificador comparando las etiquetas originales 'etiquetas' con las predicciones 'prediccion'.
rendimiento = sum(map(eq, etiquetas, prediccion))

# Imprime el rendimiento del clasificador en porcentaje.
print("Rendimiento del clasificador: ", rendimiento / len(x) * 100, "%")
