import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv("./data/processed/data_encoded.csv")

# Mostrar la información general de los datos
print(data.info())

# Verificar si hay valores nulos
print("Valores nulos por columna:")
print(data.isnull().sum())

# Ver la distribución de las variables
data.hist(figsize=(15, 10))
plt.suptitle('Distribución de las Variables')
plt.show()

# Ver la matriz de correlación
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()
