import pandas as pd
import os

# Asegurarse de que la carpeta 'processed' existe
os.makedirs("./data/processed", exist_ok=True)

# Cargar los datos desde la carpeta 'raw'
data = pd.read_csv("./data/raw/Data2.csv")


# Definir las columnas categóricas que serán codificadas
categorical_columns = ['Estructura', 'TipoProducto', 'TipoBolsa', 'TipoSellado',
                       'GrupoFormato', 'Impreso', 'Calibre', 'Valvula', 'GradoTroquel', 'UsoZipper']

# Realizar One-Hot Encoding en las columnas categóricas
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Guardar los datos procesados
data_encoded.to_csv("./data/processed/data_encoded2.csv", index=False)

print("Procesamiento de datos completado y archivo guardado en './data/processed/data_encoded.csv'")
