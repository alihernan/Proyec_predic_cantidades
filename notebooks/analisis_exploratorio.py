import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear directorios para guardar los resultados si no existen
os.makedirs("./data/analysis", exist_ok=True)
os.makedirs("./data/analysis/plots", exist_ok=True)

# Cargar los datos desde la carpeta 'raw'
data = pd.read_csv("./data/raw/Data.csv")

# Mostrar las primeras filas del DataFrame para inspeccionar
print(data.head())

# Verificar si hay valores nulos
print(data.isnull().sum())

# Descripción estadística básica de las variables numéricas
numeric_description = data.describe()
print(numeric_description)

# Descripción de las variables categóricas
categorical_description = data.describe(include=['O'])
print(categorical_description)

# Guardar las descripciones estadísticas en archivos CSV
numeric_description.to_csv("./data/analysis/numeric_description.csv")
categorical_description.to_csv("./data/analysis/categorical_description.csv")

# Histograma para la variable objetivo 'PorcentajeCumplimiento'
plt.figure(figsize=(10, 6))
sns.histplot(data['PorcentajeCumplimiento'], kde=True)
plt.title('Distribución de PorcentajeCumplimiento')
plt.xlabel('PorcentajeCumplimiento')
plt.ylabel('Frecuencia')
plt.savefig("./data/analysis/plots/histogram_PorcentajeCumplimiento.png")
plt.close()

# Boxplots para variables categóricas
categorical_columns = ['Estructura', 'TipoProducto', 'TipoBolsa', 'TipoSellado', 'GrupoFormato', 'Impreso', 'Calibre', 'Valvula', 'GradoTroquel', 'UsoZipper']

for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=col, y='PorcentajeCumplimiento', data=data)
    plt.title(f'Boxplot de {col} vs PorcentajeCumplimiento')
    plt.xticks(rotation=45)
    plt.savefig(f"./data/analysis/plots/boxplot_{col}.png")
    plt.close()

# Guardar un resumen de las estadísticas descriptivas en un archivo
data.describe(include='all').to_csv("./data/analysis/statistical_analysis.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats

# Cargar los datos procesados
categorical_description = pd.read_csv("categorical_description.csv")
numeric_description = pd.read_csv("numeric_description.csv")
statistical_analysis = pd.read_csv("statistical_analysis.csv")

# Mostrar algunos resultados para asegurar la correcta carga
print(categorical_description.head())
print(numeric_description.head())
print(statistical_analysis.head())

# Paso 1: Correlaciones no lineales
spearman_corr = statistical_analysis.corr(method='spearman')
plt.figure(figsize=(12, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación de Spearman')
plt.savefig("./data/analysis/plots/spearman_correlation_matrix.png")
plt.close()

# Paso 2: Cálculo de VIF
X = statistical_analysis.select_dtypes(include=[float, int]).dropna()
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif_data_sorted = vif_data.sort_values(by="VIF", ascending=False)
print(vif_data_sorted)
vif_data_sorted.to_csv("./data/analysis/vif_analysis.csv", index=False)

# Paso 3: Análisis de Outliers
z_scores = np.abs(stats.zscore(statistical_analysis['PorcentajeCumplimiento']))
outliers = statistical_analysis[z_scores > 3]
outliers.to_csv("./data/analysis/outliers.csv", index=False)

# Paso 4: PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(statistical_analysis.drop(columns=['PorcentajeCumplimiento']))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=statistical_analysis['PorcentajeCumplimiento'], cmap='coolwarm')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.colorbar(label='PorcentajeCumplimiento')
plt.title('PCA de los datos')
plt.savefig("./data/analysis/plots/pca_plot.png")
plt.close()

# Paso 5: Evaluación de la Linealidad
sns.pairplot(statistical_analysis)
plt.savefig("./data/analysis/plots/pairplot_analysis.png")
plt.close()
