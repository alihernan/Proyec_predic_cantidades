import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Cargar los datos procesados
data = pd.read_csv("./data/raw/Data.csv")

# Paso 1: Correlaciones no lineales (Spearman entre la variable numérica y las categóricas)
data_encoded = pd.get_dummies(data.drop(columns=['PorcentajeCumplimiento']), drop_first=True)
data_encoded['PorcentajeCumplimiento'] = data['PorcentajeCumplimiento']

spearman_corr = data_encoded.corr(method='spearman')
plt.figure(figsize=(12, 8))
sns.heatmap(spearman_corr[['PorcentajeCumplimiento']].sort_values(by='PorcentajeCumplimiento', ascending=False), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación de Spearman con PorcentajeCumplimiento')
plt.savefig("./data/analysis/plots/spearman_correlation_matrix.png")
plt.close()

# Paso 2: Análisis de Outliers
z_scores = np.abs(stats.zscore(data['PorcentajeCumplimiento']))
outliers = data[z_scores > 3]
outliers.to_csv("./data/analysis/outliers.csv", index=False)

# Paso 3: Boxplots por categoría para ver la relación con PorcentajeCumplimiento
categorical_columns = data.drop(columns=['PorcentajeCumplimiento']).columns

for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=col, y='PorcentajeCumplimiento', data=data)
    plt.title(f'Boxplot de {col} vs PorcentajeCumplimiento')
    plt.xticks(rotation=45)
    plt.savefig(f"./data/analysis/plots/boxplot_{col}.png")
    plt.close()

# Paso 4: Distribución de PorcentajeCumplimiento
plt.figure(figsize=(10, 6))
sns.histplot(data['PorcentajeCumplimiento'], kde=True)
plt.title('Distribución de PorcentajeCumplimiento')
plt.xlabel('PorcentajeCumplimiento')
plt.ylabel('Frecuencia')
plt.savefig("./data/analysis/plots/histogram_PorcentajeCumplimiento.png")
plt.close()
