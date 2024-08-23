import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, explained_variance_score
import joblib
import numpy as np

# Cargar datos procesados
data = pd.read_csv("./data/processed/data_encoded.csv")

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=["PorcentajeCumplimiento"])
y = data["PorcentajeCumplimiento"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
model = XGBRegressor(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.05, 0.001],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [1, 0.1, 10, 100],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 3, 5]
}


# Configurar el RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                   n_iter=50, scoring='neg_mean_squared_error', 
                                   cv=10, verbose=2, random_state=42, n_jobs=-1)

# Entrenar el modelo
random_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = random_search.best_estimator_

# Realizar predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Evaluación del modelo usando diferentes métricas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Imprimir resultados
print(f"Mejores hiperparámetros: {random_search.best_params_}")
print(f"XGBoost Regressor - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}, MSLE: {msle}, Explained Variance: {explained_variance}")

# Guardar el modelo entrenado
joblib.dump(best_model, "./models/xgboost_optimized_model.pkl")
