import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, explained_variance_score
import joblib

# Cargar datos procesados
data = pd.read_csv("./data/processed/data_encoded2.csv")

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=["PorcentajeCumplimiento"])
y = data["PorcentajeCumplimiento"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo XGBoost Regressor
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo usando diferentes métricas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Imprimir resultados
print(f"XGBoost Regressor - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}, MSLE: {msle}, Explained Variance: {explained_variance}")

# Guardar el modelo entrenado
joblib.dump(model, "./models/xgboost_model2.pkl")
