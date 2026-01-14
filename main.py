import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# 1️⃣ Dataset
data = {
    "hora": [8, 18, 14, 20, 9, 7, 22, 16, 10, 19],
    "distancia": [3.0, 5.0, 2.5, 7.0, 4.0, 2.0, 6.5, 4.8, 3.5, 5.5],
    "trafico": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    "tiempo": [10, 25, 9, 35, 14, 8, 32, 22, 12, 28]
}

df = pd.DataFrame(data)

# 2️⃣ Features y target
X = df[["hora", "distancia", "trafico"]]
y = df["tiempo"]

# 3️⃣ Separar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Predicciones sobre TEST
y_pred = model.predict(X_test)

# 6️⃣ Métricas de error
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("Resultados:")
print(f"MAE (error medio absoluto): {mae:.2f} minutos")
print(f"RMSE (error cuadrático medio): {rmse:.2f} minutos")
print("\nPredicciones vs Realidad:")
for real, pred in zip(y_test, y_pred):
    print(f"Real: {real} minutos, Predicción: {pred:.2f} minutos")
