import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Cargar los conjuntos de datos preparados (desde los archivos CSV generados en la Fase 2)
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()  # Convertir a array
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Fase 3.1 - Modelo de Clasificación (Vehículos baratos vs caros)
# ---------------------------------------------------------------

# 2. Crear la variable de clasificación (barato/caro) usando la mediana del precio
precio_median = np.median(y_train)
y_train_class = np.where(y_train > precio_median, 1, 0)  # 1 = caro, 0 = barato
y_test_class = np.where(y_test > precio_median, 1, 0)

# 3. Entrenar un modelo de clasificación con Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_class)

# 4. Predecir los valores del conjunto de prueba
y_pred_class = clf.predict(X_test)

# 5. Evaluar el modelo de clasificación
print("Evaluación del Modelo de Clasificación:")
print(f"Exactitud (Accuracy): {accuracy_score(y_test_class, y_pred_class)}")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_class, y_pred_class))
print("\nInforme de Clasificación:")
print(classification_report(y_test_class, y_pred_class))

# Fase 3.2 - Modelo de Regresión (Predicción exacta del precio)
# -------------------------------------------------------------

# 6. Entrenar un modelo de regresión con Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# 7. Predecir los precios en el conjunto de prueba
y_pred_reg = reg.predict(X_test)

# 8. Evaluar el modelo de regresión
print("\nEvaluación del Modelo de Regresión:")
print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred_reg)}")
print(f"R² (Coeficiente de Determinación): {r2_score(y_test, y_pred_reg)}")

# Opcional: Guardar las predicciones
predicciones_clasificacion = pd.DataFrame({'y_test': y_test_class, 'y_pred_class': y_pred_class})
predicciones_clasificacion.to_csv('predicciones/prediccion_clasificacion.txt', sep='\t', index=False)

predicciones_regresion = pd.DataFrame({'y_test': y_test, 'y_pred_reg': y_pred_reg})
predicciones_regresion.to_csv('predicciones/prediccion_regresion.txt', sep='\t', index=False)

print("Modelado completo.")
