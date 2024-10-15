import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Crear la carpeta 'data' si no existe
if not os.path.exists('data'):
    os.makedirs('data')

# 1. Cargar el dataset
df = pd.read_csv(r"C:/Users/Usuario/Desktop/PROYECTO_M6/Data/ML_cars.csv")

# 2. Manejo de valores nulos (solo columnas numéricas)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 3. Codificación de variables categóricas
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Escalado de variables numéricas
scaler = StandardScaler()
numeric_cols_encoded = df_encoded.select_dtypes(include=['float64', 'int64']).columns
df_encoded[numeric_cols_encoded] = scaler.fit_transform(df_encoded[numeric_cols_encoded])

# 5. Separar las características (X) y la variable objetivo (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 6. División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Guardar los conjuntos en archivos CSV dentro de la carpeta 'data'
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Preparación de datos completa.")