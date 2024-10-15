import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset
df = pd.read_csv(r"C:/Users/Usuario/Desktop/PROYECTO_M6/Data/ML_cars.csv")

# 2. Mostrar las primeras filas del dataset
print("Primeras 5 filas del dataset:")
print(df.head())

# 3. Resumen de las estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# 4. Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 5. Análisis de variables categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nVariables categóricas y su distribución:")
for col in categorical_cols:
    print(f"\nFrecuencia de {col}:")
    print(df[col].value_counts())

# 6. Visualización: Histograma de precios
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], bins=20, kde=True)
plt.title('Distribución del Precio de los Vehículos')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()

# 7. Visualización: Matriz de correlación para variables numéricas
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Filtrar solo columnas numéricas
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()