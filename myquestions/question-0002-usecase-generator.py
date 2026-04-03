import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
import random

def generar_caso_de_uso_agrupar_especies_marinas():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función agrupar_especies_marinas.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(120, 250)
    n_features = random.randint(4, 7)
    n_clusters = random.randint(3, 5)
    
    # 2. Generar datos aleatorios y meter NaNs
    data = np.random.uniform(10, 100, (n_rows, n_features))
    df = pd.DataFrame(data, columns=[f'medida_{i}' for i in range(n_features)])
    
    # Introducir NaNs en ~5% de las filas
    mask = np.random.choice([True, False], size=df.shape, p=[0.05, 0.95])
    df[mask] = np.nan
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'n_clusters': n_clusters
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    df_clean = input_data['df'].dropna()
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
    datos_transformados = kpca.fit_transform(df_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(datos_transformados)
    
    output_data = (kmeans.labels_, datos_transformados)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_agrupar_especies_marinas()
    print("=== INPUT (Diccionario) ===")
    print(f"Número de clusters: {entrada['n_clusters']}")
    print(f"Total filas originales (con NaNs): {len(entrada['df'])}")
    
    print("\n=== OUTPUT ESPERADO (Tupla) ===")
    labels, matriz = salida_esperada
    print(f"Shape de la matriz final (KPCA): {matriz.shape}")
    print(f"Muestra de labels: {labels[:10]}")
