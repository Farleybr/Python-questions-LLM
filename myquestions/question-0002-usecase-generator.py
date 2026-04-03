import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import random

def generar_caso_de_uso_agrupar_senales_sismicas():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función agrupar_senales_sismicas.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(150, 300)
    n_features = random.randint(5, 8)
    n_clusters = random.randint(3, 6)
    
    # 2. Generar datos aleatorios simulando frecuencias y amplitudes
    data = np.random.uniform(0.1, 5.0, (n_rows, n_features))
    df = pd.DataFrame(data, columns=[f'sensor_{i}' for i in range(n_features)])
    
    # Introducir NaNs en ~8% de las filas para forzar la limpieza
    mask = np.random.choice([True, False], size=df.shape, p=[0.08, 0.92])
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
    # A. Limpieza y escalado
    df_clean = input_data['df'].dropna()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # B. Manifold Learning con Isomap
    isomap = Isomap(n_neighbors=5, n_components=2)
    datos_transformados = isomap.fit_transform(df_scaled)
    
    # C. Clustering con KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(datos_transformados)
    
    # D. Evaluación con Davies-Bouldin
    score_db = davies_bouldin_score(datos_transformados, labels)
    
    output_data = (labels, datos_transformados, score_db)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_agrupar_senales_sismicas()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Número de clusters deseado: {entrada['n_clusters']}")
    print(f"Total filas originales (con NaNs): {len(entrada['df'])}")
    
    print("\n=== OUTPUT ESPERADO (Tupla) ===")
    labels_res, matriz_res, db_score_res = salida_esperada
    print(f"Shape de la matriz final (Isomap): {matriz_res.shape}")
    print(f"Muestra de labels: {labels_res[:10]}")
    print(f"Índice Davies-Bouldin: {db_score_res:.4f}")
