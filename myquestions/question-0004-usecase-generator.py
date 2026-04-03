import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import random

def generar_caso_de_uso_reducir_reportes_texto():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función reducir_reportes_texto.
    """
    
    # 1. Configuración aleatoria de textos simulando reportes
    palabras_falla = ["fuga", "aceite", "motor", "ruido", "vibracion", "temperatura", "filtro", "roto", "presion"]
    n_rows = random.randint(150, 300)
    n_componentes = random.randint(3, 6)
    
    reportes = []
    for _ in range(n_rows):
        longitud = random.randint(5, 12)
        frase = " ".join(random.choices(palabras_falla, k=longitud))
        reportes.append(frase)
        
    df = pd.DataFrame({'reporte_mecanico': reportes})
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'col_texto': 'reporte_mecanico',
        'n_componentes': n_componentes
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    X_texto = input_data['df'][input_data['col_texto']]
    
    vectorizer = TfidfVectorizer(max_features=500)
    matriz_tfidf = vectorizer.fit_transform(X_texto)
    
    svd = TruncatedSVD(n_components=input_data['n_componentes'], random_state=42)
    matriz_reducida = svd.fit_transform(matriz_tfidf)
    varianza = svd.explained_variance_ratio_
    
    output_data = (matriz_reducida, varianza)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_reducir_reportes_texto()
    print("=== INPUT (Diccionario) ===")
    print(f"Columna de texto: {entrada['col_texto']}")
    print(f"Componentes deseados: {entrada['n_componentes']}")
    print(f"Ejemplo de datos:\n{entrada['df'].head(2)}")
    
    print("\n=== OUTPUT ESPERADO (Tupla) ===")
    mat_red, var_exp = salida_esperada
    print(f"Shape matriz SVD resultante: {mat_red.shape}")
    print(f"Varianza por componente: {np.round(var_exp, 3)}")
