import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import RidgeClassifier
import random

def generar_caso_de_uso_predecir_calidad_vino():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función predecir_calidad_vino.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(150, 300)
    n_features = random.randint(8, 12) 
    
    # 2. Generar datos aleatorios con algunos outliers
    data = np.random.normal(0, 1, (n_rows, n_features))
    
    # Introducir outliers artificiales
    for _ in range(5):
        row_idx = random.randint(0, n_rows-1)
        col_idx = random.randint(0, n_features-1)
        data[row_idx, col_idx] = random.choice([-20, 20])
        
    feature_cols = [f'quimico_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)
    
    target_col = 'calidad'
    df[target_col] = np.random.choice([0, 1, 2], size=n_rows)
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    X_expected = input_data['df'].drop(columns=[target_col])
    y_expected = input_data['df'][target_col]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_expected)
    
    selector = SelectPercentile(score_func=f_classif, percentile=50)
    X_selected = selector.fit_transform(X_scaled, y_expected)
    
    modelo = RidgeClassifier(random_state=42)
    modelo.fit(X_selected, y_expected)
    
    output_data = (modelo, scaler, selector)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_predecir_calidad_vino()
    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print(f"Shape del DF: {entrada['df'].shape}")
    
    print("\n=== OUTPUT ESPERADO (Tupla) ===")
    mod, esc, sel = salida_esperada
    print(f"Tipo de modelo: {type(mod).__name__}")
    print(f"Características seleccionadas: {len(sel.get_support(indices=True))} de {entrada['df'].shape[1]-1}")
