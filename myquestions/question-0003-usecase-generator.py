import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import random

def generar_caso_de_uso_evaluar_cristales():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_cristales.
    """
    
    # 1. Configuración aleatoria y generación de datos simulando química
    n_samples = random.randint(300, 600)
    n_features = random.randint(7, 10)
    
    # Creamos un problema de clasificación de 3 clases de vidrios
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_classes=3, 
        n_informative=4, random_state=random.randint(1, 1000)
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    modelo = GradientBoostingClassifier(random_state=42)
    modelo.fit(input_data['X_train'], input_data['y_train'])
    
    probabilidades = modelo.predict_proba(input_data['X_test'])
    puntuacion_log_loss = log_loss(input_data['y_test'], probabilidades)
    
    output_data = (modelo, puntuacion_log_loss)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_cristales()
    print("=== INPUT (Diccionario) ===")
    print(f"Shape X_train: {entrada['X_train'].shape}")
    print(f"Shape X_test: {entrada['X_test'].shape}")
    
    print("\n=== OUTPUT ESPERADO (Tupla) ===")
    mod, score_loss = salida_esperada
    print(f"Tipo de modelo: {type(mod).__name__}")
    print(f"Log Loss calculado: {score_loss:.4f}")
