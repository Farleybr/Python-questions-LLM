import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score


def generar_caso_de_uso_fallas(df=None, test_size=0.25, *args, **kwargs):
    if df is None and "df" in kwargs:
        df = kwargs["df"]

    if "test_size" in kwargs:
        test_size = kwargs["test_size"]

    if df is not None:
        metricas, importancia, modelo = detectar_fallas(df, test_size=test_size)

        return {
            "metricas": metricas,
            "importancia_variables": importancia,
            "modelo_entrenado": modelo
        }

    n = random.randint(1000, 2000)

    vibracion = np.random.normal(5, 1.5, n)
    temperatura = np.random.normal(75, 8, n)
    presion = np.random.normal(250, 20, n)
    horas = np.random.uniform(0, 5000, n)
    combustible = np.random.uniform(10, 100, n)

    prob_falla = (
        0.03 * vibracion +
        0.02 * (temperatura - 70) +
        0.0003 * horas -
        0.01 * combustible
    )

    prob_falla = 1 / (1 + np.exp(-prob_falla))

    threshold = np.percentile(prob_falla, 95)
    falla = (prob_falla >= threshold).astype(int)

    df = pd.DataFrame({
        "vibracion": vibracion,
        "temperatura_motor": temperatura,
        "presion_hidraulica": presion,
        "horas_uso": horas,
        "nivel_combustible": combustible,
        "falla": falla
    })

    test_size = round(random.uniform(0.2, 0.3), 2)

    metricas, importancia, modelo = detectar_fallas(df, test_size=test_size)

    input_data = {
        "df": df.copy(),
        "test_size": test_size
    }

    output_data = {
        "metricas": metricas,
        "importancia_variables": importancia,
        "modelo_entrenado": modelo
    }

    return input_data, output_data


def detectar_fallas(df=None, test_size=0.25, *args, **kwargs):
    if df is None and "df" in kwargs:
        df = kwargs["df"]

    if "test_size" in kwargs:
        test_size = kwargs["test_size"]

    if df is None:
        input_data, output_data = generar_caso_de_uso_fallas()
        return (
            output_data["metricas"],
            output_data["importancia_variables"],
            output_data["modelo_entrenado"]
        )

    X = df.drop(columns=["falla"])
    y = df["falla"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

    importancia = dict(zip(X.columns, modelo.feature_importances_))

    return metricas, importancia, modelo
