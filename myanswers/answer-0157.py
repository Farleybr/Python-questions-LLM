import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def coeficientes_estandarizados(
    df: pd.DataFrame,
    target_col: str
) -> np.ndarray:
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LinearRegression()
    modelo.fit(X_scaled, y)

    coefs = modelo.coef_

    coefs_ordenados = coefs[np.argsort(np.abs(coefs))[::-1]]

    return coefs_ordenados
