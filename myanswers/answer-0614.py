import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest


def preparar_datos_credito(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformer = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(sparse_output=False))
            ]),
            categorical_cols
        )
    ])

    X_proc = transformer.fit_transform(X)

    iso = IsolationForest(
        random_state=42,
        contamination=0.05
    )

    mask = iso.fit_predict(X_proc) == 1

    X_limpio = X_proc[mask]
    y_limpio = y[mask]

    return X_limpio, y_limpio
