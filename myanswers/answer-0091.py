import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def segmentar_usuarios(df, n_componentes, n_clusters):
    df_clean = df.dropna().reset_index(drop=True)

    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(df_clean.values)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    etiquetas = kmeans.fit_predict(X_pca)

    df_clean["Segmento"] = etiquetas

    return df_clean
