import bz2
import io
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


def is_normalized(X):
    min_val = X.min()
    max_val = X.max()
    return (-1.01 <= min_val <= 0) and (0 <= max_val <= 1.01)


# classe caricamento dati
def load_data(path, threshold=1e8):
    full_path = 'Datasets/' + path

    # Gestione file bz2
    if path.endswith('.bz2'):
        with bz2.open(full_path, 'rb') as f:
            X, y = load_svmlight_file(io.BytesIO(f.read()))
    else:
        X, y = load_svmlight_file(full_path)
    y = pd.Series(y).replace({-1: -1, +1: 1, 0: -1})

    if X.shape[0] * X.shape[1] < threshold:
        X = X.toarray()
        sparse = False
    else:
        sparse = True

    # Normalizza solo se necessario
    if not is_normalized(X):
        if sparse:
            scaler = MaxAbsScaler()
        else:
            scaler = RobustScaler()
        X = scaler.fit_transform(X)
        print(f"Normalizzazione applicata a {path}")
    else:
        print(f"{path} è già normalizzato")

    return X, y
