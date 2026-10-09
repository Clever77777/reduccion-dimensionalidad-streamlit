# -*- coding: utf-8 -*-
"""
Compara el desempeño de una Regresión Logística usando las variables
originales y los datos reducidos con PCA, LDA y SelectKBest.

Uso:
    python comparacion_metodos.py
    python comparacion_metodos.py --sin-duration
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/bank-additional-full.csv")
parser.add_argument("--sin-duration", action="store_true",
                    help="Excluye 'duration' (solo se conoce al terminar la llamada)")
args = parser.parse_args()

df = pd.read_csv(args.csv, sep=None, engine="python")
y = (df["y"] == "yes").astype(int)
X = df.drop(columns=["y"]).select_dtypes(include=[np.number])
if args.sin_duration:
    X = X.drop(columns=["duration"])

print(f"Variables numéricas ({X.shape[1]}): {', '.join(X.columns)}\n")

# La reducción va DENTRO del pipeline: se ajusta solo con el train de cada fold
escenarios = [("Original", X.shape[1], [])]
escenarios += [("PCA", k, [("pca", PCA(n_components=k))]) for k in [1, 2, 3, 5]]
escenarios += [("LDA", 1, [("lda", LDA(n_components=1))])]
escenarios += [("SelectKBest", k, [("skb", SelectKBest(f_classif, k=k))]) for k in [1, 2, 3, 5]]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metricas = ["roc_auc", "average_precision", "f1", "recall", "precision"]
filas = []

for nombre, k, pasos in escenarios:
    pipe = Pipeline(
        [("scaler", StandardScaler())] + pasos +
        [("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    )
    r = cross_validate(pipe, X, y, cv=cv, scoring=metricas)
    filas.append({"Método": nombre, "Dims": k,
                  **{m: r[f"test_{m}"].mean() for m in metricas}})

tabla = pd.DataFrame(filas).rename(columns={
    "roc_auc": "ROC-AUC", "average_precision": "PR-AUC",
    "f1": "F1", "recall": "Recall", "precision": "Precision"})
print(tabla.round(3).to_string(index=False))
