# -*- coding: utf-8 -*-
# ============================================================
# APP STREAMLIT - REDUCCIÓN DE DIMENSIONALIDAD PARA CLASIFICACIÓN
# PCA · LDA · SelectKBest (ANOVA F-value)
# Basado en la "Guía Teórica y Práctica de Reducción de Dimensionalidad"
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Reducción de Dimensionalidad", layout="wide")
st.title("📊 Reducción de Dimensionalidad para Clasificación")

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:

    # CAMBIO 1: sep=None detecta el separador automáticamente.
    # bank-additional-full.csv usa ";" y con read_csv normal se lee como 1 columna.
    df = pd.read_csv(uploaded_file, sep=None, engine="python")

    st.subheader("Vista previa de los datos originales")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()

    # CAMBIO 2: por defecto se propone la última columna (normalmente el target)
    target_col = st.sidebar.selectbox(
        "Selecciona la variable Objetivo (Target):",
        all_columns,
        index=len(all_columns) - 1
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # CAMBIO 3: LDA y SelectKBest necesitan un target de clases, no continuo
    if y.nunique() > 20:
        st.error(
            f"'{target_col}' tiene {y.nunique()} valores distintos. "
            "Elige una variable objetivo categórica (por ejemplo 'y')."
        )
        st.stop()

    X_numeric = X.select_dtypes(include=[np.number])

    # CAMBIO 4 (opcional): permite excluir variables, p. ej. 'duration',
    # que solo se conoce al terminar la llamada
    excluir = st.sidebar.multiselect(
        "Variables numéricas a excluir (opcional):",
        X_numeric.columns.tolist()
    )
    X_numeric = X_numeric.drop(columns=excluir)

    if X_numeric.empty:
        st.error("El dataset no contiene características numéricas.")
    else:
        st.write(
            f"Se usan **{X_numeric.shape[1]} variables numéricas**: "
            + ", ".join(X_numeric.columns)
        )

        if X_numeric.isnull().sum().sum() > 0:
            X_numeric = X_numeric.fillna(X_numeric.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        method = st.sidebar.selectbox(
            "Método de reducción:",
            [
                "PCA (Principal Component Analysis)",
                "LDA (Linear Discriminant Analysis)",
                "SelectKBest (ANOVA F-value)"
            ]
        )

        n_features_max = X_numeric.shape[1]
        max_components = (
            min(n_features_max, len(np.unique(y)) - 1)
            if "LDA" in method else n_features_max
        )

        # CAMBIO 5: st.slider falla si mínimo = máximo (LDA binario => 1 dimensión)
        if max_components > 1:
            n_components = st.sidebar.slider(
                "Número de dimensiones/características:",
                1, max_components, min(2, max_components)
            )
        else:
            n_components = 1
            st.sidebar.info(
                "LDA con 2 clases solo puede generar 1 dimensión (C - 1 = 1)."
            )

        df_transformed = pd.DataFrame()

        if "PCA" in method:
            pca = PCA(n_components=n_components)
            X_res = pca.fit_transform(X_scaled)
            df_transformed = pd.DataFrame(
                X_res, columns=[f"PC_{i+1}" for i in range(n_components)]
            )
            st.info(
                f"Varianza explicada retenida: "
                f"{np.sum(pca.explained_variance_ratio_)*100:.2f}%"
            )

            # Agregado: varianza de TODAS las componentes para elegir k
            pca_full = PCA().fit(X_scaled)
            var = pd.DataFrame({
                "Componente": [f"PC_{i+1}" for i in range(n_features_max)],
                "Autovalor": pca_full.explained_variance_,
                "Varianza (%)": pca_full.explained_variance_ratio_ * 100,
                "Acumulada (%)": np.cumsum(pca_full.explained_variance_ratio_) * 100
            })
            st.subheader("Varianza explicada por componente")
            st.dataframe(var.round(2), use_container_width=True)

            st.subheader("Cargas (loadings) de las componentes elegidas")
            st.dataframe(
                pd.DataFrame(
                    pca.components_.T,
                    index=X_numeric.columns,
                    columns=df_transformed.columns
                ).round(3),
                use_container_width=True
            )

        elif "LDA" in method:
            lda = LDA(n_components=n_components)
            X_res = lda.fit_transform(X_scaled, y)
            df_transformed = pd.DataFrame(
                X_res, columns=[f"LD_{i+1}" for i in range(n_components)]
            )

            # Agregado: peso de cada variable en el discriminante
            st.subheader("Coeficientes del discriminante (variables estandarizadas)")
            st.dataframe(
                pd.DataFrame(
                    lda.scalings_[:, :n_components],
                    index=X_numeric.columns,
                    columns=df_transformed.columns
                ).round(3),
                use_container_width=True
            )

        elif "SelectKBest" in method:
            selector = SelectKBest(score_func=f_classif, k=n_components)
            X_res = selector.fit_transform(X_numeric, y)
            selected_cols = X_numeric.columns[selector.get_support()].tolist()
            df_transformed = pd.DataFrame(X_res, columns=selected_cols)

            # Agregado: ranking ANOVA de todas las variables
            st.subheader("Ranking ANOVA F-value")
            st.dataframe(
                pd.DataFrame({
                    "Variable": X_numeric.columns,
                    "F-value": selector.scores_,
                    "p-value": selector.pvalues_,
                    "Seleccionada": selector.get_support()
                }).sort_values("F-value", ascending=False).round(4),
                use_container_width=True
            )

        df_final = df_transformed.copy()
        df_final[target_col] = y.values

        st.subheader("Resultado de la Reducción")
        st.dataframe(df_final.head())

        # Agregado: visualización de los datos reducidos por clase
        st.subheader("Visualización por clase")
        fig, ax = plt.subplots(figsize=(8, 5))
        cols = df_transformed.columns
        for clase in np.unique(y):
            m = (y.values == clase)
            if len(cols) >= 2:
                ax.scatter(df_transformed.loc[m, cols[0]],
                           df_transformed.loc[m, cols[1]],
                           s=5, alpha=0.4, label=str(clase))
            else:
                ax.hist(df_transformed.loc[m, cols[0]], bins=60,
                        alpha=0.5, density=True, label=str(clase))
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1] if len(cols) >= 2 else "Densidad")
        ax.legend(title=target_col)
        st.pyplot(fig)

        csv_data = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar Dataset Reducido (CSV)",
            csv_data,
            f"dataset_{method.split()[0].lower()}.csv",
            "text/csv"
        )
