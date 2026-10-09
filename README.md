# Reducción de Dimensionalidad para Clasificación (PCA · LDA · SelectKBest)

Aplicación interactiva en **Streamlit** que reduce la dimensión de un dataset de clasificación con tres métodos y permite descargar el resultado. Se aplica al dataset **Bank Marketing**, cuyo objetivo es predecir si un cliente contrata un depósito a plazo.

| Método | Tipo | Estrategia | Máx. dimensiones | Interpretabilidad |
| --- | --- | --- | --- | --- |
| PCA | No supervisado | Extracción (combinación lineal) | Nº de variables | Baja |
| LDA | Supervisado | Extracción (combinación lineal) | Nº de clases − 1 | Baja |
| SelectKBest (ANOVA F) | Supervisado | Selección (conserva variables) | Nº de variables | Alta |

## Resultados principales

Dataset: 41 188 clientes, 10 variables numéricas estandarizadas. Variable objetivo `y` (no 88.7 %, yes 11.3 %).

**PCA.** Con el criterio de Kaiser (autovalor > 1) se conservan 4 componentes, que explican el 73.8 % de la varianza. Para pasar el 80 % se necesitan 5 componentes. PC1 (39 %) agrupa las variables macroeconómicas: euribor3m, emp.var.rate y nr.employed.

![PCA](img/pca_varianza.png)

**LDA.** Como la clasificación es binaria, LDA genera una sola dimensión (C − 1 = 1). Ese eje separa bien las clases: el promedio de LD_1 es −0.25 para "no" y 1.94 para "yes".

![LDA](img/lda_ld1.png)

**SelectKBest.** Con k = 3 se eligen duration, nr.employed y pdays.

![ANOVA](img/anova_f.png)

### Comparación con una Regresión Logística

Validación cruzada estratificada de 5 folds, con `class_weight="balanced"`. La reducción se ajusta dentro del pipeline de cada fold.

| Método | Dims | ROC-AUC | PR-AUC | F1 | Recall | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| Original | 10 | **0.926** | 0.567 | 0.562 | 0.849 | 0.420 |
| PCA | 1 | 0.766 | 0.368 | 0.364 | 0.712 | 0.244 |
| PCA | 2 | 0.775 | 0.386 | 0.360 | 0.714 | 0.241 |
| PCA | 3 | 0.776 | 0.388 | 0.365 | 0.690 | 0.248 |
| PCA | 5 | 0.913 | 0.538 | 0.525 | 0.838 | 0.382 |
| LDA | 1 | 0.922 | **0.581** | **0.568** | 0.823 | **0.434** |
| SelectKBest | 1 | 0.818 | 0.396 | 0.409 | 0.622 | 0.305 |
| SelectKBest | 2 | 0.917 | 0.540 | 0.540 | 0.838 | 0.398 |
| SelectKBest | 3 | 0.922 | 0.566 | 0.548 | 0.833 | 0.408 |
| SelectKBest | 5 | 0.925 | 0.564 | 0.564 | **0.854** | 0.421 |

**Conclusiones**

- **LDA** logra con 1 dimensión casi el mismo ROC-AUC que las 10 variables (0.922 frente a 0.926) y el mejor F1.
- **SelectKBest** iguala ese resultado con 3 variables y mantiene sus nombres, por lo que es la opción más fácil de explicar.
- **PCA** necesita 5 componentes, porque al no usar `y` prioriza la varianza macroeconómica y no la variable que más separa las clases (duration).
- `duration` solo se conoce cuando termina la llamada. Si se excluye (`python comparacion_metodos.py --sin-duration`), todos los métodos quedan entre 0.74 y 0.77 de ROC-AUC.

## Estructura del repositorio

```
├── app.py                     # App Streamlit (PCA, LDA, SelectKBest)
├── comparacion_metodos.py     # Script que genera la tabla de comparación
├── data/
│   └── bank-additional-full.csv
├── img/                       # Gráficos del README
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

```bash
git clone https://github.com/TU_USUARIO/reduccion-dimensionalidad-streamlit.git
cd reduccion-dimensionalidad-streamlit
pip install -r requirements.txt
streamlit run app.py
```

En la barra lateral, sube `data/bank-additional-full.csv`, elige el método y el número de dimensiones. Puedes descargar el dataset reducido en CSV.

Para reproducir la tabla de comparación:

```bash
python comparacion_metodos.py
```

## Cambios respecto al código de la guía

- Lee CSV con cualquier separador (`sep=None`), porque el archivo del banco usa `;`.
- LDA binario: el slider de Streamlit falla cuando mínimo = máximo. Ahora la dimensión queda fija en 1 con un aviso.
- La última columna aparece como target por defecto, y la app avisa si se elige una variable continua.
- Se agregaron tablas de varianza y cargas (PCA), coeficientes (LDA), el ranking ANOVA (SelectKBest), un gráfico por clase y la opción de excluir variables.

## Dataset

Bank Marketing (UCI Machine Learning Repository), licencia CC BY 4.0.
Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing*. Decision Support Systems, 62, 22–31.

## Autor

[Tu nombre] — Curso de Modelamiento Predictivo / Analítica de Datos
