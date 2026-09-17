# Predicción de Tiempos de Entrega en E-Commerce (Olist Marketplace)

Pipeline de Machine Learning para predecir los días de entrega (`target_days`) de pedidos del marketplace **Olist** (Brasil), con **Scikit-Learn** (regresión tabular, métrica principal MAE).

## Resultados

Evaluación en test temporal (30% final):

| Modelo | MAE (días) | RMSE (días) | $R^2$ |
| :--- | :---: | :---: | :---: |
| Dummy (mean, baseline) | 6.18 | 7.24 | -0.3606 |
| Decision Tree | 3.94 | 5.42 | 0.2368 |
| Random Forest | 3.78 | **5.21** | **0.2939** |
| **Gradient Boosting** (`HistGradientBoostingRegressor`) | **3.75** | 5.22 | 0.2921 |

Baseline ingenuo: `DummyRegressor(strategy="mean")` — predice la media del train. Todos los modelos lo superan ampliamente (~2.4 días menos de MAE).

Modelo ganador: **Gradient Boosting** (menor MAE: ~3 días y 18 hs de desvío promedio).

## Estructura

```
├── olist_predict.ipynb   # Notebook con el pipeline completo
├── datasets/             # CSVs de Olist
├── pyproject.toml        # Dependencias del proyecto (PEP 621)
├── uv.lock               # Lockfile generado con uv
├── .python-version       # Python 3.12
├── requirements.txt      # Alternativa para pip (autogenerado con uv export)
└── README.md
```

## Clonar e instalar

Requisitos: `uv` + Python `>=3.12` (se recomienda 3.12 vía `.python-version`).

```bash
git clone https://github.com/NicolasFleitas/olist-store-predicciones.git
cd olist-store-predicciones
```

Con `uv` (recomendado):

```bash
uv sync
uv run jupyter lab
```

Con `pip` (alternativo):

```bash
pip install -r requirements.txt
jupyter lab
```

## Notebooks sin outputs (nbstripout)

Este repo usa `nbstripout` para no versionar outputs ni metadata de ejecución (`*.ipynb filter=nbstripout` + `diff=ipynb` en `.gitattributes`).

Después de clonar, activá el filtro una vez:

```bash
uv sync
uv run nbstripout --install
```

A partir de ahí `git diff` y `git commit` guardan el notebook limpio automáticamente.

## Replicar

1. `uv run jupyter lab`
2. Abrir `olist_predict.ipynb` → *Run All*.
