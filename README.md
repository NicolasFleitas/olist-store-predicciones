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
├── requirements.txt      # Dependencias fijadas (freeze verificado)
└── README.md
```

## Instalación

Requiere Python 3.12+ (probado en 3.14.6).

Con `uv` (recomendado — te instala ese Python sin tocar el del sistema):

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Con `pip` (instalá Python 3.12 o superior desde [python.org](https://www.python.org/downloads/) primero):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> En Windows usá `.venv\Scripts\activate` en lugar de `source .venv/bin/activate`.

## Ejecución

1. Activar el entorno virtual:

   ```bash
   source .venv/bin/activate
   ```

2. Abrir el notebook:

   ```bash
   jupyter notebook olist_predict.ipynb
   ```

Ejecutar las celdas en orden. Para agregar una dependencia nueva: `pip install <paquete>` o `uv pip install <paquete>`, y luego `uv pip freeze --python .venv/bin/python > requirements.txt`.
