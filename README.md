# forecast-logistica

Desafio tecnico Data Scientist Lead (Acid Labs / Koandina): auditoria de forecast, modelo de correccion, gate de historia minima y arquitectura AWS.

## Estructura

```text
forecast-logistica/
├── src/logforecast/                      # paquete Python: toda la logica
│   ├── config.py                         # Settings desde variables de entorno (LF_*)
│   ├── io.py                             # lectura/escritura local o s3:// (mismo codigo)
│   ├── quality.py                        # chequeos de calidad de datos
│   ├── metrics.py                        # WAPE, bias y tablas de error
│   ├── diagnostics.py                    # Tarea 1: error por CD y horizonte
│   ├── plan_analysis.py                  # Tarea 2: bias del plan, fuera de plan, adhesion
│   ├── features.py                       # Tarea 3: features causales (calendario, rolling, picos)
│   ├── evaluation.py                     # Tarea 3: motor de backtest (baselines y modelos)
│   ├── cv.py                             # Tarea 3: CV temporal, decision y modelo elegido
│   ├── model.py                          # ElasticNetCorrector, RatioCorrector, GatedCorrector
│   ├── gate.py                           # Tarea 4: gate de 45 dias, nunca lanza excepciones
│   ├── gate_study.py                     # Tarea 4: estudio del umbral (cold start, curvas)
│   ├── capacity.py                       # capacidad logistica y alerta del endpoint
│   ├── plots.py                          # graficos que llaman los notebooks
│   ├── pipeline.py                       # train() / predict(): lo que usan CLI y Lambda
│   ├── cli.py                            # python -m logforecast.cli {quality,train}  -> ECS
│   └── handlers/lambda_handler.py        # inferencia                                -> Lambda
├── notebooks/                            # solo llaman funciones de src/logforecast
│   ├── 00_desafio.ipynb                  # resumen: Tareas 1 a 5 e impacto en negocio
│   ├── 01_tarea1_wape_bias.ipynb         # Tarea 1: WAPE y bias por CD y horizonte
│   ├── 02_tarea2_plan_entrega.ipynb      # Tarea 2: plan de entrega y compras fuera de plan
│   ├── 03_exploracion_t3.ipynb           # Tarea 3: exploracion y baselines
│   ├── 04_validacion_cruzada_t3.ipynb    # Tarea 3: CV de ridge, ElasticNet y LightGBM, test
│   ├── 05_tarea4_gate_t4.ipynb           # Tarea 4: gate de 45 dias y umbral
│   ├── 06_tarea5_arquitectura_aws.ipynb  # Tarea 5: arquitectura AWS
│   └── 07_diagnostico_capacidad.ipynb    # pronostico vs capacidad logistica
├── docs/diagramas/                       # diagramas de arquitectura: .mmd, .svg y .png
├── data/raw/                             # CSV del desafio
├── tests/                                # pytest
├── artifacts/                            # modelo entrenado (no versionado)
├── Dockerfile, docker-compose.yml        # entorno JupyterLab y tests
└── pyproject.toml, requirements*.txt     # dependencias
```

### Donde esta cada tarea

| Tarea | Notebook | Codigo |
|---|---|---|
| Resumen del desafio | `00_desafio` | - |
| 1. Diagnostico WAPE y bias | `01_tarea1_wape_bias` | `diagnostics.py`, `metrics.py`, `quality.py` |
| 2. Plan de entrega | `02_tarea2_plan_entrega` | `plan_analysis.py` |
| 3. Modelo complementario | `03_exploracion_t3`, `04_validacion_cruzada_t3` | `features.py`, `evaluation.py`, `cv.py`, `model.py` |
| 4. Gate de 45 dias | `05_tarea4_gate_t4` | `gate.py`, `gate_study.py`, `model.py` |
| 5. Arquitectura AWS | `06_tarea5_arquitectura_aws`, `docs/diagramas/` | `pipeline.py`, `cli.py`, `handlers/` |
| Capacidad e impacto | `07_diagnostico_capacidad` | `capacity.py` |

## Uso local (Docker)

```bash
docker compose up --build            # JupyterLab en http://localhost:8888
docker compose run --rm tests        # pytest
docker compose run --rm lab python -m logforecast.cli quality
docker compose run --rm lab python -m logforecast.cli train
```
## Configuracion

Toda la configuracion se define por variables de entorno (`src/logforecast/config.py`).
`.env.example` documenta cada una con su valor por defecto.

| Variable | Default | Docker | AWS |
|---|---|---|---|
| `LF_DATA_URI` | `data/raw` | `/app/data/raw` | `s3://<bucket>/raw` |
| `LF_ARTIFACTS_URI` | `artifacts` | `/app/artifacts` | `s3://<bucket>/artifacts` |
| `LF_MIN_HISTORY_DAYS` | `45` | idem | idem |
| `LF_BIAS_TOLERANCE` | `0.05` | idem | idem |
| `LF_CAPACITY_WARN_RATIO` | `0.9` | idem | idem |

Los defaults de rutas son relativos a la raiz del repo. En Docker los fija
`docker-compose.yml`; en ECS/Lambda se definen en la task definition o en
la configuracion de la funcion. El modelo (`artifacts/model.joblib`) no se
versiona: se genera con `python -m logforecast.cli train`.

## Modelo en produccion
`pipeline.train()` entrena un `GatedCorrector` con el `ElasticNetCorrector` elegido por CV en
`04_validacion_cruzada_t3` (`cv.ELEGIDO`); los CD con menos de `LF_MIN_HISTORY_DAYS` (45) dias de
historia continua reciben el pronostico base. El artefacto (`model.joblib`) incluye la historia reciente
que necesitan las features de inferencia, por lo que el entrenamiento debe correr a diario.

## Pendiente
- Ampliar las grillas de ElasticNet y LightGBM (los mejores quedan en el borde).
- Publicacion condicionada al backtest, puntero de version en DynamoDB y monitoreo: solo diseno (notebook 06).
- Validar el umbral del gate con CD que entren en operacion en momentos distintos.

