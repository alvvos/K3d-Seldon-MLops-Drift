# Datos del modelo de riesgo crediticio

Este directorio almacena el dataset de entrenamiento y la muestra de referencia utilizada para monitorizar **Data Drift** en producción.

## Archivos

* `credit_risk.csv`: Dataset tabular generado de forma determinista por `model/src/data.py` (5.000 registros por defecto, semilla `42`), con correlaciones realistas entre variables financieras, frontera de decisión no lineal y ~2.5% de valores nulos controlados para ejercitar los imputadores del pipeline.
* `reference_data.csv`: Muestra limpia de 1.000 observaciones del conjunto de entrenamiento (`X_train`) generada por `model/src/train.py`. Es la distribución de referencia $X_{\text{ref}}$ que consume `drift/drift_detector.py` (test de Kolmogorov-Smirnov con corrección de Bonferroni).

## Esquema de features

| Feature | Tipo | Rango típico | Unidad | Transformación en Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `age` | numérica | 21 – 75 | años | `SimpleImputer(median)` + `StandardScaler` |
| `annual_income` | numérica | 18.000 – 320.000 | USD | `SimpleImputer(median)` + `RobustScaler` |
| `employment_years` | numérica | 0 – 45 | años | `SimpleImputer(median)` + `StandardScaler` |
| `loan_amount` | numérica | 1.500 – 100.000 | USD | `SimpleImputer(median)` + `RobustScaler` |
| `interest_rate` | numérica | 4.0 – 26.0 | % | `SimpleImputer(median)` + `StandardScaler` |
| `debt_to_income` | numérica | 0.03 – 0.78 | ratio | `SimpleImputer(median)` + `StandardScaler` |
| `credit_score` | numérica | 300 – 850 | score | `SimpleImputer(median)` + `StandardScaler` |
| `num_open_accounts` | numérica | 1 – 24 | cuentas | `SimpleImputer(median)` + `StandardScaler` |
| `default` *(target)* | binaria | `{0, 1}` | clase | `0 = approved`, `1 = default` |

## Regeneración

```bash
make train
# o forzando regeneración del CSV:
python model/src/train.py --force-regenerate
```
