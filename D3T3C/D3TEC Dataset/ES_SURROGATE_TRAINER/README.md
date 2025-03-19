# Módulo de Entrenamiento de Modelos Surrogate

Este módulo implementa un sistema para entrenar modelos surrogate que predicen el rendimiento (F1 Score) de arquitecturas neuronales sin necesidad de entrenarlas completamente. Forma parte del proyecto de búsqueda de arquitectura neuronal (NAS) para identificar trastornos mentales.

## Estructura del Módulo

El módulo está organizado en varios archivos:

- `surrogate_trainer.py`: Script principal que implementa la carga de datos, entrenamiento de modelos y visualización de resultados.
- `architecture.py`: Contiene funciones para codificar, decodificar y manipular arquitecturas de redes neuronales.
- `synflow.py`: Implementa la métrica SynFlow para evaluar arquitecturas sin entrenamiento.
- `visualization.py`: Funciones para generar gráficos y visualizaciones.
- `models.py`: Implementaciones de diferentes modelos surrogate.

## Características Principales

- Carga y prepara datos de arquitecturas codificadas desde un archivo CSV.
- Entrena múltiples modelos surrogate:
  - XGBoost
  - SVM
  - CatBoost
  - Random Forest
  - Red Neuronal Profunda (DeepNN)
- Evalúa y compara el rendimiento de los modelos con métricas como MSE, MAE y MAPE.
- Genera visualizaciones para comparar predicciones vs valores reales.
- Guarda los modelos entrenados para su uso posterior en algoritmos evolutivos.

## Uso

Para entrenar los modelos surrogate:

```bash
python surrogate_trainer.py --file ./EncodedChromosomes_V3_results.csv --output ./surrogates
```

Argumentos:
- `--file`: Ruta al archivo CSV con arquitecturas codificadas y sus métricas de rendimiento.
- `--output`: Directorio donde se guardarán los modelos entrenados.

## Flujo de Trabajo

1. El script carga los datos de arquitecturas codificadas desde un CSV.
2. Prepara los datos normalizando las características y dividiendo en conjuntos de entrenamiento y prueba.
3. Entrena modelos tradicionales de scikit-learn con búsqueda de hiperparámetros.
4. Entrena un modelo de red neuronal profunda con TensorFlow/Keras.
5. Evalúa todos los modelos y genera visualizaciones comparativas.
6. Guarda los modelos entrenados y el normalizador para uso posterior.

## Integración con el Sistema NAS

Los modelos surrogate entrenados con este módulo se utilizan posteriormente en el algoritmo de Evolución Diferencial (DE) para evaluar rápidamente nuevas arquitecturas sin necesidad de entrenarlas completamente, acelerando significativamente el proceso de búsqueda de arquitectura.

## Requisitos

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- CatBoost
- TensorFlow 2.x
- Joblib
