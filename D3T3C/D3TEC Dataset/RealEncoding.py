# Importar bibliotecas necesarias
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import joblib

from .utils import (
    generate_random_architecture, encode_model_architecture,
    fixArch, generate_and_train_models, build_tf_model_from_dict,
    decode_model_architecture, predefined_architectures
)
import xgboost as xgb
# Generar y entrenar modelos predefinidos y aleatorios
print("Generando y entrenando modelos...")
trained_models_df = generate_and_train_models(
    predefined_architectures, num_random_models=270, target_shape=(128, 128, 1), use_kfold=True
)

# Cargar resultados del entrenamiento
print("Cargando resultados del entrenamiento...")
model_results = pd.read_csv(trained_models_df)

# Procesar la columna 'Encoded Architecture'
print("Procesando arquitecturas codificadas...")
model_results['Encoded Architecture'] = model_results['Encoded Architecture'].apply(eval)

# Preparar los datos para entrenamiento y prueba
X_features = pd.DataFrame(model_results['Encoded Architecture'].tolist())
y_metrics = model_results[['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']]
X_train, X_test, y_train, y_test = train_test_split(X_features, y_metrics, test_size=0.2, random_state=42)

# Definir espacio de búsqueda para hiperparámetros
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
param_dist_xgb = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Optimizar y evaluar modelos para cada métrica
print("Optimizando modelos para cada métrica...")
results_summary = []
for metric in y_metrics.columns:
    print(f"Optimizando para la métrica: {metric}")

    # Crear modelo base
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model, param_grid=param_grid_xgb,
        scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=10, n_jobs=-1
    )
    grid_search.fit(X_train, y_train[metric])
    best_grid_model = grid_search.best_estimator_
    best_grid_score = -grid_search.best_score_

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model, param_distributions=param_dist_xgb, n_iter=20,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=10, random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train[metric])
    best_random_model = random_search.best_estimator_
    best_random_score = -random_search.best_score_

    # Evaluar modelos optimizados
    for search_type, best_model, best_score in [
        ("GridSearchCV", best_grid_model, best_grid_score),
        ("RandomizedSearchCV", best_random_model, best_random_score)
    ]:
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test[metric], predictions)
        joblib.dump(best_model, f"{metric}_{search_type}_model.pkl")

        # Agregar resultados al resumen
        results_summary.append({
            'Metric': metric,
            'Search Type': search_type,
            'Best Score': best_score,
            'Test MSE': mse,
            'Best Params': best_model.get_params()
        })

# Guardar el resumen de los resultados
results_df = pd.DataFrame(results_summary)
print("\nResumen de resultados de optimización:")
print(results_df)

# Función para calcular SynFlow
def compute_synflow_scores(model, input_size):
    input_tensor = tf.ones(input_size)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor)
        gradients = tape.gradient(output, model.trainable_weights)
    return sum(tf.reduce_sum(tf.abs(w * g)).numpy() for w, g in zip(model.trainable_weights, gradients))

# Calcular SynFlow para cada arquitectura
print("Calculando puntajes SynFlow...")
model_results['SynFlow'] = [
    compute_synflow_scores(build_tf_model_from_dict(decode_model_architecture(arch), input_shape=(128, 128, 1)), (1, 128, 128, 1))
    for arch in model_results['Encoded Architecture']
]

# Guardar dataset actualizado con SynFlow
model_results.to_csv("model_results_with_synflow.csv", index=False)
print("Resultados con SynFlow guardados en 'model_results_with_synflow.csv'.")
