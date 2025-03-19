"""
Surrogate Trainer Module

Este módulo se encarga de cargar datos de arquitecturas neuronales desde un CSV
y entrenar modelos surrogate para predecir su rendimiento (F1 Score).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Función para calcular MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calcula el error porcentual absoluto medio (MAPE).
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        MAPE como porcentaje
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100

def load_and_prepare_data(file_path, test_size=0.2, random_state=42):
    """
    Carga y prepara los datos para el entrenamiento de modelos surrogate.
    
    Args:
        file_path: Ruta al archivo CSV con arquitecturas codificadas
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        
    Returns:
        X_train, X_test, y_train, y_test, scaler: Datos preparados y normalizador
    """
    print(f"📊 Cargando datos desde {file_path}...")
    data = pd.read_csv(file_path)
    
    # Convertir 'Encoded Architecture' en listas de enteros
    data['Encoded Architecture'] = data['Encoded Architecture'].apply(eval)
    
    # Expandir 'Encoded Architecture' en múltiples columnas
    X = pd.DataFrame(data['Encoded Architecture'].tolist())
    
    # Normalizar los datos de entrada
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Métrica objetivo (solo F1 Score)
    y = data[['F1']]
    print(f"✅ Datos cargados: {len(y)} muestras con {X.shape[1]} características")
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_sklearn_models(X_train, X_test, y_train, y_test, output_dir):
    """
    Entrena y evalúa modelos tradicionales de scikit-learn.
    
    Args:
        X_train, X_test: Datos de entrenamiento y prueba
        y_train, y_test: Etiquetas de entrenamiento y prueba
        output_dir: Directorio para guardar los modelos
        
    Returns:
        models, predictions, metrics: Modelos entrenados, predicciones y métricas
    """
    # Espacios de búsqueda de hiperparámetros
    param_grids = {
        'XGBoost': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        },
        'CatBoost': {
            'iterations': [50, 100],
            'depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        },
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    }

    # Modelos a entrenar
    models = {
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'SVM': SVR(),
        'CatBoost': CatBoostRegressor(silent=True),
        'RandomForest': RandomForestRegressor()
    }

    # Diccionarios para almacenar resultados
    trained_models = {}
    predictions = {}
    metrics = {}

    # Entrenar cada modelo
    for name, model in models.items():
        print(f"\n🔹 Entrenando {name} para F1 Score...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train['F1'])
        best_model = grid_search.best_estimator_
        
        # Predicciones y métricas
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test['F1'], y_pred)
        mae = mean_absolute_error(y_test['F1'], y_pred)
        mape = mean_absolute_percentage_error(y_test['F1'], y_pred)
        
        # Guardar resultados
        trained_models[name] = best_model
        predictions[name] = y_pred
        metrics[name] = (mse, mae, mape)
        
        # Guardar modelo en formato .pkl
        model_path = os.path.join(output_dir, f"F1_{name}_optimized.pkl")
        joblib.dump(best_model, model_path)
        print(f"✅ Modelo {name} guardado en {model_path}")
        print(f"   MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        print(f"   Mejores parámetros: {grid_search.best_params_}")
    
    return trained_models, predictions, metrics

def train_deep_neural_network(X_train, X_test, y_train, y_test, output_dir):
    """
    Entrena y evalúa un modelo de red neuronal profunda.
    
    Args:
        X_train, X_test: Datos de entrenamiento y prueba
        y_train, y_test: Etiquetas de entrenamiento y prueba
        output_dir: Directorio para guardar el modelo
        
    Returns:
        model, predictions, metrics: Modelo entrenado, predicciones y métricas
    """
    # Configurar semilla para reproducibilidad
    tf.random.set_seed(42)
    
    # Obtener dimensiones de entrada
    input_dim = X_train.shape[1]
    
    # Definir la arquitectura del modelo
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    # Definir callbacks
    checkpoint_path = os.path.join(output_dir, "DeepNN_model.h5")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=64,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    
    # Evaluar el modelo
    loss, mae_val = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test['F1'], y_pred)
    mae = mean_absolute_error(y_test['F1'], y_pred)
    mape = mean_absolute_percentage_error(y_test['F1'], y_pred)
    
    print(f"✅ Modelo DeepNN guardado en {checkpoint_path}")
    print(f"   MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    
    return model, y_pred, (mse, mae, mape)

def plot_model_predictions(predictions, y_test, metrics):
    """
    Genera gráficos individuales por modelo comparando predicciones vs valores reales.
    
    Args:
        predictions: Diccionario con predicciones de cada modelo
        y_test: Valores reales de prueba
        metrics: Diccionario con métricas (MSE, MAE, MAPE) de cada modelo
    """
    for model_name, y_pred in predictions.items():
        mse_val, mae_val, mape_val = metrics[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, label=model_name)
        ax.plot(
            [min(y_test.values), max(y_test.values)],
            [min(y_test.values), max(y_test.values)],
            'r--', label="Ideal"
        )
        ax.set_xlabel("F1 Score Real")
        ax.set_ylabel("F1 Score Predicho")
        ax.set_title(f"{model_name}\nMSE: {mse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")
        ax.legend()
        plt.tight_layout()
        plt.show()

def main(file_path="./EncodedChromosomes_V3_results.csv", output_dir="./surrogates"):
    """
    Función principal para entrenar y evaluar modelos surrogate.
    
    Args:
        file_path: Ruta al archivo CSV con arquitecturas codificadas
        output_dir: Directorio para guardar los modelos entrenados
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar y preparar datos
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(file_path)
    
    # Guardar el scaler para uso futuro
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler guardado en {scaler_path}")
    
    # Entrenar modelos tradicionales de scikit-learn
    print("\n🔹 Entrenando modelos tradicionales...")
    sklearn_models, sklearn_predictions, sklearn_metrics = train_sklearn_models(
        X_train, X_test, y_train, y_test, output_dir
    )
    
    # Entrenar modelo de red neuronal profunda
    print("\n🔹 Entrenando modelo de red neuronal profunda...")
    deep_model, deep_predictions, deep_metrics = train_deep_neural_network(
        X_train, X_test, y_train, y_test, output_dir
    )
    
    # Combinar resultados
    all_predictions = sklearn_predictions.copy()
    all_predictions["DeepNN"] = deep_predictions
    
    all_metrics = sklearn_metrics.copy()
    all_metrics["DeepNN"] = deep_metrics
    
    # Visualizar predicciones
    print("\n🔹 Generando visualizaciones...")
    plot_model_predictions(all_predictions, y_test, all_metrics)
    
    print("\n✅ Entrenamiento de modelos surrogate completado.")
    print(f"✅ Todos los modelos y resultados guardados en {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos surrogate para NAS")
    parser.add_argument("--file", type=str, default="./EncodedChromosomes_V3_results.csv",
                        help="Ruta al archivo CSV con arquitecturas codificadas")
    parser.add_argument("--output", type=str, default="./surrogates",
                        help="Directorio para guardar los modelos entrenados")
    
    args = parser.parse_args()
    main(args.file, args.output)
