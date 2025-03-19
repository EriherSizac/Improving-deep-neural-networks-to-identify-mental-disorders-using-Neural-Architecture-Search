"""
Models Module

Este módulo contiene las implementaciones de diferentes modelos surrogate:
- Modelos tradicionales de scikit-learn (XGBoost, SVM, CatBoost, RandomForest)
- Modelo de red neuronal profunda con TensorFlow/Keras
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
