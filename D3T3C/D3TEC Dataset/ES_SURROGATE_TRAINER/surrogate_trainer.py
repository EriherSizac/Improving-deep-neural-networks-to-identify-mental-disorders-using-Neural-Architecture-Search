"""
Surrogate Trainer Module

Este módulo se encarga de cargar datos de arquitecturas neuronales desde un CSV
y entrenar modelos surrogate para predecir su rendimiento (F1 Score).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV

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
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10)))

# Clase para escalar y desescalar el target (F1 Score)
class TargetScaler(BaseEstimator, TransformerMixin):
    """
    Escala y desescala el target (F1 Score) para mejorar el aprendizaje.
    Aplica una transformación exponencial para amplificar las diferencias.
    """
    def __init__(self, exponent=4, use_exp_transform=True):
        self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        self.exponent = exponent
        self.use_exp_transform = use_exp_transform
        self.min_val = None
        self.max_val = None
        
    def fit(self, y):
        # Guardar valores mínimo y máximo originales
        if isinstance(y, pd.DataFrame):
            self.min_val = y.values.min()
            self.max_val = y.values.max()
            
            # Aplicar transformación exponencial si está habilitada
            if self.use_exp_transform:
                # Normalizar a [0,1] antes de aplicar exponencial
                normalized = (y.values - self.min_val) / (self.max_val - self.min_val)
                # Asegurar que no haya valores negativos o cero para evitar problemas con la exponenciación
                normalized = np.clip(normalized, 1e-10, 1.0)
                transformed = np.power(normalized, self.exponent)
                # Escalar con MinMaxScaler
                self.scaler.fit(transformed.reshape(-1, 1))
            else:
                # Usar escalado normal
                self.scaler.fit(y.values.reshape(-1, 1))
        else:
            self.min_val = y.min()
            self.max_val = y.max()
            
            # Aplicar transformación exponencial si está habilitada
            if self.use_exp_transform:
                # Normalizar a [0,1] antes de aplicar exponencial
                normalized = (y - self.min_val) / (self.max_val - self.min_val)
                # Asegurar que no haya valores negativos o cero para evitar problemas con la exponenciación
                normalized = np.clip(normalized, 1e-10, 1.0)
                transformed = np.power(normalized, self.exponent)
                # Escalar con MinMaxScaler
                self.scaler.fit(transformed.reshape(-1, 1))
            else:
                # Usar escalado normal
                self.scaler.fit(y.reshape(-1, 1))
                
        return self
        
    def transform(self, y):
        if isinstance(y, pd.DataFrame):
            y_vals = y.values
        else:
            y_vals = y
            
        # Aplicar transformación exponencial si está habilitada
        if self.use_exp_transform:
            # Normalizar a [0,1] antes de aplicar exponencial
            normalized = (y_vals - self.min_val) / (self.max_val - self.min_val)
            # Asegurar que no haya valores negativos o cero para evitar problemas con la exponenciación
            normalized = np.clip(normalized, 1e-10, 1.0)
            transformed = np.power(normalized, self.exponent)
            # Escalar con MinMaxScaler
            return self.scaler.transform(transformed.reshape(-1, 1)).flatten()
        else:
            # Usar escalado normal
            return self.scaler.transform(y_vals.reshape(-1, 1)).flatten()
        
    def inverse_transform(self, y):
        # Desescalar con MinMaxScaler
        if self.use_exp_transform:
            # Primero desescalar con MinMaxScaler
            descaled = self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            # Asegurar que no haya valores negativos o cero para evitar problemas con la raíz
            descaled = np.clip(descaled, 1e-10, 1.0)
            # Luego aplicar la raíz para revertir la exponenciación
            normalized_back = np.power(descaled, 1/self.exponent)
            # Finalmente, volver al rango original
            return normalized_back * (self.max_val - self.min_val) + self.min_val
        else:
            # Desescalar normalmente
            return self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()

def load_and_prepare_data(file_path, test_size=0.2, random_state=42, add_polynomial=True):
    """
    Carga y prepara los datos para el entrenamiento de modelos surrogate.
    
    Args:
        file_path: Ruta al archivo CSV con arquitecturas codificadas
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        add_polynomial: Si se deben añadir características polinómicas
        
    Returns:
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler: Datos preparados y normalizadores
    """
    print(f"📊 Cargando datos desde {file_path}...")
    data = pd.read_csv(file_path)
    
    # Convertir 'Encoded Architecture' en listas de enteros
    data['Encoded Architecture'] = data['Encoded Architecture'].apply(eval)
    
    # Expandir 'Encoded Architecture' en múltiples columnas
    X = pd.DataFrame(data['Encoded Architecture'].tolist())
    
    # Convertir todos los nombres de columnas a strings para evitar problemas con StandardScaler
    X.columns = X.columns.astype(str)
    
    # Añadir características adicionales para capturar patrones no lineales
    if add_polynomial:
        print("🔄 Generando características polinómicas...")
        # Seleccionar un subconjunto de columnas para evitar explosión dimensional
        # Podemos seleccionar columnas que representen tipos de capa, por ejemplo
        layer_type_cols = [str(i) for i in range(0, X.shape[1], 4)]  # Cada 4 columnas (tipo de capa)
        X_layer_types = X[layer_type_cols]
        
        # Generar características polinómicas de grado 2
        poly = PolynomialFeatures(2, include_bias=False)
        X_poly = poly.fit_transform(X_layer_types)
        
        # Añadir estas características al DataFrame original
        X_poly_df = pd.DataFrame(
            X_poly, 
            columns=[f'poly_{i}' for i in range(X_poly.shape[1])]
        )
        
        # Concatenar con las características originales
        X = pd.concat([X, X_poly_df], axis=1)
        
        # Añadir características de interacción entre columnas adyacentes
        print("🔄 Generando características de interacción...")
        for i in range(0, X.shape[1] - 4, 4):
            col1 = X.columns[i]
            col2 = X.columns[i + 4]
            X[f'interact_{i}_{i+4}'] = X[col1] * X[col2]
        
        print(f"✅ Características aumentadas: {X.shape[1]} columnas totales")
    
    # Métrica objetivo (solo F1 Score)
    y = data[['F1']]
    
    # Análisis detallado de la distribución de F1 Score
    print("\n📊 Análisis detallado de F1 Score:")
    print(f"   Min: {y['F1'].min():.4f}, Max: {y['F1'].max():.4f}")
    print(f"   Media: {y['F1'].mean():.4f}, Mediana: {y['F1'].median():.4f}")
    print(f"   Desviación estándar: {y['F1'].std():.4f}")
    
    # Crear histograma para visualizar la distribución
    plt.figure(figsize=(10, 6))
    plt.hist(y['F1'], bins=30, alpha=0.7, color='blue')
    plt.axvline(y['F1'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Media: {y["F1"].mean():.4f}')
    plt.axvline(y['F1'].median(), color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {y["F1"].median():.4f}')
    plt.title('Distribución de F1 Score')
    plt.xlabel('F1 Score')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(os.path.dirname(file_path), 'f1_distribution.png'))
    plt.close()
    
    # Verificar si hay poca variabilidad en los datos
    if y['F1'].std() < 0.05:
        print("\n⚠️ ADVERTENCIA: Baja variabilidad en F1 Score")
        print("   Esto puede dificultar que los modelos capturen patrones significativos.")
    
    # Normalizar los datos de entrada
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # Escalar el target (F1 Score) con transformación exponencial
    target_scaler = TargetScaler(exponent=4, use_exp_transform=True)
    y_scaled = target_scaler.fit_transform(y)
    
    print(f"✅ Datos cargados: {len(y)} muestras con {X.shape[1]} características")
    print(f"   Rango de F1 Score original: [{y['F1'].min():.4f}, {y['F1'].max():.4f}]")
    print(f"   Rango de F1 Score escalado: [{np.min(y_scaled):.4f}, {np.max(y_scaled):.4f}]")
    
    # Visualizar la distribución del F1 Score original vs transformado
    plt.figure(figsize=(12, 5))
    
    # F1 Score original
    plt.subplot(1, 2, 1)
    plt.hist(y['F1'], bins=30, alpha=0.7, color='blue')
    plt.axvline(y['F1'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Media: {y["F1"].mean():.4f}')
    plt.axvline(y['F1'].median(), color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {y["F1"].median():.4f}')
    plt.title('Distribución de F1 Score Original')
    plt.xlabel('F1 Score')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score transformado
    plt.subplot(1, 2, 2)
    plt.hist(y_scaled, bins=30, alpha=0.7, color='purple')
    plt.axvline(np.mean(y_scaled), color='red', linestyle='dashed', linewidth=2, label=f'Media: {np.mean(y_scaled):.4f}')
    plt.axvline(np.median(y_scaled), color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {np.median(y_scaled):.4f}')
    plt.title('Distribución de F1 Score Transformado (Exponencial)')
    plt.xlabel('F1 Score Transformado')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), 'f1_distribution_comparison.png'))
    plt.close()
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def train_base_models(X_train, y_train, cv=5):
    """
    Entrena modelos base para usar en el ensemble.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Target de entrenamiento
        cv: Número de folds para validación cruzada
        
    Returns:
        base_models: Lista de modelos base entrenados
    """
    print("\n🔹 Entrenando modelos base para ensemble...")
    
    # Definir modelos base con hiperparámetros optimizados
    base_models = [
        ('xgb', xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )),
        ('svr', SVR(
            C=10,
            gamma='scale',
            epsilon=0.05,
            kernel='rbf'
        )),
        ('catboost', CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.01,
            l2_leaf_reg=3,
            border_count=128,
            bagging_temperature=1,
            random_state=42,
            verbose=0
        )),
        ('rf', RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )),
        ('gbm', GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )),
        ('knn', KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2
        )),
        ('elastic', ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=[0.0001, 0.001, 0.01, 0.1, 1.0],
            max_iter=2000,
            cv=5,
            random_state=42
        ))
    ]
    
    # Entrenar cada modelo base con validación cruzada
    trained_models = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in base_models:
        print(f"   Entrenando {name}...")
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Clonar el modelo para cada fold
            model_clone = clone(model)
            model_clone.fit(X_cv_train, y_cv_train)
            
            # Evaluar en el fold de validación
            y_cv_pred = model_clone.predict(X_cv_val)
            mse = mean_squared_error(y_cv_val, y_cv_pred)
            cv_scores.append(mse)
        
        # Entrenar el modelo final con todos los datos
        model.fit(X_train, y_train)
        trained_models.append((name, model))
        
        print(f"      MSE CV: {np.mean(cv_scores):.6f} (±{np.std(cv_scores):.6f})")
    
    return trained_models

def create_voting_ensemble(base_models):
    """
    Crea un modelo ensemble de tipo voting.
    
    Args:
        base_models: Lista de modelos base entrenados
        
    Returns:
        voting_ensemble: Modelo ensemble de tipo voting
    """
    print("\n🔹 Creando ensemble de tipo voting...")
    
    # Crear el ensemble con pesos iguales
    voting_ensemble = VotingRegressor(
        estimators=base_models,
        weights=[1] * len(base_models)
    )
    
    return voting_ensemble

def create_stacking_ensemble(base_models, X_train, y_train):
    """
    Crea un modelo ensemble de tipo stacking.
    
    Args:
        base_models: Lista de modelos base entrenados
        X_train: Datos de entrenamiento
        y_train: Target de entrenamiento
        
    Returns:
        stacking_ensemble: Modelo ensemble de tipo stacking
    """
    print("\n🔹 Creando ensemble de tipo stacking...")
    
    # Definir el meta-modelo
    meta_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    # Crear el ensemble de stacking
    stacking_ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    # Entrenar el ensemble
    stacking_ensemble.fit(X_train, y_train)
    
    return stacking_ensemble

def create_neural_ensemble(base_models, X_train, y_train, X_test, y_test, output_dir):
    """
    Crea un modelo ensemble basado en redes neuronales con transfer learning.
    
    Args:
        base_models: Lista de modelos base entrenados
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        output_dir: Directorio para guardar el modelo
        
    Returns:
        neural_ensemble: Modelo ensemble basado en redes neuronales
    """
    print("\n🔹 Creando ensemble neuronal con transfer learning...")
    
    # Obtener predicciones de los modelos base para el conjunto de entrenamiento
    base_train_predictions = np.column_stack([
        model.predict(X_train) for _, model in base_models
    ])
    
    # Obtener predicciones de los modelos base para el conjunto de prueba
    base_test_predictions = np.column_stack([
        model.predict(X_test) for _, model in base_models
    ])
    
    # Configurar semilla para reproducibilidad
    tf.random.set_seed(42)
    
    # Definir la arquitectura del modelo de ensemble
    # Entrada 1: Características originales
    input_features = Input(shape=(X_train.shape[1],), name='input_features')
    
    # Entrada 2: Predicciones de los modelos base
    input_predictions = Input(shape=(len(base_models),), name='input_predictions')
    
    # Rama 1: Procesar características originales
    x1 = Dense(256, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(input_features)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Dense(128, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    # Rama 2: Procesar predicciones de los modelos base
    x2 = Dense(32, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(input_predictions)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    
    # Combinar ambas ramas
    combined = Concatenate()([x1, x2])
    
    # Capas finales
    x = Dense(64, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(combined)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Capa de salida
    output = Dense(1, activation='linear')(x)
    
    # Crear el modelo
    model = Model(inputs=[input_features, input_predictions], outputs=output)
    
    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss='mse',
        metrics=['mae']
    )
    
    # Definir callbacks
    checkpoint_path = os.path.join(output_dir, "neural_ensemble.h5")
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Guardar mejor modelo
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Reducción de tasa de aprendizaje en meseta
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Entrenar el modelo
    history = model.fit(
        [X_train, base_train_predictions],
        y_train,
        validation_data=([X_test, base_test_predictions], y_test),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar gráfico de pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Curvas de Pérdida del Ensemble Neuronal')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(output_dir, "neural_ensemble_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    print(f"✅ Ensemble neuronal guardado en {checkpoint_path}")
    print(f"   Gráfico de pérdida guardado en {loss_plot_path}")
    
    return model, [X_test, base_test_predictions]

def evaluate_models(models, X_test, y_test, target_scaler, output_dir):
    """
    Evalúa los modelos entrenados y genera visualizaciones.
    
    Args:
        models: Diccionario con los modelos entrenados
        X_test: Datos de prueba
        y_test: Target de prueba
        target_scaler: Escalador del target para desescalar las predicciones
        output_dir: Directorio para guardar las visualizaciones
    """
    print("\n🔹 Evaluando modelos...")
    
    # Diccionarios para almacenar resultados
    predictions = {}
    metrics = {}
    
    # Evaluar cada modelo
    for name, model_info in models.items():
        if name == 'Neural Ensemble':
            model, test_inputs = model_info
            y_pred_scaled = model.predict(test_inputs).flatten()
        else:
            model = model_info
            y_pred_scaled = model.predict(X_test)
        
        # Desescalar las predicciones
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Guardar resultados
        predictions[name] = y_pred
        metrics[name] = (mse, mae, mape, r2)
        
        # Imprimir resultados
        print(f"\n✅ Evaluación de {name}:")
        print(f"   MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"   MAPE: {mape:.2f}%, R²: {r2:.6f}")
        
        # Verificar si el modelo está prediciendo valores constantes
        pred_std = np.std(y_pred)
        if pred_std < 0.001:
            print(f"⚠️ ADVERTENCIA: El modelo {name} está prediciendo valores casi constantes")
            print(f"   Desviación estándar de predicciones: {pred_std:.6f}")
        
        # Generar gráfico de predicciones vs valores reales
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Línea ideal
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--', label="Ideal"
        )
        
        # Añadir línea de tendencia
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_true, p(y_true), 'g-', alpha=0.7, label="Tendencia")
        
        # Configurar gráfico
        plt.xlabel("F1 Score Real")
        plt.ylabel("F1 Score Predicho")
        plt.title(f"{name}\nMSE: {mse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%, R²: {r2:.6f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plot_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_predictions.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"   Gráfico guardado en {plot_path}")
    
    # Comparar todos los modelos
    plt.figure(figsize=(12, 8))
    
    # Crear un gráfico de barras para MSE
    model_names = list(metrics.keys())
    mse_values = [metrics[name][0] for name in model_names]
    r2_values = [metrics[name][3] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # MSE en el eje izquierdo
    bars1 = ax1.bar(x - width/2, mse_values, width, label='MSE', color='skyblue')
    ax1.set_ylabel('MSE', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    # R² en el eje derecho
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='salmon')
    ax2.set_ylabel('R²', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    
    # Configuración general
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_title('Comparación de Modelos: MSE y R²')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"\n✅ Gráfico de comparación guardado en {comparison_path}")
    
    # Devolver el mejor modelo según MSE
    best_model_name = min(metrics, key=lambda k: metrics[k][0])
    print(f"\n🏆 Mejor modelo: {best_model_name}")
    print(f"   MSE: {metrics[best_model_name][0]:.6f}")
    print(f"   R²: {metrics[best_model_name][3]:.6f}")
    
    return predictions, metrics, best_model_name

def main(file_path, output_dir):
    """
    Función principal para entrenar y evaluar modelos surrogate.
    
    Args:
        file_path: Ruta al archivo CSV con datos
        output_dir: Directorio para guardar resultados
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar y preparar datos
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler = load_and_prepare_data(
        file_path, 
        test_size=0.2, 
        random_state=42,
        add_polynomial=True
    )
    
    # Guardar los scalers para uso futuro
    joblib.dump(feature_scaler, os.path.join(output_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(output_dir, "target_scaler.pkl"))
    print(f"✅ Scalers guardados en {output_dir}")
    
    # Entrenar modelos base
    base_models = train_base_models(X_train, y_train, cv=5)
    
    # Crear ensemble de tipo voting
    voting_ensemble = create_voting_ensemble(base_models)
    
    # Entrenar ensemble de tipo voting
    voting_ensemble.fit(X_train, y_train)
    
    # Crear ensemble de tipo stacking
    stacking_ensemble = create_stacking_ensemble(base_models, X_train, y_train)
    
    # Crear ensemble neuronal con transfer learning
    neural_ensemble, test_inputs = create_neural_ensemble(
        base_models, X_train, y_train, X_test, y_test, output_dir
    )
    
    # Guardar modelos
    for name, model in base_models:
        joblib.dump(model, os.path.join(output_dir, f"{name}_model.pkl"))
    
    joblib.dump(voting_ensemble, os.path.join(output_dir, "voting_ensemble.pkl"))
    joblib.dump(stacking_ensemble, os.path.join(output_dir, "stacking_ensemble.pkl"))
    
    # Evaluar todos los modelos
    models = {
        **{name: model for name, model in base_models},
        'Voting Ensemble': voting_ensemble,
        'Stacking Ensemble': stacking_ensemble,
        'Neural Ensemble': (neural_ensemble, test_inputs)
    }
    
    predictions, metrics, best_model_name = evaluate_models(
        models, X_test, y_test, target_scaler, output_dir
    )
    
    print("\n✅ Entrenamiento completado. Todos los modelos y resultados guardados en:", output_dir)
    print(f"   Mejor modelo: {best_model_name}")

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar modelos surrogate para NAS')
    parser.add_argument('--file', type=str, required=True, help='Ruta al archivo CSV con datos')
    parser.add_argument('--output', type=str, default='./surrogates', help='Directorio para guardar resultados')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar función principal
    main(args.file, args.output)
