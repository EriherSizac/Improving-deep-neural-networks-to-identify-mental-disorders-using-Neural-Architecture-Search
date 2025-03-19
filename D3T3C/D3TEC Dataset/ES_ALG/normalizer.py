"""
Módulo para normalización de datos en el proceso de búsqueda de arquitecturas neurales.
Proporciona funciones para normalizar individuos antes de pasarlos al modelo surrogate.
"""

import numpy as np
import os
import pickle
import warnings

# Ruta al scaler preentrenado
SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'ES_SURROGATE_TRAINER', 'surrogates', 'feature_scaler.pkl'
)

# Cargar el scaler si existe
_scaler = None
def get_scaler():
    """
    Carga y devuelve el scaler preentrenado.
    Si no existe, devuelve None y se usará la normalización manual.
    """
    global _scaler
    if _scaler is None:
        try:
            if os.path.exists(SCALER_PATH):
                # Intentar diferentes métodos para cargar el scaler
                try:
                    # Método 1: pickle estándar
                    with open(SCALER_PATH, 'rb') as f:
                        _scaler = pickle.load(f)
                    print(f"Scaler cargado desde: {SCALER_PATH}")
                except Exception as e1:
                    try:
                        # Método 2: pickle con encoding latin1
                        with open(SCALER_PATH, 'rb') as f:
                            _scaler = pickle.load(f, encoding='latin1')
                        print(f"Scaler cargado con encoding latin1 desde: {SCALER_PATH}")
                    except Exception as e2:
                        try:
                            # Método 3: joblib
                            import joblib
                            _scaler = joblib.load(SCALER_PATH)
                            print(f"Scaler cargado con joblib desde: {SCALER_PATH}")
                        except Exception as e3:
                            warnings.warn(f"No se pudo cargar el scaler después de varios intentos. Se usará normalización manual.")
            else:
                warnings.warn(f"No se encontró el scaler en {SCALER_PATH}. Se usará normalización manual.")
        except Exception as e:
            warnings.warn(f"Error al cargar el scaler: {str(e)}. Se usará normalización manual.")
    return _scaler

def normalize_individual(individual):
    """
    Normaliza un individuo (arquitectura) para que todos sus valores estén en el rango [0, 1].
    Intenta usar el scaler preentrenado si está disponible, de lo contrario usa normalización manual.
    
    Args:
        individual: Array o lista que representa una arquitectura neural
        
    Returns:
        Array normalizado con valores en el rango [0, 1]
    """
    # Convertir a array de numpy si no lo es
    if not isinstance(individual, np.ndarray):
        individual = np.array(individual)
    
    # Intentar usar el scaler preentrenado
    scaler = get_scaler()
    if scaler is not None:
        # Reshape para que tenga la forma correcta para el scaler
        reshaped_ind = individual.reshape(1, -1)
        try:
            # Aplicar el scaler
            normalized = scaler.transform(reshaped_ind)
            return normalized.flatten()
        except Exception as e:
            warnings.warn(f"Error al aplicar el scaler: {str(e)}. Se usará normalización manual.")
    
    # Si no hay scaler o falló, usar normalización manual
    # Crear copia para no modificar el original
    normalized = individual.copy().astype(float)
    
    # Normalizar cada valor según su tipo y rango esperado
    for i in range(len(normalized)):
        # Cada posición en el array tiene un significado específico y un rango
        pos = i % 3  # Posición dentro del triplete (tipo, param1, param2)
        
        if pos == 0:  # Tipo de capa (0-8)
            normalized[i] = normalized[i] / 9.0
        elif pos == 1:  # Primer parámetro (varía según el tipo)
            # Normalizar según el tipo de capa (posición anterior)
            layer_type = int(individual[i-1])
            if layer_type == 0:  # Conv2D
                # filters: [4, 32]
                normalized[i] = (normalized[i] - 4) / 28.0 if normalized[i] >= 4 else 0
            elif layer_type == 1:  # SelfAttention
                # filters: [4, 64]
                normalized[i] = (normalized[i] - 4) / 60.0 if normalized[i] >= 4 else 0
            elif layer_type == 2:  # BatchNorm
                normalized[i] = 0  # No tiene parámetros relevantes
            elif layer_type == 3:  # MaxPooling
                # strides: 1 o 2
                normalized[i] = 0 if normalized[i] <= 1 else 1
            elif layer_type == 4:  # Dropout
                # rate: [0.2, 0.5]
                normalized[i] = (normalized[i] - 0.2) / 0.3 if 0.2 <= normalized[i] <= 0.5 else (0 if normalized[i] < 0.2 else 1)
            elif layer_type == 5:  # Dense
                # units: [1, 512]
                normalized[i] = (normalized[i] - 1) / 511.0 if normalized[i] >= 1 else 0
            elif layer_type == 6:  # Flatten
                normalized[i] = 0  # No tiene parámetros relevantes
            elif layer_type == 7:  # DontCare
                normalized[i] = 0  # No tiene parámetros relevantes
            elif layer_type == 8:  # Repetition
                # repetition_layers: [1, 4]
                normalized[i] = (normalized[i] - 1) / 3.0 if normalized[i] >= 1 else 0
        elif pos == 2:  # Segundo parámetro (varía según el tipo)
            # Normalizar según el tipo de capa (posición -2)
            layer_type = int(individual[i-2])
            if layer_type == 0:  # Conv2D
                # strides: 1 o 2
                normalized[i] = 0 if normalized[i] <= 1 else 1
            elif layer_type == 1:  # SelfAttention
                # attention_heads: [1, 8]
                normalized[i] = (normalized[i] - 1) / 7.0 if normalized[i] >= 1 else 0
            elif layer_type == 8:  # Repetition
                # repetition_count: [1, 3]
                normalized[i] = (normalized[i] - 1) / 2.0 if normalized[i] >= 1 else 0
            else:
                normalized[i] = 0  # Otros tipos no usan este parámetro
    
    return normalized

def batch_normalize_individuals(individuals):
    """
    Normaliza un lote de individuos para pasarlos al modelo surrogate.
    Intenta usar el scaler preentrenado si está disponible, de lo contrario usa normalización manual.
    
    Args:
        individuals: Lista o array de individuos a normalizar
        
    Returns:
        Array con todos los individuos normalizados
    """
    if not isinstance(individuals, np.ndarray):
        individuals = np.array(individuals)
    
    # Si es un solo individuo, asegurarse de que tenga la forma correcta
    if individuals.ndim == 1:
        individuals = individuals.reshape(1, -1)
    
    # Intentar usar el scaler preentrenado
    scaler = get_scaler()
    if scaler is not None:
        try:
            # Aplicar el scaler a todo el batch
            return scaler.transform(individuals)
        except Exception as e:
            warnings.warn(f"Error al aplicar el scaler al batch: {str(e)}. Se usará normalización manual.")
    
    # Si no hay scaler o falló, usar normalización manual
    normalized_batch = np.zeros_like(individuals, dtype=float)
    for i in range(len(individuals)):
        normalized_batch[i] = normalize_individual(individuals[i])
    
    return normalized_batch
