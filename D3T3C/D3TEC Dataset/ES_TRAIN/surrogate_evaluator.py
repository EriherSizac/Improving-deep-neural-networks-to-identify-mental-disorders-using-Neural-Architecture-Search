import numpy as np
import sys
import os
from ES_TRAIN.normalizer import normalize_surrogate_input

def evaluate_architecture(architecture, surrogate_model):
    """
    Evalúa una arquitectura utilizando un modelo surrogate, normalizando los datos previamente.
    
    Args:
        architecture (list): Representación codificada de la arquitectura.
        surrogate_model: Modelo surrogate que evaluará la arquitectura.
        
    Returns:
        float: Puntuación de fitness predicha por el modelo surrogate.
    """
    # Normalizar la arquitectura para el modelo surrogate
    normalized_input = normalize_surrogate_input(architecture, surrogate_model)
    
    # Asegurar que la entrada tenga la forma correcta para el modelo
    if len(normalized_input.shape) == 1:
        normalized_input = normalized_input.reshape(1, -1)
    
    # Realizar la predicción con el modelo surrogate
    try:
        prediction = surrogate_model.predict(normalized_input)
        
        # Si la predicción es un array, extraer el valor escalar
        if isinstance(prediction, np.ndarray) and prediction.size == 1:
            prediction = prediction.item()
        
        return prediction
    except Exception as e:
        print(f"Error al evaluar la arquitectura: {e}")
        return float('-inf')  # Devolver un valor muy bajo en caso de error

def batch_evaluate_architectures(architectures, surrogate_model):
    """
    Evalúa un lote de arquitecturas utilizando un modelo surrogate.
    
    Args:
        architectures (list): Lista de arquitecturas codificadas.
        surrogate_model: Modelo surrogate que evaluará las arquitecturas.
        
    Returns:
        list: Lista de puntuaciones de fitness predichas.
    """
    # Normalizar todas las arquitecturas
    normalized_inputs = np.array([normalize_surrogate_input(arch, surrogate_model) for arch in architectures])
    
    # Realizar predicciones en lote
    try:
        predictions = surrogate_model.predict(normalized_inputs)
        
        # Convertir a lista de valores escalares
        if isinstance(predictions, np.ndarray):
            predictions = predictions.flatten().tolist()
        
        return predictions
    except Exception as e:
        print(f"Error al evaluar el lote de arquitecturas: {e}")
        return [float('-inf')] * len(architectures)  # Devolver valores muy bajos en caso de error
