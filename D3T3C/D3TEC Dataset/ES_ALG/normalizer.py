"""
Módulo para normalización de datos en el proceso de búsqueda de arquitecturas neurales.
Proporciona funciones para normalizar individuos antes de pasarlos al modelo surrogate.
"""

import numpy as np
import warnings
from pathlib import Path
import joblib

# Nombre del archivo del scaler preentrenado\SCALER_FILENAME = "surrogates_v5.2/scaler.joblib"

# Caché del objeto scaler cargado\_scaler = None

def get_scaler():
    """
    Carga y devuelve el scaler preentrenado usando joblib.
    Lanza excepción si no se encuentra o no es válido.
    """
    global _scaler
    if _scaler is None:
        project_root = Path(__file__).resolve().parent.parent
        matches = list(project_root.rglob(SCALER_FILENAME))
        if not matches:
            raise RuntimeError(f"No se encontró '{SCALER_FILENAME}' bajo {project_root}")
        scaler_path = matches[0]
        try:
            _scaler = joblib.load(scaler_path)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el scaler desde {scaler_path}: {e}")
        if not hasattr(_scaler, 'transform'):
            raise RuntimeError(f"El objeto cargado desde {scaler_path} no tiene el método 'transform'.")
    return _scaler

def normalize_individual(individual):
    """
    Normaliza un individuo (arquitectura) para que todos sus valores estén en el rango [0, 1].
    Usa el scaler preentrenado o, si falla, una normalización manual.
    """
    if not isinstance(individual, np.ndarray):
        individual = np.array(individual)
    try:
        scaler = get_scaler()
        reshaped = individual.reshape(1, -1)
        return scaler.transform(reshaped).flatten()
    except Exception:
        warnings.warn("Fallo al aplicar scaler, usando normalización manual.")
    normalized = individual.astype(float).copy()
    for i in range(len(normalized)):
        pos = i % 3
        if pos == 0:
            normalized[i] = normalized[i] / 9.0
        elif pos == 1:
            layer_type = int(individual[i-1])
            if layer_type == 0:
                normalized[i] = (normalized[i] - 4) / 28.0 if normalized[i] >= 4 else 0
            elif layer_type == 1:
                normalized[i] = (normalized[i] - 4) / 60.0 if normalized[i] >= 4 else 0
            elif layer_type in (2, 6, 7):
                normalized[i] = 0
            elif layer_type == 3:
                normalized[i] = 0 if normalized[i] <= 1 else 1
            elif layer_type == 4:
                normalized[i] = (normalized[i] - 0.2) / 0.3 if 0.2 <= normalized[i] <= 0.5 else (0 if normalized[i] < 0.2 else 1)
            elif layer_type == 5:
                normalized[i] = (normalized[i] - 1) / 511.0 if normalized[i] >= 1 else 0
            elif layer_type == 8:
                normalized[i] = (normalized[i] - 1) / 3.0 if normalized[i] >= 1 else 0
        else:
            layer_type = int(individual[i - (2 if pos == 2 else 1)])
            if pos == 2:
                if layer_type == 0:
                    normalized[i] = 0 if normalized[i] <= 1 else 1
                elif layer_type == 1:
                    normalized[i] = (normalized[i] - 1) / 7.0 if normalized[i] >= 1 else 0
                elif layer_type == 8:
                    normalized[i] = (normalized[i] - 1) / 2.0 if normalized[i] >= 1 else 0
            else:
                normalized[i] = normalized[i]
    return normalized

def batch_normalize_individuals(individuals):
    """
    Normaliza un lote de individuos para pasarlos al modelo surrogate.
    Usa el scaler preentrenado o, si falla, la normalización manual.
    """
    array = np.array(individuals)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    try:
        scaler = get_scaler()
        return scaler.transform(array)
    except Exception:
        warnings.warn("Fallo al aplicar scaler al batch, usando normalización manual.")
    return np.vstack([normalize_individual(ind) for ind in array])
