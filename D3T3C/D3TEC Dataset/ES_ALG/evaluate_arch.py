import joblib
import numpy as np
import os
import copy

def evaluate_architecture(architecture_array, model_path, pca_path=None):
    """
    Evalúa una arquitectura específica usando un modelo surrogate y aplicando PCA si es necesario.
    
    Args:
        architecture_array: Lista o array con la codificación de la arquitectura
        model_path: Ruta al modelo surrogate
        pca_path: Ruta al transformador PCA (opcional)
    
    Returns:
        Puntuación predicha por el modelo surrogate
    """
    # Cargar el modelo surrogate
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    surrogate_model = joblib.load(model_path)
    
    # Hacer una copia profunda del array para no modificar el original
    arch_copy = copy.deepcopy(architecture_array)
    
    # Convertir a numpy array y reshape para la predicción
    arch_array = np.array(arch_copy).reshape(1, -1)
    
    # Aplicar transformación PCA si se proporciona la ruta
    if pca_path and os.path.exists(pca_path):
        pca_transformer = joblib.load(pca_path)
        arch_array = pca_transformer.transform(arch_array)
    
    # Realizar la predicción
    score = surrogate_model.predict(arch_array)
    
    return score[0]

if __name__ == "__main__":
    # Rutas a los archivos necesarios
    base_dir = "../surrogates_v5.2"
    model_path = os.path.join(base_dir, "xgboost_noscaler_model.pkl")
    pca_path = os.path.join(base_dir, "pca_transformer.pkl")
    
    # Verificar que los archivos existen
    if not os.path.exists(model_path):
        print(f"ADVERTENCIA: No se encontró el modelo en {model_path}")
    
    if not os.path.exists(pca_path):
        print(f"ADVERTENCIA: No se encontró el transformador PCA en {pca_path}")
    
    # Arquitecturas a evaluar
    """  [8, 0, 1, 0, 0, 4, 1, 0, 8, 2, 1, 0, 8, 3, 7, 0, 8, 2, 1, 0, 0, 32, 0, 0, 0, 4, 0, 0, 8, 1, 3, 0, 8, 1, 1, 0, 0, 32, 1, 0, 8, 7, 5, 0, 0, 4, 0, 0],
        [8, 0, 7, 0, 8, 1, 27, 0, 0, 32, 1, 0, 0, 4, 0, 0, 8, 4, 3, 0, 0, 32, 0, 0, 0, 4, 1, 0, 0, 4, 0, 0, 8, 1, 1, 0, 0, 32, 0, 0, 0, 4, 1, 0, 8, 1, 1, 0],
        [8, 0, 15, 0, 8, 1, 1, 0, 0, 4, 1, 0, 8, 3, 3, 0, 8, 4, 1, 0, 0, 32, 0, 0, 0, 4, 1, 0, 0, 4, 0, 0, 8, 1, 1, 0, 0, 32, 1, 0, 0, 4, 1, 0, 0, 4, 0, 0], """
    architectures_to_evaluate = [
      
        [0, 30, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0, 4, 1, 2, 0],
        [1, 0, 0, 0, 0, 16, 0, 1, 1, 0, 0, 0, 0, 8, 0, 1, 1, 0, 0, 0, 5, 0, 0, 0, 4, 32, 1, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0]

    ]
    #Arquitectura 1: 0.4836139976978302
    #Arquitectura 2: 0.4971272349357605
    
    print("Evaluación de arquitecturas:")
    for i, arch in enumerate(architectures_to_evaluate):
        try:
            score = evaluate_architecture(arch, model_path, pca_path)
            print(f"Arquitectura {i+1}: {score}")
        except Exception as e:
            print(f"Error al evaluar arquitectura {i+1}: {str(e)}")