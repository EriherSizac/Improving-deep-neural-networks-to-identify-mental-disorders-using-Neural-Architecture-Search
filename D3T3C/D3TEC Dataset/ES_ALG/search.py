import numpy as np
import tensorflow as tf
import copy
import os
import json
from tqdm import tqdm
from .utils.encoding import decode_model_architecture, convert_individual, fixArch, encode_model_architecture, layer_type_options
from .utils.latin_hypercube import generate_latin_hypercube_samples
import datetime
import glob
import re
import matplotlib.pyplot as plt
from .normalizer import normalize_individual, batch_normalize_individuals
import inspect

     # Prepare checkpoint data - convert all NumPy arrays to Python native types
def numpy_to_python(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [numpy_to_python(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(x) for x in obj]
    else:
        return obj  

def pop_gen(num_models, max_alleles=48):
    """
    Genera una población inicial utilizando el hipercubo latino y las funciones existentes.

    Args:
        num_models: int - Número de individuos a generar.
        max_alleles: int - Número máximo de alelos en los cromosomas.

    Returns:
        list - Lista de diccionarios con individuos y su fitness inicializado a 0.
    """
    # Padres iniciales predefinidos (arquitecturas base)
    initial_parents = [
        [0, 30, 0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0],
        [1, 0, 0, 0, 0, 16, 0, 1, 1, 0, 0, 0, 0, 8, 0, 1, 1, 0, 0, 0, 5, 0, 0, 0, 4, 32, 1, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0],
        [0, 32, 0, 1, 1, 0, 0, 0, 2, 1, 0, 0, 8, 3, 31, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 3, 0, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0]
    ]
    
    # Añadir arquitecturas base a la población
    population = [{'individual': fixArch(parent)} for parent in initial_parents]
    
    # Número de modelos aleatorios a generar
    num_random = num_models - len(initial_parents)
    
    if num_random <= 0:
        return population
    
    # Dimensiones para el hipercubo latino (12 capas x 3 parámetros por capa)
    dimensions = 12 * 3
    
    # Generar muestras del hipercubo latino
    latin_samples = generate_latin_hypercube_samples(num_random, dimensions)
    
    # Conjunto para almacenar arquitecturas únicas (como tuplas para poder usar set)
    unique_architectures = set(tuple(convert_individual(p['individual'], to_real=False)) for p in population)
    
    # Contador de intentos para evitar bucles infinitos
    max_attempts = 100
    
    for sample in latin_samples:
        # Transformar cada muestra en una arquitectura
        model_samples = np.array(sample).reshape(12, 3)
        model_dict = {
            "layers": [map_to_architecture_params(layer_sample) for layer_sample in model_samples]
        }
        
        # Codificar el modelo y repararlo
        encoded_chromosome = encode_model_architecture(model_dict, max_alleles=max_alleles)
        repaired_architecture = fixArch(encoded_chromosome)
        
        # Verificar si esta arquitectura ya existe en la población
        arch_tuple = tuple(convert_individual(repaired_architecture, to_real=False))
        
        attempts = 0
        while arch_tuple in unique_architectures and attempts < max_attempts:
            # Si ya existe, aplicar una pequeña perturbación
            perturbed_architecture = perturb_architecture(repaired_architecture)
            repaired_architecture = fixArch(perturbed_architecture)
            arch_tuple = tuple(convert_individual(repaired_architecture, to_real=False))
            attempts += 1
        
        if attempts < max_attempts:  # Solo añadir si encontramos una arquitectura única
            unique_architectures.add(arch_tuple)
            population.append({"individual": repaired_architecture})
    
    # Si no tenemos suficientes modelos, generar más hasta alcanzar num_models
    while len(population) < num_models:
        # Generar una nueva muestra aleatoria
        random_sample = np.random.random(dimensions)
        model_samples = random_sample.reshape(12, 3)
        model_dict = {
            "layers": [map_to_architecture_params(layer_sample) for layer_sample in model_samples]
        }
        
        encoded_chromosome = encode_model_architecture(model_dict, max_alleles=max_alleles)
        repaired_architecture = fixArch(encoded_chromosome)
        
        # Verificar si esta arquitectura ya existe
        arch_tuple = tuple(convert_individual(repaired_architecture, to_real=False))
        if arch_tuple not in unique_architectures:
            unique_architectures.add(arch_tuple)
            population.append({"individual": repaired_architecture})
    
    return population

def perturb_architecture(architecture, perturbation_rate=0.2):
    """
    Aplica una pequeña perturbación a una arquitectura para generar variación.
    
    Args:
        architecture: list - Arquitectura codificada a perturbar.
        perturbation_rate: float - Tasa de perturbación (probabilidad de modificar cada alelo).
        
    Returns:
        list - Arquitectura perturbada.
    """
    perturbed = architecture.copy()
    
    for i in range(0, len(perturbed), 4):
        # No perturbamos el tipo de capa para mantener la estructura general
        # Solo perturbamos los parámetros de la capa
        
        layer_type = perturbed[i]
        
        # Perturbación para Conv2D
        if layer_type == 0 and i+3 < len(perturbed):
            if np.random.random() < perturbation_rate:
                # Perturbar filtros (entre 4 y 32)
                perturbed[i+1] = max(4, min(perturbed[i+1] + np.random.randint(-4, 5), 32))
            if np.random.random() < perturbation_rate:
                # Perturbar stride (0 o 1)
                perturbed[i+2] = 1 if perturbed[i+2] == 0 else 0
            if np.random.random() < perturbation_rate:
                # Perturbar activación (0-3)
                perturbed[i+3] = np.random.randint(0, 4)
                
        # Perturbación para SelfAttention
        elif layer_type == 6 and i+3 < len(perturbed):
            if np.random.random() < perturbation_rate:
                # Perturbar filtros (entre 4 y 64)
                perturbed[i+1] = max(4, min(perturbed[i+1] + np.random.randint(-8, 9), 64))
            if np.random.random() < perturbation_rate:
                # Perturbar attention_heads (entre 1 y 8)
                perturbed[i+2] = max(1, min(perturbed[i+2] + np.random.randint(-2, 3), 8))
            if np.random.random() < perturbation_rate:
                # Perturbar activación (0-3)
                perturbed[i+3] = np.random.randint(0, 4)
                
        # Perturbación para MaxPooling
        elif layer_type == 2 and i+1 < len(perturbed):
            if np.random.random() < perturbation_rate:
                # Perturbar stride (0 o 1)
                perturbed[i+1] = 1 if perturbed[i+1] == 0 else 0
                
        # Perturbación para Dropout
        elif layer_type == 3 and i+1 < len(perturbed):
            if np.random.random() < perturbation_rate:
                # Perturbar rate (0-3)
                perturbed[i+1] = np.random.randint(0, 4)
                
        # Perturbación para Dense
        elif layer_type == 4 and i+2 < len(perturbed):
            if np.random.random() < perturbation_rate:
                # Perturbar unidades (entre 1 y 512)
                perturbed[i+1] = max(1, min(perturbed[i+1] + np.random.randint(-32, 33), 512))
            if np.random.random() < perturbation_rate:
                # Perturbar activación (0-3)
                perturbed[i+2] = np.random.randint(0, 4)
                
        # Perturbación para Repetition
        elif layer_type == 8 and i+2 < len(perturbed):
            if np.random.random() < perturbation_rate:
                # Perturbar capas a repetir (entre 1 y 4)
                perturbed[i+1] = max(1, min(perturbed[i+1] + np.random.randint(-1, 2), 4))
            if np.random.random() < perturbation_rate:
                # Perturbar número de repeticiones (entre 1 y 32)
                perturbed[i+2] = max(1, min(perturbed[i+2] + np.random.randint(-2, 3), 32))
    
    return perturbed

def map_to_architecture_params(latin_hypercube_sample):
    """
    Mapea una muestra del hipercubo latino a parámetros de arquitectura.
    
    Args:
        latin_hypercube_sample: array - Muestra de 3 valores del hipercubo latino.
        
    Returns:
        dict - Diccionario con parámetros de capa.
    """
    layer_type = int(latin_hypercube_sample[0] * 9)  # 9 tipos de capas
    layer_mapping = ['Conv2D', 'SelfAttention', 'BatchNorm', 'MaxPooling',
                     'Dropout', 'Dense', 'Flatten', 'DontCare', 'Repetition']

    layer_type_name = layer_mapping[min(layer_type, len(layer_mapping)-1)]

    if layer_type_name == 'Conv2D':
        return {
            "type": "Conv2D",
            "filters": int(latin_hypercube_sample[1] * (32 - 4) + 4),  # [4, 32]
            "strides": 1 if latin_hypercube_sample[2] < 0.5 else 2,
            "activation": "relu"
        }
    elif layer_type_name == 'SelfAttention':  # Reemplazo de DepthwiseConv2D
        return {
            "type": "SelfAttention",
            "filters": int(latin_hypercube_sample[1] * (64 - 4) + 4),  # [4, 64]
            "attention_heads": int(latin_hypercube_sample[2] * (8 - 1) + 1),  # [1, 8]
            "activation": "relu"
        }
    elif layer_type_name == 'BatchNorm':
        return {"type": "BatchNorm"}
    elif layer_type_name == 'MaxPooling':
        return {"type": "MaxPooling", "strides": 1 if latin_hypercube_sample[1] < 0.5 else 2}
    elif layer_type_name == 'Dropout':
        return {"type": "Dropout", "rate": latin_hypercube_sample[1] * (0.5 - 0.2) + 0.2}
    elif layer_type_name == 'Dense':
        return {
            "type": "Dense",
            "units": int(latin_hypercube_sample[1] * (512 - 1) + 1),
            "activation": "relu"
        }
    elif layer_type_name == 'Flatten':
        return {"type": "Flatten"}
    elif layer_type_name == 'DontCare':
        return {"type": "DontCare"}
    elif layer_type_name == 'Repetition':
        return {
            "type": "Repetition",
            "repetition_layers": int(latin_hypercube_sample[1] * 3 + 1),
            "repetition_count": int(latin_hypercube_sample[2] * 2 + 1)
        }
    return {"type": "DontCare"}

def load_surrogate_model(model_path):
    """
    Carga el modelo surrogate para evaluación de arquitecturas.
    
    Args:
        model_path: Ruta al archivo del modelo (.h5 para modelos Keras, .pkl para modelos scikit-learn)
        
    Returns:
        Modelo cargado
        
    Raises:
        ValueError: Si la extensión del archivo no es soportada
    """
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == '.h5':
        # Cargar modelo Keras
        import tensorflow as tf
        return tf.keras.models.load_model(model_path, compile=False)
    elif ext == '.pkl':
        # Cargar modelo scikit-learn (como Random Forest)
        import pickle
        import joblib
        
        print(f"Intentando cargar modelo scikit-learn desde {model_path}")
        
        # Verificar que el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo {model_path} no existe")
        
        # Verificar el tamaño del archivo
        file_size = os.path.getsize(model_path)
        print(f"Tamaño del archivo: {file_size} bytes")
        
        # Intentar cargar con pickle
        try:
            print("Intentando cargar con pickle...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Modelo cargado exitosamente con pickle")
            return model
        except Exception as e:
            print(f"Error al cargar con pickle: {str(e)}")
            
            # Intentar cargar con joblib como alternativa
            try:
                print("Intentando cargar con joblib...")
                model = joblib.load(model_path)
                print("Modelo cargado exitosamente con joblib")
                return model
            except Exception as e2:
                print(f"Error al cargar con joblib: {str(e2)}")
                
                # Si ambos métodos fallan, intentar con una versión específica de pickle
                try:
                    print("Intentando cargar con pickle protocolo 4...")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, fix_imports=True, encoding='latin1')
                    print("Modelo cargado exitosamente con pickle protocolo 4")
                    return model
                except Exception as e3:
                    print(f"Error al cargar con pickle protocolo 4: {str(e3)}")
                    
                    # Si todo falla, proporcionar un mensaje de error detallado
                    error_msg = f"""
                    Error al cargar el modelo desde {model_path}
                    Error con pickle: {str(e)}
                    Error con joblib: {str(e2)}
                    Error con pickle protocolo 4: {str(e3)}

                    Por favor, asegúrese de que:
                    1. El modelo fue guardado correctamente
                    2. El modelo fue guardado con una versión compatible de scikit-learn
                    3. El archivo no está corrupto
"""
                    raise RuntimeError(error_msg)
    else:
        raise ValueError(f"Extensión de modelo no soportada: {ext}. Use .h5 para modelos Keras o .pkl para modelos scikit-learn.")

def evaluate_architecture(ind, surrogate_model):
    """
    Evalúa una arquitectura utilizando el modelo surrogate.
    
    Args:
        ind: Individuo a evaluar
        surrogate_model: Modelo surrogate para la evaluación
        
    Returns:
        Fitness predicho por el modelo surrogate
    """
    ind_copy = copy.deepcopy(ind)
    normalized_ind = normalize_individual(ind_copy)
    reshaped_ind = np.array(normalized_ind).reshape(1, -1)
    
    # Verificar si el modelo es de Keras o scikit-learn
    if hasattr(surrogate_model, 'predict') and 'verbose' in inspect.signature(surrogate_model.predict).parameters:
        # Es un modelo Keras que acepta verbose
        return surrogate_model.predict(reshaped_ind, verbose=0)[0]
    else:
        # Es un modelo scikit-learn que no acepta verbose
        return surrogate_model.predict(reshaped_ind)[0]

def evaluate_population(population, surrogate_model):
    """
    Evalúa una población completa utilizando el modelo surrogate.
    
    Args:
        population: Lista de individuos a evaluar
        surrogate_model: Modelo surrogate para la evaluación
        
    Returns:
        Lista de valores de fitness para cada individuo
    """
    # Normalizar toda la población
    normalized_population = batch_normalize_individuals(population)
    
    # Verificar si el modelo es de Keras o scikit-learn
    if hasattr(surrogate_model, 'predict') and 'verbose' in inspect.signature(surrogate_model.predict).parameters:
        # Es un modelo Keras que acepta verbose
        fitness_values = surrogate_model.predict(normalized_population, verbose=0)
    else:
        # Es un modelo scikit-learn que no acepta verbose
        fitness_values = surrogate_model.predict(normalized_population)
    
    return fitness_values

def crossover(parent1, parent2, cr_rate=0.5):
    """Perform crossover between two parent architectures."""
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    mask = np.random.random(len(parent1)) < cr_rate
    child = np.where(mask, parent1, parent2)
    return child

def find_latest_checkpoint(checkpoint_dir='./checkpoints'):
    """
    Encuentra el checkpoint más reciente en el directorio especificado.
    
    Args:
        checkpoint_dir: Directorio donde buscar checkpoints
        
    Returns:
        Ruta al checkpoint más reciente, o None si no se encuentra ninguno
    """
    try:
        # Verificar que el directorio exista
        if not os.path.exists(checkpoint_dir):
            print(f"El directorio {checkpoint_dir} no existe.")
            return None
        
        # Buscar directorios de saves
        save_dirs = glob.glob(os.path.join(checkpoint_dir, "saves_*"))
        if not save_dirs:
            print(f"No se encontraron directorios de saves en {checkpoint_dir}.")
            return None
        
        # Ordenar por fecha de modificación (más reciente primero)
        save_dirs.sort(key=os.path.getmtime, reverse=True)
        latest_save_dir = save_dirs[0]
        
        # Buscar archivos de checkpoint en el directorio más reciente
        checkpoint_files = glob.glob(os.path.join(latest_save_dir, "checkpoint_*.json"))
        if not checkpoint_files:
            print(f"No se encontraron archivos de checkpoint en {latest_save_dir}.")
            return None
        
        # Extraer números de checkpoint y ordenar
        checkpoint_numbers = []
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)_', f)
            if match:
                checkpoint_numbers.append((int(match.group(1)), f))
        
        if not checkpoint_numbers:
            print("No se pudieron extraer números de checkpoint.")
            return None
        
        # Ordenar por número de checkpoint (mayor primero)
        checkpoint_numbers.sort(reverse=True)
        latest_checkpoint = checkpoint_numbers[0][1]
        
        print(f"Checkpoint más reciente encontrado: {latest_checkpoint}")
        return latest_checkpoint
    
    except Exception as e:
        print(f"Error al buscar el checkpoint más reciente: {e}")
        return None

def get_succ_m(trial_fitness, parent_fitness):
    """
    Cuenta cuántas mutaciones fueron exitosas comparadas con el fitness del padre.
    
    Args:
        trial_fitness: Lista de fitness de los individuos mutados.
        parent_fitness: Lista de fitness de los padres.
        
    Returns:
        int: Número de mutaciones exitosas.
    """
    succ_m_count = sum(1 for i in range(len(trial_fitness)) if trial_fitness[i] > parent_fitness[i])
    return succ_m_count

def tournament_selection(population, fitness, tournament_size=3):
    """
    Realiza selección por torneo para elegir un individuo de la población.
    
    Args:
        population: Lista de individuos en la población
        fitness: Array de valores de fitness para cada individuo
        tournament_size: Tamaño del torneo (número de individuos que compiten)
        
    Returns:
        Índice del individuo seleccionado
    """
    # Asegurar que tournament_size no sea mayor que el tamaño de la población
    tournament_size = min(tournament_size, len(population))
    
    # Seleccionar aleatoriamente individuos para el torneo
    tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    
    # Obtener fitness de los individuos seleccionados usando el array de fitness
    tournament_fitness = [fitness[i] for i in tournament_indices]
    
    # Seleccionar el mejor individuo del torneo (mayor fitness)
    best_tournament_idx = np.argmax(tournament_fitness)
    winner_idx = tournament_indices[best_tournament_idx]
    
    return winner_idx

def are_individuals_different(ind1, ind2, threshold=0.1):
    """
    Compara dos individuos para determinar si son significativamente diferentes.
    
    Args:
        ind1: Primer individuo
        ind2: Segundo individuo
        threshold: Umbral de diferencia (porcentaje de genes diferentes requerido)
        
    Returns:
        True si los individuos son diferentes, False en caso contrario
    """
    if len(ind1) != len(ind2):
        return True
    
    # Contar cuántos genes son diferentes
    different_genes = sum(1 for g1, g2 in zip(ind1, ind2) if g1 != g2)
    
    # Calcular el porcentaje de diferencia
    difference_percentage = different_genes / len(ind1)
    
    return difference_percentage >= threshold

def sus_selection(fitness, num_selections):
    """
    Realiza Stochastic Universal Sampling (SUS) sobre el vector de fitness.
    
    Args:
        fitness (np.ndarray): Array de fitness de la población.
        num_selections (int): Número de individuos a seleccionar.
        
    Returns:
        List[int]: Índices de los individuos seleccionados.
    """
    total_fitness = np.sum(fitness)
    pointer_distance = total_fitness / num_selections
    start_point = np.random.uniform(0, pointer_distance)
    pointers = [start_point + i * pointer_distance for i in range(num_selections)]
    cum_sum = np.cumsum(fitness)
    selected_indices = []
    i = 0
    for pointer in pointers:
        # Avanzar en la suma acumulativa hasta que el puntero sea menor o igual
        while i < len(cum_sum) and pointer > cum_sum[i]:
            i += 1
        selected_indices.append(i if i < len(fitness) else len(fitness) - 1)
    return selected_indices

def unified_search(surrogate_model, population_size=10, generations=100, n_experiments=1, 
                  F=0.5, cr_rate=0.5, auto_adaptation=True, checkpoint_dir='./checkpoints',
                  resume_from=None, new_run=False, selection_method='tournament'):
    """
    Función unificada para búsqueda de arquitecturas neurales usando Evolución Diferencial.
    
    Args:
        surrogate_model: Modelo surrogate para evaluar arquitecturas
        population_size: Tamaño de la población
        generations: Número de generaciones
        n_experiments: Número de experimentos a realizar
        F: Factor de mutación
        cr_rate: Tasa de cruce
        auto_adaptation: Si se debe adaptar automáticamente el factor F
        checkpoint_dir: Directorio para guardar checkpoints
        resume_from: Ruta a un checkpoint para reanudar la búsqueda
        new_run: Si se debe iniciar una nueva búsqueda, ignorando checkpoints existentes
        selection_method: Método de selección ('random', 'tournament', o 'sus')
        
    Returns:
        Diccionario con resultados de la búsqueda
    """
    # Asegurar que el directorio de checkpoints exista
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Si no es una nueva ejecución, buscar el último checkpoint
    start_experiment = 0
    start_generation = 0
    population = None
    fitness = None
    best_model_exp = None
    best_model_overall = None
    best_fitness_overall = float('-inf')  # Problema de maximización, inicializar con -inf
    fitness_history_exp = []
    f_history_exp = []
    
    # Inicializar listas para almacenar historiales de todos los experimentos
    all_fitness_histories = []
    all_F_histories = []
    
    # Directorio para guardar checkpoints
    saves_dir = None
    
    # Si se proporciona un checkpoint específico, cargarlo
    if resume_from and os.path.exists(resume_from):
        print(f"Cargando checkpoint desde {resume_from}...")
        try:
            with open(resume_from, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Extraer información del checkpoint
            exp_idx = checkpoint_data.get('experiment', 0)
            gen = checkpoint_data.get('generation', 0)
            
            # Establecer el experimento y generación de inicio
            start_experiment = exp_idx
            start_generation = gen
            
            # Cargar población y fitness
            population = []
            for p in checkpoint_data.get('population', []):
                population.append({'individual': p.get('individual')})
            
            fitness = np.array(checkpoint_data.get('fitness', []))
            
            # Cargar el mejor modelo del experimento
            best_model_exp = checkpoint_data.get('best_model_exp', {})
            
            # Cargar historiales de fitness y F
            fitness_history_exp = checkpoint_data.get('fitness_history_exp', [])
            f_history_exp = checkpoint_data.get('F_history', [])
            
            # Cargar el mejor modelo global si existe
            if 'best_model' in checkpoint_data:
                best_model_overall = checkpoint_data.get('best_model')
                best_fitness_overall = best_model_overall.get('fitness', float('-inf'))
            
            # Obtener el directorio de guardado
            checkpoint_dir_path = os.path.dirname(resume_from)
            if os.path.exists(checkpoint_dir_path):
                saves_dir = checkpoint_dir_path
            
            print(f"Checkpoint cargado. Reanudando desde experimento {exp_idx + 1}, generación {gen}")
            if best_model_overall:
                print(f"Mejor fitness encontrado hasta ahora: {best_fitness_overall}")
        except Exception as e:
            print(f"Error al cargar checkpoint: {e}")
            print("Iniciando nueva búsqueda")
            new_run = True
    elif not new_run:
        # Buscar el último checkpoint automáticamente
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Encontrado checkpoint automáticamente: {latest_checkpoint}")
            try:
                with open(latest_checkpoint, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Extraer información del checkpoint
                exp_idx = checkpoint_data.get('experiment', 0)
                gen = checkpoint_data.get('generation', 0)
                
                # Establecer el experimento y generación de inicio
                start_experiment = exp_idx
                start_generation = gen
                
                # Cargar población y fitness
                population = []
                for p in checkpoint_data.get('population', []):
                    population.append({'individual': p.get('individual')})
                
                fitness = np.array(checkpoint_data.get('fitness', []))
                
                # Cargar el mejor modelo del experimento
                best_model_exp = checkpoint_data.get('best_model_exp', {})
                
                # Cargar historiales de fitness y F
                fitness_history_exp = checkpoint_data.get('fitness_history_exp', [])
                f_history_exp = checkpoint_data.get('F_history', [])
                
                # Cargar el mejor modelo global si existe
                if 'best_model' in checkpoint_data:
                    best_model_overall = checkpoint_data.get('best_model')
                    best_fitness_overall = best_model_overall.get('fitness', float('-inf'))
                
                # Obtener el directorio de guardado
                checkpoint_dir_path = os.path.dirname(latest_checkpoint)
                if os.path.exists(checkpoint_dir_path):
                    saves_dir = checkpoint_dir_path
                
                print(f"Checkpoint cargado. Reanudando desde experimento {exp_idx + 1}, generación {gen}")
                if best_model_overall:
                    print(f"Mejor fitness encontrado hasta ahora: {best_fitness_overall}")
            except Exception as e:
                print(f"Error al cargar checkpoint: {e}")
                print("Iniciando nueva búsqueda")
                new_run = True
        else:
            print("No se encontraron checkpoints anteriores. Iniciando nueva búsqueda...")
            new_run = True
    
    # Si es una nueva búsqueda o no se encontró un checkpoint válido, crear un nuevo directorio
    if new_run or saves_dir is None:
        # Crear directorio para guardar checkpoints con timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saves_dir = os.path.join(checkpoint_dir, f"saves_{timestamp}")
        os.makedirs(saves_dir, exist_ok=True)
        print(f"Creado nuevo directorio para checkpoints: {saves_dir}")
    
    # Iniciar búsqueda
    for exp_idx in range(start_experiment, n_experiments):
        print(f"\nIniciando experimento {exp_idx + 1}/{n_experiments}")
        
        # Verificar si estamos reanudando un experimento en progreso
        if exp_idx == start_experiment and start_generation > 0 and 'population' in locals() and len(population) > 0:
            print(f"Reanudando experimento {exp_idx + 1} desde la generación {start_generation + 1}")
            # Ya tenemos la población y fitness cargados del checkpoint
        else:
            # Inicializar nueva población y evaluarla
            print("Generando población inicial...")
            population = pop_gen(population_size)
            
            # Evaluar población inicial
            print("Evaluando población inicial...")
            trial_population = []
            for i in range(population_size):
                trial_population.append(population[i]['individual'])
            
            # Usar la función evaluate_population para evaluar toda la población de una vez
            fitness_values = evaluate_population(trial_population, surrogate_model)
            
            for i in range(population_size):
                population[i]['fitness'] = fitness_values[i]
            
            # Inicializar historiales
            fitness_history_exp = []
            f_history_exp = []
            
            # Inicializar mejor modelo del experimento
            best_idx = np.argmax(fitness_values)
            best_model_exp = {
                'individual': population[best_idx]['individual'],
                'fitness': fitness_values[best_idx]
            }
            
            # Inicializar mejor modelo global si es necesario
            if best_model_overall is None:
                best_model_overall = best_model_exp.copy()
                best_fitness_overall = best_model_exp['fitness']
        
        # Evolution loop
        for gen in range(start_generation, generations):
            print(f"\nGeneración {gen + 1}/{generations} (Experimento {exp_idx + 1}/{n_experiments})")
            
            # Initialize trial population
            trial_population = []
            
            # Mutation and crossover
            for i in range(len(population)):
                # Select random indices for mutation
                if selection_method == 'tournament':
                    # Selección por torneo para elegir los individuos para la mutación
                    a_idx = tournament_selection(population, fitness_values, tournament_size=3)
                    
                    # Para b_idx y c_idx, asegurarnos de seleccionar individuos diferentes
                    # Intentar hasta 10 veces encontrar individuos diferentes
                    max_attempts = 10
                    attempts = 0
                    found_different = False
                    
                    while not found_different and attempts < max_attempts:
                        b_idx = tournament_selection(population, fitness_values, tournament_size=3)
                        c_idx = tournament_selection(population, fitness_values, tournament_size=3)
                        
                        # Verificar si los individuos son diferentes comparando sus cromosomas
                        a_ind = population[a_idx]['individual']
                        b_ind = population[b_idx]['individual']
                        c_ind = population[c_idx]['individual']
                        
                        # Verificar si son significativamente diferentes
                        ab_different = are_individuals_different(a_ind, b_ind, threshold=0.1)
                        ac_different = are_individuals_different(a_ind, c_ind, threshold=0.1)
                        bc_different = are_individuals_different(b_ind, c_ind, threshold=0.1)
                        
                        if ab_different and ac_different and bc_different:
                            found_different = True
                        
                        attempts += 1
                    
                    # Si no se encontraron individuos diferentes, seleccionar índices aleatorios
                    if not found_different:
                        indices = list(range(len(population)))
                        np.random.shuffle(indices)
                        a_idx, b_idx, c_idx = indices[:3]
                        
                    # Get individuals for mutation
                    a, b, c = population[a_idx]['individual'], population[b_idx]['individual'], population[c_idx]['individual']
                elif selection_method == 'sus':
                    # Stochastic Universal Sampling
                    selected_indices = sus_selection(fitness_values, 3)
                    a_idx, b_idx, c_idx = selected_indices
                    a, b, c = population[a_idx]['individual'], population[b_idx]['individual'], population[c_idx]['individual']
                else:
                    # Selección aleatoria clásica de DE
                    indices = list(range(len(population)))
                    indices.remove(i)  # Remove current index
                    a_idx, b_idx, c_idx = np.random.choice(indices, 3, replace=False)
                    a, b, c = population[a_idx]['individual'], population[b_idx]['individual'], population[c_idx]['individual']
                
                # Create mutant vector using DE/rand/1 strategy
                a_real = convert_individual(a, to_real=True)
                b_real = convert_individual(b, to_real=True)
                c_real = convert_individual(c, to_real=True)
                
                mutant = []
                for j in range(len(a_real)):
                    mutant.append(a_real[j] + F * (b_real[j] - c_real[j]))
                
                # Convert back to integer representation for fixArch
                mutant_int = convert_individual(mutant, to_real=False)
                
                # Fix architecture to ensure valid encoding
                mutant_fixed = fixArch(mutant_int, verbose=False)
                
                # Perform crossover
                if selection_method == 'tournament':
                    # Seleccionar padres para cruce mediante torneo
                    parent_idx = tournament_selection(population, fitness_values, tournament_size=3)
                    parent = population[parent_idx]['individual']
                elif selection_method == 'sus':
                    # Stochastic Universal Sampling para seleccionar padre
                    selected_indices = sus_selection(fitness_values, 1)
                    parent_idx = selected_indices[0]
                    parent = population[parent_idx]['individual']
                else:
                    parent = population[i]['individual']
                
                trial = crossover(convert_individual(mutant_fixed, to_real=True), 
                                 convert_individual(parent, to_real=True), 
                                 cr_rate)
                
                # Convert back to integer representation for evaluation
                trial_int = convert_individual(trial, to_real=False)
                
                # Fix architecture again to ensure valid encoding after crossover
                trial_fixed = fixArch(trial_int, verbose=False)
                
                # Add to trial population
                trial_population.append(trial_fixed)
            
            # Evaluate trial population
            print("Evaluando población de prueba...")
            trial_population_list = []
            for i in range(population_size):
                trial_population_list.append(trial_population[i])
            
            # Usar la función evaluate_population para evaluar toda la población de una vez
            trial_fitness = evaluate_population(trial_population_list, surrogate_model)
            
            # Auto-adaptation of F parameter based on successful mutations
            if auto_adaptation and gen > 0:
                # Contar mutaciones exitosas
                succ_m = get_succ_m(trial_fitness, fitness_values)
                succ_rate = succ_m / len(population)
                
                # Ajustar F según la tasa de éxito
                if succ_rate < 0.2:  # Pocas mutaciones exitosas, reducir F
                    F = max(0.1, F * 0.8)
                    print(f"Pocas mutaciones exitosas ({succ_rate:.2f}). Reduciendo F a {F:.4f}")
                elif succ_rate > 0.3:  # Muchas mutaciones exitosas, aumentar F
                    F = min(1.0, F * 1.2)
                    print(f"Muchas mutaciones exitosas ({succ_rate:.2f}). Aumentando F a {F:.4f}")
                else:
                    print(f"Tasa de mutaciones exitosas: {succ_rate:.2f}. Manteniendo F = {F:.4f}")
            
            # Guardar F para graficar
            f_history_exp.append(F)
            
            # Selection
            for i in range(len(population)):
                if selection_method == 'tournament':
                    # En modo torneo, seleccionamos individuos para competir
                    # El individuo de prueba compite con un individuo seleccionado por torneo
                    competitor_idx = tournament_selection(population, fitness_values, tournament_size=3)
                    
                    # Verificar si el individuo de prueba es diferente al competidor
                    trial_ind = trial_population[i]
                    competitor_ind = population[competitor_idx]['individual']
                    
                    # Verificar si son significativamente diferentes
                    are_different = are_individuals_different(trial_ind, competitor_ind, threshold=0.1)
                    
                    # Solo reemplazar si el fitness es mejor Y son individuos diferentes
                    # o si el fitness es significativamente mejor (>5%)
                    if (trial_fitness[i] > population[competitor_idx]['fitness'] and 
                        (are_different or trial_fitness[i] > population[competitor_idx]['fitness'] * 1.05)):
                        population[competitor_idx] = {
                            'individual': trial_population[i],
                            'fitness': trial_fitness[i]
                        }
                elif selection_method == 'sus':
                    # Stochastic Universal Sampling para seleccionar individuos
                    selected_indices = sus_selection(fitness_values, 1)
                    competitor_idx = selected_indices[0]
                    
                    # Verificar si el individuo de prueba es diferente al competidor
                    trial_ind = trial_population[i]
                    competitor_ind = population[competitor_idx]['individual']
                    
                    # Verificar si son significativamente diferentes
                    are_different = are_individuals_different(trial_ind, competitor_ind, threshold=0.1)
                    
                    # Solo reemplazar si el fitness es mejor Y son individuos diferentes
                    # o si el fitness es significativamente mejor (>5%)
                    if (trial_fitness[i] > population[competitor_idx]['fitness'] and 
                        (are_different or trial_fitness[i] > population[competitor_idx]['fitness'] * 1.05)):
                        population[competitor_idx] = {
                            'individual': trial_population[i],
                            'fitness': trial_fitness[i]
                        }
                else:
                    # Selección clásica de DE (one-to-one)
                    # Solo reemplazar si el fitness es mejor
                    if trial_fitness[i] > population[i]['fitness']:
                        population[i] = {
                            'individual': trial_population[i],
                            'fitness': trial_fitness[i]
                        }
            
            # Update fitness array
            fitness_values = np.array([ind['fitness'] for ind in population])
            
            # Update best model
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > best_model_exp['fitness']:
                best_model_exp = {
                    'individual': population[best_idx]['individual'],
                    'fitness': fitness_values[best_idx]
                }
            
            # Update fitness history
            fitness_history_exp.append(np.mean(fitness_values))
            
            # Print current best fitness
            best_fitness = best_model_exp['fitness']
            if isinstance(best_fitness, np.ndarray):
                best_fitness = best_fitness.item()
            print(f"Mejor fitness en generación {gen+1}: {best_fitness:.6f}")
            
            # Imprimir top 5 de fitness
            sorted_indices = np.argsort(fitness_values)[::-1]  # Ordenar de mayor a menor
            print(f"\nTop 5 fitness en generación {gen+1}:")
            for j in range(min(5, len(population))):
                idx = int(sorted_indices[j])  # Convertir a entero escalar
                # Obtener el fitness directamente del individuo en la población
                fitness_value = population[idx]['fitness']
                if isinstance(fitness_value, np.ndarray):
                    fitness_value = fitness_value.item()
                
                # Obtener el individuo completo
                ind = population[idx]['individual']
                
                # Mostrar los primeros 5 y últimos 5 elementos
                if isinstance(ind, np.ndarray):
                    ind_preview_start = ind[:5].tolist()
                    ind_preview_end = ind[-5:].tolist()
                else:
                    ind_preview_start = ind[:5]
                    ind_preview_end = ind[-5:]
                
                print(f"  {j+1}. Fitness: {fitness_value:.6f}")
                print(f"     Primeros 5: {ind_preview_start}")
                print(f"     Últimos 5: {ind_preview_end}")
                print(f"     Longitud: {len(ind)}")
            
            # Verificar diversidad de la población
            unique_individuals = []
            for i, ind in enumerate(population):
                is_unique = True
                ind_array = np.array(ind['individual'])
                
                # Comparar con individuos ya identificados como únicos
                for unique_ind in unique_individuals:
                    unique_array = np.array(unique_ind['individual'])
                    if not are_individuals_different(ind_array, unique_array, threshold=0.1):
                        is_unique = False
                        break
                
                if is_unique:
                    unique_individuals.append(ind)
            
            print(f"Diversidad: {len(unique_individuals)}/{len(population)} individuos significativamente diferentes (umbral 10%)")
            
            # Imprimir valor F actual
            print(f"Valor F actual: {F:.6f}")
            
            # Save checkpoint every 10 generations or at the end
            if (gen + 1) % 10 == 0 or gen == generations - 1:
                print(f"Guardando checkpoint en generación {gen+1}...")
                
                # Si es el primer experimento y generación, best_model_overall podría ser None
                if best_model_overall is None:
                    best_model_overall = {
                        'individual': best_model_exp['individual'],
                        'fitness': best_model_exp['fitness']
                    }
                
                # Asegurar que el directorio existe
                os.makedirs(saves_dir, exist_ok=True)
                
                # Obtener el número del próximo checkpoint
                checkpoint_files = [f for f in os.listdir(saves_dir) if f.startswith('checkpoint_') and f.endswith('.json')]
                checkpoint_numbers = [int(re.search(r'checkpoint_(\d+)_', f).group(1)) for f in checkpoint_files if re.search(r'checkpoint_(\d+)_', f)]
                checkpoint_number = 1 if not checkpoint_numbers else max(checkpoint_numbers) + 1
                
                checkpoint_data = {
                    'checkpoint_number': checkpoint_number,
                    'experiment': exp_idx,
                    'generation': gen + 1,
                    'population': [
                        {
                            'individual': p['individual'],
                            # Usar el valor de fitness del array de fitness en lugar de buscarlo en el diccionario
                            'fitness': float(fitness_values[i]) if isinstance(fitness_values[i], (np.ndarray, np.number)) else fitness_values[i]
                        } for i, p in enumerate(population)
                    ],
                    'fitness': [float(f) for f in fitness_values] if isinstance(fitness_values, np.ndarray) else fitness_values,
                    'best_model_exp': {
                        'individual': best_model_exp['individual'],
                        'fitness': float(best_model_exp['fitness']) if isinstance(best_model_exp['fitness'], (np.ndarray, np.number)) else best_model_exp['fitness']
                    },
                    'best_model': {
                        'individual': best_model_overall['individual'],
                        'fitness': float(best_model_overall['fitness']) if isinstance(best_model_overall['fitness'], (np.ndarray, np.number)) else best_model_overall['fitness']
                    },
                    'fitness_history_exp': [float(x) for x in fitness_history_exp],
                    'F': float(F) if isinstance(F, (np.ndarray, np.number)) else F,
                    'cr_rate': float(cr_rate) if isinstance(cr_rate, (np.ndarray, np.number)) else cr_rate
                }
                
                # Convertir todo a tipos Python nativos
                checkpoint_data = numpy_to_python(checkpoint_data)
                
                # Save checkpoint
                checkpoint_path = os.path.join(saves_dir, f"checkpoint_{checkpoint_number}_exp{exp_idx+1}_gen{gen+1}.json")
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                print(f"Checkpoint guardado en {checkpoint_path}")
        
        # Get top 3 models from current experiment
        sorted_indices = np.argsort(fitness_values)[::-1]  # Ordenados de mayor a menor fitness
        top_3_models = []
        
        # Tomamos los 3 mejores modelos
        for i in range(min(3, len(sorted_indices))):
            idx = int(sorted_indices[i])
            top_3_models.append({
                'individual': population[idx]['individual'],
                'fitness': float(fitness_values[idx]) if isinstance(fitness_values[idx], (np.ndarray, np.number)) else fitness_values[idx]
            })
        
        # Update best model overall if needed
        if top_3_models[0]['fitness'] > best_fitness_overall:
            best_fitness_overall = top_3_models[0]['fitness']
            best_model_overall = top_3_models[0].copy()
        
        # Generar y guardar gráficas de métricas
        plot_and_save_metrics(fitness_history_exp, f_history_exp, saves_dir, exp_idx)
        
        # Almacenar historiales para la gráfica combinada
        all_fitness_histories = []
        all_F_histories = []
        all_fitness_histories.append(fitness_history_exp)
        all_F_histories.append(f_history_exp)
        
        # Store results from this experiment
        top_models_per_experiment = []
        all_best_models = []
        all_fitness_histories = []
        all_F_histories = []
        top_models_per_experiment.append({
            'top_3_models': [
                {
                    'individual': model['individual'],
                    'fitness': float(model['fitness']) if isinstance(model['fitness'], (np.ndarray, np.number)) else model['fitness']
                } for model in top_3_models
            ],
            'fitness_history': [float(x) if isinstance(x, (np.ndarray, np.number)) else x for x in fitness_history_exp],
            'F_history': [float(x) if isinstance(x, (np.ndarray, np.number)) else x for x in f_history_exp]
        })
        all_best_models.append(top_3_models[0])
        all_fitness_histories.append(fitness_history_exp)
        all_F_histories.append(f_history_exp)
        
        # Save final checkpoint for this experiment
        final_checkpoint_path = os.path.join(saves_dir, f'final_checkpoint_exp{exp_idx+1}.json')
        
        # Prepare checkpoint data
        checkpoint_data = {
            'experiment': exp_idx,
            'generation': generations,
            'population': [{'individual': p['individual']} for p in population],
            'fitness': [float(f) for f in fitness_values],
            'best_model_exp': {
                'individual': best_model_exp['individual'],
                'fitness': float(best_model_exp['fitness']) if isinstance(best_model_exp['fitness'], (np.ndarray, np.number)) else best_model_exp['fitness']
            },
            'top_3_models': [
                {
                    'individual': model['individual'],
                    'fitness': float(model['fitness']) if isinstance(model['fitness'], (np.ndarray, np.number)) else model['fitness']
                } for model in top_3_models
            ],
            'fitness_history': [float(f) for f in fitness_history_exp],
            'F_history': [float(f) for f in f_history_exp]
        }
        
        # Convertir todo a tipos Python nativos
        checkpoint_data = numpy_to_python(checkpoint_data)
        
        with open(final_checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Checkpoint final del experimento guardado en {final_checkpoint_path}")
        print(f"Mejor fitness en experimento {exp_idx+1}: {top_3_models[0]['fitness']}")
        
        # Reiniciar start_generation para los siguientes experimentos
        start_generation = 0
    
    # Prepare final results
    results = {
        'best_model': {
            'individual': best_model_overall['individual'],
            'fitness': float(best_model_overall['fitness']) if isinstance(best_model_overall['fitness'], (np.ndarray, np.number)) else best_model_overall['fitness']
        },
        'top_models_per_experiment': top_models_per_experiment,
        'all_fitness_histories': [[float(f) if isinstance(f, (np.ndarray, np.number)) else f for f in history] for history in all_fitness_histories],
        'all_F_histories': [[float(f) if isinstance(f, (np.ndarray, np.number)) else f for f in history] for history in all_F_histories]
    }
    
    # Convertir todo a tipos Python nativos
    results = numpy_to_python(results)
    
    # Guardar checkpoint final
    final_checkpoint_path = os.path.join(saves_dir, f'final_checkpoint.json')
    final_checkpoint_data = {
        'best_fitness_overall': float(best_fitness_overall) if isinstance(best_fitness_overall, (np.ndarray, np.number)) else best_fitness_overall,
        'best_model_overall': {
            'individual': best_model_overall['individual'],
            'fitness': float(best_model_overall['fitness']) if isinstance(best_model_overall['fitness'], (np.ndarray, np.number)) else best_model_overall['fitness']
        },
        'top_models_per_experiment': numpy_to_python(top_models_per_experiment),
        'all_best_models': [
            {
                'individual': model['individual'],
                'fitness': float(model['fitness']) if isinstance(model['fitness'], (np.ndarray, np.number)) else model['fitness']
            } for model in all_best_models
        ],
        'all_fitness_histories': numpy_to_python(all_fitness_histories),
        'all_F_histories': numpy_to_python(all_F_histories)
    }
    
    # Convertir todo a tipos Python nativos
    final_checkpoint_data = numpy_to_python(final_checkpoint_data)
    
    with open(final_checkpoint_path, 'w') as f:
        json.dump(final_checkpoint_data, f, indent=2)
    
    print(f"\nBúsqueda completada. Checkpoint final guardado en {final_checkpoint_path}")
    
    # Generar gráfica combinada de todos los experimentos
    plot_combined_metrics(all_fitness_histories, all_F_histories, saves_dir, n_experiments)
    
    return results

def plot_and_save_metrics(fitness_history, f_history, save_path, exp_idx):
    """
    Genera y guarda gráficas de fitness y valor F para un experimento.
    
    Args:
        fitness_history: Lista de fitness por generación
        f_history: Lista de valores F por generación
        save_path: Ruta donde guardar las gráficas
        exp_idx: Índice del experimento
    """
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Graficar fitness
    generations_fitness = range(1, len(fitness_history) + 1)
    ax1.plot(generations_fitness, fitness_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title(f'Fitness por Generación - Experimento {exp_idx+1}')
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Fitness (mayor es mejor)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir línea de tendencia
    if len(fitness_history) > 1:
        z = np.polyfit(generations_fitness, fitness_history, 1)
        p = np.poly1d(z)
        ax1.plot(generations_fitness, p(generations_fitness), "r--", alpha=0.5, label=f"Tendencia: {z[0]:.4f}x + {z[1]:.4f}")
        ax1.legend()
    
    # Graficar valor F
    if f_history:
        generations_f = range(1, len(f_history) + 1)
        ax2.plot(generations_f, f_history, 'r-', linewidth=2, marker='o', markersize=4)
        ax2.set_title(f'Valor F por Generación - Experimento {exp_idx+1}')
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Valor F (factor de mutación)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir línea de tendencia
        if len(f_history) > 1:
            z = np.polyfit(generations_f, f_history, 1)
            p = np.poly1d(z)
            ax2.plot(generations_f, p(generations_f), "b--", alpha=0.5, label=f"Tendencia: {z[0]:.4f}x + {z[1]:.4f}")
            ax2.legend()
    
    # Añadir información adicional
    plt.figtext(0.5, 0.01, f"Mejor fitness: {max(fitness_history):.4f}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    # Ajustar layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Guardar figura
    plt.savefig(os.path.join(save_path, f'metrics_exp{exp_idx+1}.png'), dpi=300)
    plt.close(fig)

def plot_combined_metrics(all_fitness_histories, all_f_histories, save_path, n_experiments):
    """
    Genera y guarda gráficas combinadas de todos los experimentos.
    
    Args:
        all_fitness_histories: Lista de historiales de fitness por experimento
        all_f_histories: Lista de historiales de valor F por experimento
        save_path: Ruta donde guardar las gráficas
        n_experiments: Número total de experimentos
    """
    # Verificar que hay datos de fitness para graficar
    valid_fitness_histories = [h for h in all_fitness_histories if h]
    if not valid_fitness_histories:
        print("No hay datos de fitness para generar gráficas combinadas.")
        return
    
    # Crear figura para fitness
    plt.figure(figsize=(12, 8))
    
    # Determinar el número máximo de generaciones
    max_generations = max(len(fitness) for fitness in valid_fitness_histories)
    generations = np.arange(1, max_generations + 1)
    
    # Inicializar matriz para almacenar valores de fitness
    fitness_matrix = np.full((n_experiments, max_generations), np.nan)
    
    # Llenar la matriz con los valores de fitness
    for i, fitness_history in enumerate(all_fitness_histories):
        if fitness_history:
            fitness_length = len(fitness_history)
            fitness_matrix[i, :fitness_length] = fitness_history
    
    # Graficar todas las ejecuciones en gris claro
    for i in range(n_experiments):
        if not np.all(np.isnan(fitness_matrix[i])):
            plt.plot(
                generations,
                fitness_matrix[i],
                linestyle='-',
                color='red',
                alpha=0.5
            )
    
    # Calcular la media y desviación estándar
    mean_fitness = np.nanmean(fitness_matrix, axis=0)
    std_fitness = np.nanstd(fitness_matrix, axis=0)
    
    # Graficar la media del fitness
    plt.plot(
        generations,
        mean_fitness,
        linestyle='-',
        color='blue',
        linewidth=2,
        label='Fitness Medio'
    )
    
    # Rellenar el área entre (media - std) y (media + std)
    plt.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        color='blue',
        alpha=0.2,
        label='Desviación Estándar'
    )
    
    plt.title('Convergencia de Fitness por Generación en Todos los Experimentos')
    plt.xlabel('Generaciones')
    plt.ylabel('Fitness (mayor es mejor)')
    plt.legend()
    plt.grid(True)
    
    # Guardar gráfica de fitness
    plt.savefig(os.path.join(save_path, 'combined_fitness.png'), dpi=300)
    plt.close()
    
    # Verificar que hay datos de valor F para graficar
    valid_f_histories = [h for h in all_f_histories if h]
    if not valid_f_histories:
        print("No hay datos de valor F para generar gráficas combinadas.")
        return
    
    # Crear figura para valor F
    plt.figure(figsize=(12, 8))
    
    # Determinar el número máximo de generaciones para F
    max_generations_f = max(len(f_history) for f_history in valid_f_histories)
    generations_f = np.arange(1, max_generations_f + 1)
    
    # Inicializar matriz para almacenar valores de F
    f_matrix = np.full((n_experiments, max_generations_f), np.nan)
    
    # Llenar la matriz con los valores de F
    for i, f_history in enumerate(all_f_histories):
        if f_history:
            f_length = len(f_history)
            f_matrix[i, :f_length] = f_history
    
    # Graficar todas las ejecuciones en gris claro
    for i in range(n_experiments):
        if not np.all(np.isnan(f_matrix[i])):
            plt.plot(
                generations_f,
                f_matrix[i],
                linestyle='-',
                color='green',
                alpha=0.5
            )
    
    # Calcular la media y desviación estándar
    mean_f = np.nanmean(f_matrix, axis=0)
    std_f = np.nanstd(f_matrix, axis=0)
    
    # Graficar la media del valor F
    plt.plot(
        generations_f,
        mean_f,
        linestyle='-',
        color='purple',
        linewidth=2,
        label='Valor F Medio'
    )
    
    # Rellenar el área entre (media - std) y (media + std)
    plt.fill_between(
        generations_f,
        mean_f - std_f,
        mean_f + std_f,
        color='purple',
        alpha=0.2,
        label='Desviación Estándar'
    )
    
    plt.title('Evolución del Factor F por Generación en Todos los Experimentos')
    plt.xlabel('Generaciones')
    plt.ylabel('Valor F (factor de mutación)')
    plt.legend()
    plt.grid(True)
    
    # Guardar gráfica de valor F
    plt.savefig(os.path.join(save_path, 'combined_f_values.png'), dpi=300)
    plt.close()