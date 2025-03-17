import numpy as np
import tensorflow as tf
import copy
import os
import json
from tqdm import tqdm
from .utils.encoding import decode_model_architecture, convert_individual, fixArch, encode_model_architecture, layer_type_options
from .utils.latin_hypercube import generate_latin_hypercube_samples

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
    """Load the surrogate model for architecture evaluation."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.h5':
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        raise ValueError(f"Unsupported model extension: {ext}")

def evaluate_architecture(ind, surrogate_model):
    """Evaluate an architecture using the surrogate model."""
    ind_copy = copy.deepcopy(ind)
    reshaped_ind = np.array(ind_copy).reshape(1, -1)
    return surrogate_model.predict(reshaped_ind, verbose=0)[0]

def crossover(parent1, parent2, cr_rate=0.5):
    """Perform crossover between two parent architectures."""
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    mask = np.random.random(len(parent1)) < cr_rate
    child = np.where(mask, parent1, parent2)
    return child

def unified_search(surrogate_model, population_size=10, generations=100, n_experiments=1, 
                  F=0.5, cr_rate=0.5, auto_adaptation=True, checkpoint_dir='./checkpoints'):
    """
    Ejecuta la estrategia evolutiva con múltiples experimentos.
    
    Args:
        surrogate_model: Modelo sustituto para evaluar las arquitecturas.
        population_size: Tamaño de la población.
        generations: Número de generaciones.
        n_experiments: Número de experimentos a ejecutar.
        F: Factor de mutación.
        cr_rate: Tasa de cruce.
        auto_adaptation: Si se debe adaptar automáticamente el factor F.
        checkpoint_dir: Directorio para guardar los checkpoints.
    
    Returns:
        dict: Resultados de la búsqueda, incluyendo el mejor modelo general y los mejores modelos por experimento.
    """
    # Crear directorio de checkpoints si no existe
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Guardando checkpoints en: {checkpoint_dir}")
    
    # Variables para almacenar resultados
    best_fitness_overall = float('-inf')
    best_model_overall = None
    top_models_per_experiment = {}
    all_best_models = []
    all_fitness_histories = []
    all_F_histories = []
    
    # Ejecutar n experimentos
    for exp_idx in range(n_experiments):
        print(f"\nIniciando experimento {exp_idx+1}/{n_experiments}")
        
        # Initialize population using pop_gen function
        print(f"Inicializando población de tamaño {population_size}...")
        population = pop_gen(population_size)
        
        # Evaluate initial population
        fitness = np.array([evaluate_architecture(ind['individual'], surrogate_model) for ind in tqdm(population, desc="Evaluando población inicial")])
        
        # Initialize best model for this experiment
        best_idx = np.argmax(fitness)
        best_model_exp = {
            'individual': population[best_idx]['individual'],
            'fitness': fitness[best_idx]
        }
        
        # Initialize fitness history and F history
        fitness_history_exp = [np.mean(fitness)]
        f_history_exp = [F]
        
        # Evolution loop
        for gen in range(generations):
            print(f"\nGeneración {gen+1}/{generations} (Experimento {exp_idx+1}/{n_experiments})")
            
            # Auto-adaptation of F parameter
            if auto_adaptation and gen > 0:
                if fitness_history_exp[-1] <= fitness_history_exp[-2]:
                    F = np.random.uniform(0.1, 1.0)
                    print(f"Adaptando F a {F:.4f}")
            
            # Initialize trial population
            trial_population = []
            
            # Mutation and crossover
            for i in range(population_size):
                # Select random indices for mutation
                while True:
                    indices = np.random.choice(population_size, 3, replace=False)
                    if i not in indices:
                        break
                    if population_size <= 3:
                        indices = np.random.choice(population_size, 3, replace=False)
                
                # Get individuals for mutation
                a, b, c = population[indices[0]]['individual'], population[indices[1]]['individual'], population[indices[2]]['individual']
                
                # Create mutant vector
                mutant = np.array(a) + F * (np.array(b) - np.array(c))
                
                # Perform crossover
                trial = crossover(mutant, population[i]['individual'], cr_rate)
                
                # Fix architecture to ensure valid encoding
                trial = fixArch(trial.tolist(), verbose=False)
                
                # Add to trial population
                trial_population.append(trial)
            
            # Evaluate trial population
            trial_fitness = np.array([evaluate_architecture(ind, surrogate_model) for ind in tqdm(trial_population, desc="Evaluando población de prueba")])
            
            # Selection
            for i in range(population_size):
                if trial_fitness[i] > fitness[i]:
                    population[i]['individual'] = trial_population[i]
                    fitness[i] = trial_fitness[i]
                    
                    # Update best model
                    if fitness[i] > best_model_exp['fitness']:
                        best_model_exp = {
                            'individual': population[i]['individual'],
                            'fitness': fitness[i]
                        }
            
            # Update fitness history
            fitness_history_exp.append(np.mean(fitness))
            f_history_exp.append(F)
            
            # Print current best fitness
            print(f"Mejor fitness en generación {gen+1}: {best_model_exp['fitness']}")
        
        # Get top 3 models from current experiment
        # Primero ordenamos todos los individuos por fitness
        sorted_indices = np.argsort(fitness)[::-1]  # Ordenados de mayor a menor fitness
        
        # Tomamos los 3 mejores modelos diferentes
        top_3_models = []
        used_architectures = set()
        
        for i in range(len(sorted_indices)):
            idx = int(sorted_indices[i])
            current_arch = tuple(convert_individual(population[idx]['individual'], to_real=False))
            
            # Solo añadimos si la arquitectura no está ya en el conjunto
            if current_arch not in used_architectures:
                top_3_models.append({
                    'individual': population[idx]['individual'],
                    'fitness': float(fitness[idx])
                })
                used_architectures.add(current_arch)
                
                # Si ya tenemos 3 modelos diferentes, terminamos
                if len(top_3_models) >= 3:
                    break
        
        # Si no tenemos suficientes modelos diferentes, tomamos los siguientes mejores
        if len(top_3_models) < 3:
            print(f"Advertencia: Solo se encontraron {len(top_3_models)} arquitecturas diferentes en el experimento {exp_idx+1}")
            for i in range(len(sorted_indices)):
                idx = int(sorted_indices[i])
                # Verificamos si este modelo ya está en nuestra lista
                already_added = False
                for model in top_3_models:
                    if np.array_equal(population[idx]['individual'], model['individual']):
                        already_added = True
                        break
                
                if not already_added:
                    top_3_models.append({
                        'individual': population[idx]['individual'],
                        'fitness': float(fitness[idx])
                    })
                    
                    if len(top_3_models) >= 3:
                        break
        
        # Convert real-valued individuals to integer representation
        top_3_encoded = [{
            'rank': idx + 1,
            'encoded_architecture': convert_individual(model['individual'], to_real=False),
            'fitness': float(model['fitness'])
        } for idx, model in enumerate(top_3_models)]
        
        # Update best overall model if current experiment found better solution
        if top_3_models[0]['fitness'] > best_fitness_overall:
            best_fitness_overall = top_3_models[0]['fitness']
            best_model_overall = top_3_models[0].copy()
        
        # Store results from this experiment
        top_models_per_experiment[f"experiment_{exp_idx + 1}"] = {
            'top_3_models': top_3_encoded,
            'fitness_history': [float(x) for x in fitness_history_exp],
            'F_history': [float(x) for x in f_history_exp]
        }
        all_best_models.append(top_3_models[0])
        all_fitness_histories.append(fitness_history_exp)
        all_F_histories.append(f_history_exp)
        
        # Save checkpoint after each experiment
        checkpoint_path = os.path.join(checkpoint_dir, 'all_experiments.json')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(top_models_per_experiment, f, indent=2)
        
        print(f"Resultados guardados en {checkpoint_path}")
        print(f"Mejor fitness en experimento {exp_idx+1}: {top_3_models[0]['fitness']}")
    
    # Save final results
    best_architectures = {
        "best_overall": {
            "encoded_architecture": convert_individual(best_model_overall['individual'], to_real=False),
            "fitness": float(best_model_overall['fitness'])
        },
        "experiments": top_models_per_experiment
    }
    
    best_arch_path = os.path.join(checkpoint_dir, 'best_architectures.json')
    os.makedirs(os.path.dirname(best_arch_path), exist_ok=True)
    with open(best_arch_path, 'w') as f:
        json.dump(best_architectures, f, indent=2)
    
    print(f"Mejores arquitecturas guardadas en {best_arch_path}")
    
    return {
        'best_model': best_model_overall,
        'top_models_per_experiment': top_models_per_experiment,
        'all_best_models': all_best_models,
        'all_fitness_histories': all_fitness_histories,
        'all_F_histories': all_F_histories
    }