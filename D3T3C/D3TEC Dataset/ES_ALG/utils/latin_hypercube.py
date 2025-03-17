import numpy as np
import pandas as pd
from pyDOE2 import lhs
from .encoding import (
    layer_type_options,
    stride_options,
    dropout_options,
    activation_options,
    encode_layer_params,
    encode_model_architecture,
    fixArch
)

def generate_latin_hypercube_samples(num_samples, dimensions):
    """
    Genera muestras usando Latin Hypercube Sampling y las convierte a valores enteros
    adecuados para la codificación de arquitecturas de redes neuronales.
    
    Args:
        num_samples: Número de muestras a generar
        dimensions: Número de dimensiones (alelos) para cada muestra
    
    Returns:
        Lista de arquitecturas codificadas como listas de enteros
    """
    # Generar muestras en el espacio continuo [0,1]
    lhs_samples = lhs(dimensions, samples=num_samples)
    
    # Convertir las muestras a valores enteros adecuados para la arquitectura
    encoded_samples = []
    
    for sample in lhs_samples:
        encoded_sample = []
        
        # Procesar cada grupo de 4 valores (tipo de capa + 3 parámetros)
        for i in range(0, dimensions, 4):
            if i + 3 < dimensions:
                # Convertir el tipo de capa (0-8)
                layer_type = int(sample[i] * 8)  # 9 tipos de capas (0-8)
                encoded_sample.append(layer_type)
                
                # Procesar parámetros según el tipo de capa
                if layer_type == 0:  # Conv2D
                    filters = int(sample[i+1] * 28) + 4  # Filtros entre [4, 32]
                    stride_idx = int(sample[i+2] * 2)    # Stride 0 o 1
                    activation_idx = int(sample[i+3] * 4) # Activación 0-3
                    encoded_sample.extend([filters, stride_idx, activation_idx])
                
                elif layer_type == 6:  # SelfAttention
                    filters = int(sample[i+1] * 60) + 4  # Filtros entre [4, 64]
                    attention_heads = int(sample[i+2] * 7) + 1  # Heads entre [1, 8]
                    activation_idx = int(sample[i+3] * 4)  # Activación 0-3
                    encoded_sample.extend([filters, attention_heads, activation_idx])
                
                elif layer_type == 2:  # MaxPooling
                    stride_idx = int(sample[i+2] * 2)  # Stride 0 o 1
                    encoded_sample.extend([stride_idx, 0, 0])
                
                elif layer_type == 3:  # Dropout
                    rate_idx = int(sample[i+1] * 4)  # Índice de tasa de dropout 0-3
                    encoded_sample.extend([rate_idx, 0, 0])
                
                elif layer_type == 4:  # Dense
                    units = int(sample[i+1] * 511) + 1  # Unidades entre [1, 512]
                    activation_idx = int(sample[i+2] * 4)  # Activación 0-3
                    encoded_sample.extend([units, activation_idx, 0])
                
                elif layer_type == 8:  # Repetition
                    layers_to_repeat = int(sample[i+1] * 3) + 1  # Capas a repetir [1, 4]
                    repetition_count = int(sample[i+2] * 31) + 1  # Número de repeticiones [1, 32]
                    encoded_sample.extend([layers_to_repeat, repetition_count, 0])
                
                else:  # BatchNorm, Flatten, DontCare
                    encoded_sample.extend([0, 0, 0])
            else:
                # Si no hay suficientes valores para completar un grupo de 4,
                # rellenar con ceros
                remaining = dimensions - i
                encoded_sample.extend([0] * remaining)
        
        encoded_samples.append(encoded_sample)
    
    return encoded_samples

def validate_latin_hypercube(num_models=100):
    dimensions = 12 * 3  # 12 capas, 3 parámetros por capa
    latin_samples = generate_latin_hypercube_samples(num_models, dimensions)
    
    # Validar cada muestra generada
    for sample_idx, sample in enumerate(latin_samples):
        reshaped_sample = sample.reshape(12, 3)  # Cada modelo tiene 12 capas
        
        for layer_idx, layer_params in enumerate(reshaped_sample):
            type_idx = int(layer_params[0] * 9)  # 9 tipos de capas
            param1 = layer_params[1]
            param2 = layer_params[2]

            layer_mapping = ['Conv2D', 'SelfAttention', 'BatchNorm', 'MaxPooling', 
                             'Dropout', 'Dense', 'Flatten', 'DontCare', 'Repetition']
            layer_type = layer_mapping[type_idx]

            if layer_type == 'Conv2D':
                filters = int(param1 * (32 - 4) + 4)  # Filtros entre [4, 32]
                if not (4 <= filters <= 32):
                    print(f"ERROR en Modelo {sample_idx + 1}, Capa {layer_idx + 1}: Filtros fuera de rango {filters}")
                    return False

            elif layer_type == 'SelfAttention':
                filters = int(param1 * (64 - 4) + 4)  # Filtros entre [4, 64]
                attention_heads = int(param2 * (8 - 1) + 1)  # Heads entre [1, 8]
                if not (4 <= filters <= 64) or not (1 <= attention_heads <= 8):
                    print(f"ERROR en Modelo {sample_idx + 1}, Capa {layer_idx + 1}: Parámetros fuera de rango")
                    return False

            elif layer_type == 'Dropout':
                rate = param1 * (0.5 - 0.2) + 0.2  # Tasa de dropout entre [0.2, 0.5]
                if not (0.2 <= rate <= 0.5):
                    print(f"ERROR en Modelo {sample_idx + 1}, Capa {layer_idx + 1}: Dropout fuera de rango {rate}")
                    return False

            elif layer_type == 'Dense':
                units = int(param1 * (512 - 1) + 1)  # Unidades entre [1, 512]
                if not (1 <= units <= 512):
                    print(f"ERROR en Modelo {sample_idx + 1}, Capa {layer_idx + 1}: Unidades fuera de rango {units}")
                    return False

    print(f"Validación exitosa para {num_models} modelos")
    return True

def map_to_architecture_params(latin_hypercube_sample):
    """
    Mapea una muestra del hipercubo latino a parámetros de arquitectura de red neuronal.
    
    Args:
        latin_hypercube_sample: Muestra del hipercubo latino (valores entre 0 y 1)
        
    Returns:
        Diccionario con la arquitectura de la red neuronal
    """
    model_dict = {'layers': []}
    
    # Asumimos que la muestra tiene dimensiones múltiplo de 3 (tipo, param1, param2)
    num_layers = len(latin_hypercube_sample) // 3
    
    for i in range(num_layers):
        layer_params = latin_hypercube_sample[i*3:(i+1)*3]
        
        # Mapear tipo de capa (9 tipos posibles)
        layer_type_idx = int(layer_params[0] * 9)
        layer_mapping = ['Conv2D', 'SelfAttention', 'BatchNorm', 'MaxPooling', 
                         'Dropout', 'Dense', 'Flatten', 'DontCare', 'Repetition']
        
        if layer_type_idx >= len(layer_mapping):
            layer_type = 'DontCare'
        else:
            layer_type = layer_mapping[layer_type_idx]
        
        # Crear diccionario de capa según su tipo
        layer_dict = {'type': layer_type}
        
        if layer_type == 'Conv2D':
            layer_dict['filters'] = int(layer_params[1] * (32 - 4) + 4)  # [4, 32]
            stride_idx = int(layer_params[2] * 2)  # [0, 1]
            layer_dict['strides'] = stride_options.get(stride_idx, 1)
            layer_dict['activation'] = 'relu'  # Default
            
        elif layer_type == 'SelfAttention':
            layer_dict['filters'] = int(layer_params[1] * (64 - 4) + 4)  # [4, 64]
            layer_dict['attention_heads'] = int(layer_params[2] * (8 - 1) + 1)  # [1, 8]
            layer_dict['activation'] = 'relu'  # Default
            
        elif layer_type == 'MaxPooling':
            stride_idx = int(layer_params[1] * 2)  # [0, 1]
            layer_dict['strides'] = stride_options.get(stride_idx, 1)
            
        elif layer_type == 'Dropout':
            rate_idx = int(layer_params[1] * 4)  # [0, 3]
            layer_dict['rate'] = dropout_options.get(rate_idx, 0.2)
            
        elif layer_type == 'Dense':
            layer_dict['units'] = int(layer_params[1] * (512 - 1) + 1)  # [1, 512]
            layer_dict['activation'] = 'relu'  # Default
        
        model_dict['layers'].append(layer_dict)
    
    return model_dict

def save_encoded_models_to_csv(num_models, filename, max_alleles=48):
    """
    Genera modelos usando Latin Hypercube Sampling y guarda sus codificaciones en un CSV.
    
    Args:
        num_models: Número de modelos a generar
        filename: Nombre del archivo CSV para guardar los resultados
        max_alleles: Número máximo de alelos por modelo
    """
    # Generar muestras
    samples = generate_latin_hypercube_samples(num_models, max_alleles)
    
    # Crear DataFrame para guardar los resultados
    columns = [f'allele_{i}' for i in range(max_alleles)]
    df = pd.DataFrame(samples, columns=columns)
    
    # Guardar en CSV
    df.to_csv(filename, index=False)
    print(f"Se han guardado {num_models} modelos en {filename}")