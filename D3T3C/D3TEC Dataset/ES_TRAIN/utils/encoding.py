# Encoding and decoding utilities for neural architecture search
import torch
import torch.nn as nn
import torch.nn.functional as F

def int_to_real_dom(num, domain):
    """Convert an integer value to a real value in [0,1] based on the given domain.
    
    Args:
        num: The integer value to convert
        domain: A tuple of (min, max) defining the integer domain
        
    Returns:
        A real value in [0,1] representing the normalized position in the domain
    """
    min_i, max_i = domain
    r = (num - min_i) / (max_i - min_i)
    return r

def real_to_int_dom(num, domain):
    """Convert a real value in [0,1] to an integer value based on the given domain.
    
    Args:
        num: The real value in [0,1] to convert
        domain: A tuple of (min, max) defining the integer domain
        
    Returns:
        An integer value within the specified domain
    """
    min_i, max_i = domain
    value = min_i + num * (max_i - min_i)
    if isinstance(min_i, int) and isinstance(max_i, int):
        value = int(round(value))
    return value

layer_type_options = {
    0: 'Conv2D',
    1: 'BatchNorm',
    2: 'MaxPooling',
    3: 'Dropout',
    4: 'Dense',
    5: 'Flatten',
    6: 'SelfAttention',  # Reemplazo de DepthwiseConv2D por Self-Attention
    7: 'DontCare',
    8: 'Repetition'
}

stride_options = {0: 1, 1: 2}
dropout_options = {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.5}
activation_options = {0: 'relu', 1: 'leaky_relu', 2: 'sigmoid', 3: 'tanh'}

def encode_layer_params(layer_type_idx, param1=0, param2=0, param3=0):
    """
    Codifica una capa en una lista en función del tipo de capa y sus parámetros.
    
    layer_type_idx : int : índice del tipo de capa según layer_type_options.
    param1         : int/float : filtros, neuronas, capas de repetición, etc.
    param2         : int : stride, número de repeticiones, etc.
    param3         : int : índice de activación o tasa de dropout.
    """
    return [layer_type_idx, param1, param2, param3]
def decode_layer_params(encoded_params):
    """
    Decodifica una capa desde su representación codificada en parámetros interpretables.
    
    encoded_params : list : [tipo de capa, param1, param2, param3].
    """
    layer_type_idx = encoded_params[0]
    layer_type = layer_type_options.get(layer_type_idx, 'DontCare')
    
    # Decodificar en función del tipo de capa
    if layer_type == 'Conv2D':
        filters = max(4, min(encoded_params[1], 32))  # Limitar filtros entre 4 y 32
        strides = stride_options.get(encoded_params[2], 1)
        activation = activation_options.get(encoded_params[3], 'relu')
        return {
            'type': 'Conv2D',
            'filters': filters,
            'strides': strides,
            'activation': activation
        }
    elif layer_type == 'BatchNorm':
        return {'type': 'BatchNorm'}
    elif layer_type == 'MaxPooling':
        strides = stride_options.get(encoded_params[1], 1)
        return {'type': 'MaxPooling', 'strides': strides}
    elif layer_type == 'Dropout':
        rate = dropout_options.get(encoded_params[1], 0.2)
        return {'type': 'Dropout', 'rate': rate}
    elif layer_type == 'Dense':
        units = max(1, min(encoded_params[1], 512))  # Limitar unidades entre 1 y 512
        activation = activation_options.get(encoded_params[2], 'relu')
        return {'type': 'Dense', 'units': units, 'activation': activation}
    elif layer_type == 'Flatten':
        return {'type': 'Flatten'}
    elif layer_type == 'Repetition':
        return {
            'type': 'Repetition',
            'repetition_layers': int(encoded_params[1]),
            'repetition_count': int(encoded_params[2])
        }
    elif layer_type == 'SelfAttention':
        filters = max(4, min(encoded_params[1], 64))  # Atención con 4-64 filtros
        attention_heads = max(1, min(encoded_params[2], 8))  # Máximo 8 cabezas de atención
        activation = activation_options.get(encoded_params[3], 'relu')  # Activación opcional
        return {
            'type': 'SelfAttention',
            'filters': filters,
            'attention_heads': attention_heads,
            'activation': activation
        }
    elif layer_type == 'DontCare':
        return {'type': "DontCare"}

    return None


def encode_model_architecture(model_dict, max_alleles=48):
    """
    Codifica la arquitectura del modelo en una lista de valores con un máximo de `max_alleles`.
    Cada capa se codifica en función de sus parámetros.
    """
    encoded_layers = []
    total_alleles = 0

    for layer in model_dict['layers']:
        if layer['type'] == 'Repetition':  # Codificar capa de repetición
            encoded_layer = encode_layer_params(
                layer_type_idx=8,  # índice para 'Repetition'
                param1=layer.get('repetition_layers', 0),
                param2=layer.get('repetition_count', 1)
            )
        else:
            layer_type_idx = next(
                key for key, value in layer_type_options.items() if value == layer['type']
            )
            
            # Codificar parámetros específicos de cada tipo de capa
            if layer['type'] == 'Conv2D':  
                param1 = max(4, min(layer.get('filters', 8), 32))  # Limitar filtros dentro del rango [4, 32]
                param2 = next((key for key, value in stride_options.items() if value == layer.get('strides', 1.0)), 0)
                param3 = next((key for key, value in activation_options.items() if value == layer.get('activation', 'relu')), 0)
                encoded_layer = [layer_type_idx, param1, param2, param3]

            elif layer['type'] == 'SelfAttention':  # Reemplazo de DepthwiseConv2D por SelfAttention
                param1 = max(4, min(layer.get('filters', 8), 64))  # Limitar filtros dentro del rango [4, 64]
                param2 = max(1, min(layer.get('attention_heads', 1), 8))  # Número de cabezas de atención [1, 8]
                param3 = next((key for key, value in activation_options.items() if value == layer.get('activation', 'relu')), 0)
                encoded_layer = [layer_type_idx, param1, param2, param3]

            elif layer['type'] == 'Dense':
                param1 = max(1, min(layer.get('units', 1), 512))  # Limitar neuronas dentro del rango [1, 512]
                param2 = next((key for key, value in activation_options.items() if value == layer.get('activation', 'relu')), 0)
                encoded_layer = [layer_type_idx, param1, param2, 0]

            elif layer['type'] == 'MaxPooling':
                param1 = next((key for key, value in stride_options.items() if value == layer.get('strides', 1.0)), 0)
                encoded_layer = [layer_type_idx, param1, 0, 0]

            elif layer['type'] == 'Dropout':
                param1 = next((key for key, value in dropout_options.items() if value == layer.get('rate', 0.2)), 0)
                encoded_layer = [layer_type_idx, param1, 0, 0]

            elif layer['type'] == 'BatchNorm':
                encoded_layer = [layer_type_idx, 0, 0, 0]

            elif layer['type'] == 'Flatten':
                encoded_layer = [layer_type_idx, 0, 0, 0]

            elif layer['type'] == 'DontCare':
                encoded_layer = [layer_type_idx, 0, 0, 0]

        # Añadir la codificación de la capa a la lista de alelos
        encoded_layers.extend(encoded_layer)
        total_alleles += len(encoded_layer)

    # Rellenar con 'DontCare' si el total de alelos es menor que `max_alleles`
    while total_alleles < max_alleles:
        dont_care_encoding = encode_layer_params(7)  # índice de 'DontCare'
        encoded_layers.extend(dont_care_encoding)
        total_alleles += len(dont_care_encoding)

    # Recortar si excede `max_alleles`
    final_encoding = encoded_layers[:max_alleles]
    print(f"Final Encoded Model: {final_encoding}")
    
    return final_encoding

def convert_individual(ind, to_real=True):
    real_rep = []
    N = max(layer_type_options.keys())  # Get maximum layer type index

    for i in range(0, len(ind), 4):
        layer_type_idx = ind[i]
        domain_layer_type = [0, N]

        if to_real:
            real_rep.append(int_to_real_dom(layer_type_idx, domain_layer_type))
            layer_type = layer_type_options.get(layer_type_idx, 'DontCare')
        else:
            real_rep.append(real_to_int_dom(layer_type_idx, domain_layer_type))
            layer_type = layer_type_options.get(real_rep[i], 'DontCare')

        # Decode based on layer type
        if layer_type == 'Conv2D':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [4, 32]))  # Filters
                real_rep.append(int_to_real_dom(ind[i + 2], [0, 1]))  # Stride
                real_rep.append(int_to_real_dom(ind[i + 3], [0, 3]))  # Activation
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [4, 32]))
                real_rep.append(real_to_int_dom(ind[i + 2], [0, 1]))
                real_rep.append(real_to_int_dom(ind[i + 3], [0, 3]))

        elif layer_type == 'SelfAttention':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [4, 64]))  # Filters
                real_rep.append(int_to_real_dom(ind[i + 2], [1, 8]))  # Attention heads
                real_rep.append(int_to_real_dom(ind[i + 3], [0, 3]))  # Activation
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [4, 64]))
                real_rep.append(real_to_int_dom(ind[i + 2], [1, 8]))
                real_rep.append(real_to_int_dom(ind[i + 3], [0, 3]))

        elif layer_type == 'BatchNorm':
            real_rep.extend([0, 0, 0])

        elif layer_type == 'MaxPooling':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [0, 1]))
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [0, 1]))
            real_rep.extend([0, 0])

        elif layer_type == 'Dropout':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [0, 3]))
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [0, 3]))
            real_rep.extend([0, 0])

        elif layer_type == 'Dense':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [1, 512]))  # Neurons
                real_rep.append(int_to_real_dom(ind[i + 2], [0, 3]))  # Activation
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [1, 512]))
                real_rep.append(real_to_int_dom(ind[i + 2], [0, 3]))
            real_rep.append(0)

        elif layer_type == 'Flatten':
            real_rep.extend([0, 0, 0])

        elif layer_type == 'Repetition':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [1, 4]))  # Layers to repeat
                real_rep.append(int_to_real_dom(ind[i + 2], [1, 32]))  # Repetition count
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [1, 4]))
                real_rep.append(real_to_int_dom(ind[i + 2], [1, 32]))
            real_rep.append(0)

        elif layer_type == 'DontCare':
            real_rep.extend([0, 0, 0])

    return real_rep

def fixArch(encoded_model, verbose=False):
    """
    Corrige la arquitectura codificada del modelo, asegurando que:
      - No se permitan capas incompatibles después de una capa Dense.
      - Se ajuste el alcance de repetición en caso de una capa Repetition.
      - Solo se permita una única capa SelfAttention, y se ajusten sus parámetros para que los filtros sean divisibles por attention_heads.

    Parameters:
      encoded_model (list): Lista codificada de la arquitectura del modelo.
      verbose (bool): Si es True, muestra mensajes de corrección.

    Returns:
      list: Lista con la arquitectura corregida, truncada a un máximo de 48 alelos.
    """

    fixed_layers = []         # Lista que almacenará la arquitectura corregida
    input_is_flattened = False # Indicador de que ya se aplicó una capa Flatten
    dense_started = False      # Indicador de que ya se encontró una capa Dense
    index = 0                  # Índice para recorrer el encoding
    found_self_attention = False  # Flag para asegurar que solo se use una SelfAttention

    while index < len(encoded_model) and len(fixed_layers) < 48:
        layer_type = int(encoded_model[index])  # Tipo de capa actual

        # Si ya se ha encontrado una Dense, solo se permiten Dense, BatchNorm o DontCare
        if dense_started and layer_type not in [4, 1, 7]:
            if verbose:
                print(f"Se encontró una capa de tipo {layer_type} después de una Dense, reemplazando con DontCare")
            fixed_layers.extend([7, 0, 0, 0])
            index += 4
            continue

        # Procesar capa de Repetition
        if layer_type == 8:
            repetition_layers = int(encoded_model[index + 1])
            repetition_count = min(max(int(encoded_model[index + 2]), 0), 32)
            actual_layers_to_repeat = min(repetition_layers, len(fixed_layers) // 4)
            if actual_layers_to_repeat != repetition_layers:
                if verbose:
                    print(f"Ajustando alcance de repetición de {repetition_layers} a {actual_layers_to_repeat} debido a falta de capas.")
                repetition_layers = actual_layers_to_repeat
            fixed_layers.extend([layer_type, repetition_layers, repetition_count, 0])
            index += 4
            continue

        # Procesar capas según su tipo
        if layer_type == 0:  # Conv2D
            if input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])  # Reemplazar con DontCare
            else:
                filters = min(max(int(encoded_model[index + 1]), 4), 32)
                stride_idx = min(max(int(encoded_model[index + 2]), 0), 1)
                activation_idx = min(max(int(encoded_model[index + 3]), 0), 3)
                fixed_layers.extend([layer_type, filters, stride_idx, activation_idx])

        elif layer_type == 6:  # SelfAttention
            if input_is_flattened or found_self_attention:
                fixed_layers.extend([7, 0, 0, 0])
                if verbose and found_self_attention:
                    print("Capa SelfAttention adicional reemplazada con DontCare.")
            else:
                filters = min(max(int(encoded_model[index + 1]), 4), 64)
                attention_heads = min(max(int(encoded_model[index + 2]), 1), 4)
                activation_idx = min(max(int(encoded_model[index + 3]), 0), 3)
                # Ajuste: Asegurar que 'filters' sea divisible por 'attention_heads'
                if filters % attention_heads != 0:
                    nuevo_valor = filters - (filters % attention_heads)
                    if verbose:
                        print(f"Warning: SelfAttention filters {filters} no es divisible por attention_heads {attention_heads}. Ajustando filters a {nuevo_valor}")
                    filters = nuevo_valor if nuevo_valor >= 4 else 4
                fixed_layers.extend([layer_type, filters, attention_heads, activation_idx])
                found_self_attention = True

        elif layer_type == 2:  # MaxPooling
            if input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])
            else:
                stride_idx = min(max(int(encoded_model[index + 1]), 0), 1)
                fixed_layers.extend([layer_type, stride_idx, 0, 0])

        elif layer_type == 3:  # Dropout
            rate_idx = min(max(int(encoded_model[index + 1]), 0), 3)
            fixed_layers.extend([layer_type, rate_idx, 0, 0])

        elif layer_type == 4:  # Dense
            neurons = min(max(int(encoded_model[index + 1]), 1), 512)
            activation_idx = min(max(int(encoded_model[index + 2]), 0), 3)
            fixed_layers.extend([layer_type, neurons, activation_idx, 0])
            dense_started = True

        elif layer_type == 1:  # BatchNorm
            if len(fixed_layers) > 0:
                prev_layer = fixed_layers[-4:]
                prev_layer_type = prev_layer[0]
                if prev_layer_type in [0, 6]:
                    num_features = prev_layer[1]
                else:
                    num_features = 4
            else:
                num_features = 4
            if verbose:
                print(f"Configurando BatchNorm con {num_features} canales")
            fixed_layers.extend([layer_type, num_features, 0, 0])

        elif layer_type == 5:  # Flatten
            if input_is_flattened or dense_started:
                fixed_layers.extend([7, 0, 0, 0])
            else:
                if index + 4 < len(encoded_model):
                    next_layer_type = int(encoded_model[index + 4])
                    if next_layer_type not in [4, 7]:  # Flatten debería ir seguido de Dense o DontCare
                        if verbose:
                            print(f"Warning: Flatten seguido de {next_layer_type}, reemplazando con DontCare")
                        fixed_layers.extend([7, 0, 0, 0])
                    else:
                        fixed_layers.extend([layer_type, 0, 0, 0])
                        input_is_flattened = True
                else:
                    fixed_layers.extend([layer_type, 0, 0, 0])
                    input_is_flattened = True

        elif layer_type == 7:  # DontCare
            fixed_layers.extend([layer_type, 0, 0, 0])

        else:  # Tipo desconocido
            fixed_layers.extend([7, 0, 0, 0])

        index += 4

    return fixed_layers[:48]


def es(target_func, mu=10, lamb=1, F=2, rp=0.5, gens=100, n=10, auto_adapt=False):
    # Initialize parent population
    pop = pop_gen(mu)
    succ_m_count = 0
    best_fitness_per_gen = []
    Fs = []

    # Progress bar for generations
    with tqdm(total=gens, desc="Generations", leave=False) as pbar_gens:
        for gen in range(gens):
            children = []

            # Calculate fitness and select best element
            for parent in pop:
                parent['fitness'] = target_func(parent['individual'])
            best_parent = max(pop, key=lambda x: x['fitness'])

            for i in range(lamb):
                parent = pop[i]['individual']

                # Map individuals to real values
                parent_unit = convert_individual(parent)
                best_parent_c = convert_individual(best_parent['individual'])

                # Mutation and repair mechanism
                x2, x3 = random.sample(pop, 2)
                x2 = convert_individual(x2['individual'])
                x3 = convert_individual(x3['individual'])
                u = []
                for j in range(len(best_parent_c)):
                    u.append(best_parent_c[j] + F * (x2[j] - x3[j]))
                u = fixArch(convert_individual(u, to_real=False))
                u = convert_individual(u)

                # Recombination
                cr_points = get_cr_points(rp, len(parent))
                child = [u[j] if j in cr_points else parent[j] for j in range(len(u))]
                child = fixArch(convert_individual(child, to_real=False))

                children.append({'individual': child, 'fitness': 0})

            # Calculate fitness of children
            children = [{'individual': child['individual'], 'fitness': target_func(child['individual'])} for child in children]

            # Select the best mu elements for next generation
            complete_pop = pop + children

            if auto_adapt:
                # Count successful mutations
                succ_m_count += get_succ_m(pop, children)

                # During each time window (n) self-adapt F
                if (gen + 1) % n == 0:
                    Fs.append(F)
                    ps = succ_m_count / (n * lamb)

                    if ps > 1/5:
                        F /= 0.817
                    elif ps < 1/5:
                        F *= 0.817

                    succ_m_count = 0

            pop = sorted(complete_pop, key=lambda x: x['fitness'], reverse=True)[:mu]
            best_fitness_per_gen.append(max(pop, key=lambda x: x['fitness'])['fitness'])

            # Update generation progress bar
            pbar_gens.update(1)

    best_element = max(pop, key=lambda x: x['fitness'])
    Fs.append(F)
    return best_element, best_fitness_per_gen, Fs

def decode_model_architecture(encoded_model):
    """
    Decodifica la arquitectura del modelo a partir de la lista codificada de valores (índices),
    aplicando las reglas de repetición y asegurando la inclusión de una capa convolucional inicial.
    """
    model_dict = {'layers': [{'type': 'Conv2D', 'filters': 32, 'strides': 1, 'activation': 'relu'}]} 
    index = 0
    found_self_attention = False  # Flag para asegurar una sola SelfAttention

    while index < len(encoded_model):
        layer_type = int(encoded_model[index])
        param1 = encoded_model[index + 1]
        param2 = encoded_model[index + 2]
        param3 = encoded_model[index + 3]

        if layer_type == 8:  # Capa de Repetition
            repetition_layers = int(param1)
            repetition_count = int(param2)

            # Selecciona solo capas válidas para la repetición (sin incluir SelfAttention)
            layers_to_repeat = select_group_for_repetition(model_dict['layers'], repetition_layers)

            if len(layers_to_repeat) > 0:
                for _ in range(repetition_count):
                    for layer in layers_to_repeat:
                        # Si la capa es SelfAttention, reemplazarla con DontCare
                        if layer['type'] == 'SelfAttention':
                            model_dict['layers'].append({'type': 'DontCare'})
                        else:
                            model_dict['layers'].append(layer)

        else:
            decoded_layer = {}

            if layer_type == 0:  # Conv2D
                decoded_layer = {
                    'type': 'Conv2D',
                    'filters': max(4, min(param1, 32)),  # Limita `filters` entre 4 y 32
                    'strides': stride_options.get(param2, 1),
                    'activation': activation_options.get(param3, 'relu')
                }
            elif layer_type == 6:  # SelfAttention
                if found_self_attention:  # Si ya hay una SelfAttention, la ignoramos
                    index += 4
                    continue
                decoded_layer = {
                    'type': 'SelfAttention',
                    'filters': max(4, min(param1, 64)),  # Limita `filters` entre 4 y 64
                    'attention_heads': max(1, min(param2, 4)),  # Limita `attention_heads` entre 1 y 4
                    'activation': activation_options.get(param3, 'relu')
                }
                found_self_attention = True  # Marca que ya se agregó una SelfAttention
            elif layer_type == 2:  # MaxPooling
                decoded_layer = {
                    'type': 'MaxPooling',
                    'strides': stride_options.get(param1, 1)
                }
            elif layer_type == 3:  # Dropout
                decoded_layer = {
                    'type': 'Dropout',
                    'rate': dropout_options.get(param1, 0.2)
                }
            elif layer_type == 4:  # Dense
                decoded_layer = {
                    'type': 'Dense',
                    'units': max(1, min(param1, 512)),  # Limita `units` entre 1 y 512
                    'activation': activation_options.get(param2, 'relu')
                }
            elif layer_type == 1:  # BatchNorm
                decoded_layer = {'type': 'BatchNorm'}
            elif layer_type == 5:  # Flatten
                decoded_layer = {'type': 'Flatten'}
            elif layer_type == 7:  # DontCare
                decoded_layer = {'type': 'DontCare'}

            model_dict['layers'].append(decoded_layer)

        index += 4

    # Asegura que haya una capa Flatten antes de la capa Dense final, si no ya existe una Flatten
    if model_dict['layers'][-1]['type'] != 'Flatten':
        model_dict['layers'].append({'type': 'Flatten'})

    # Añade la capa Dense final obligatoria
    model_dict['layers'].append({'type': 'Dense', 'units': 1, 'activation': 'sigmoid'})

    return model_dict

def select_group_for_repetition(layers, repetition_layers):
    """
    Selecciona el primer grupo válido para repetición en función de las reglas de compatibilidad,
    evitando la duplicación de SelfAttention.

    Parameters:
        layers (list): Lista de capas ya procesadas, donde cada capa es un diccionario.
        repetition_layers (int): Número de capas hacia atrás para considerar en la repetición.

    Returns:
        list: Lista de capas compatibles para repetición, sin SelfAttention.
    """
    valid_layers = []
    group_type = None

    # Retrocede desde el final de `layers` para encontrar el grupo válido
    for layer in reversed(layers[-repetition_layers:]):
        if group_type is None:
            # Determina el tipo de grupo
            if layer['type'] in ['Flatten', 'Dense']:
                group_type = 'dense'
                valid_layers.insert(0, layer)
            elif layer['type'] in ['Conv2D', 'SelfAttention', 'MaxPooling']:
                group_type = 'convolutional'
                valid_layers.insert(0, layer)
            elif layer['type'] in ['BatchNorm', 'DontCare']:  # BatchNorm y DontCare son compatibles con ambos grupos
                valid_layers.insert(0, layer)
        else:
            # Agrega solo capas compatibles con el grupo seleccionado
            if group_type == 'dense' and layer['type'] in ['Flatten', 'Dense', 'BatchNorm', 'DontCare']:
                valid_layers.insert(0, layer)
            elif group_type == 'convolutional' and layer['type'] in ['Conv2D', 'SelfAttention', 'MaxPooling', 'BatchNorm', 'DontCare']:
                valid_layers.insert(0, layer)

    return valid_layers

class BuildPyTorchModel(nn.Module):
    def __init__(self, model_dict, input_shape=(1, 128, 128), verbose=False):
        """
        Construye un modelo de PyTorch a partir de un diccionario de arquitectura.
        """
        super(BuildPyTorchModel, self).__init__()
        self.verbose = verbose
        self.input_shape = input_shape
        model_dict = decode_model_architecture(model_dict)
        
        target_in_channels = 4  # Número mínimo de canales requeridos en la arquitectura
        layers = []
        if input_shape[0] != target_in_channels:
            if self.verbose:
                print(f"📌 Insertando capa de conversión: de {input_shape[0]} canal(es) a {target_in_channels} canales.")
            self.initial_conv = nn.Conv2d(in_channels=input_shape[0],
                                          out_channels=target_in_channels,
                                          kernel_size=1)
            in_channels = target_in_channels
        else:
            self.initial_conv = None
            in_channels = input_shape[0]

        self.linear_layers = []

        for layer in model_dict['layers']:
            if layer['type'] == 'Conv2D':
                layers.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=layer['filters'],
                                        kernel_size=3,
                                        stride=layer['strides'],
                                        padding=1))
                layers.append(nn.ReLU() if layer['activation'] == "relu" else nn.LeakyReLU())
                in_channels = layer['filters']
            elif layer['type'] == 'SelfAttention':
                # Antes de la atención local, se verifica que los canales coincidan
                desired_filters = layer['filters']
                if in_channels != desired_filters:
                    if self.verbose:
                        print(f"Ajustando canales de entrada: {in_channels} -> {desired_filters}")
                    layers.append(nn.Conv2d(in_channels, desired_filters, kernel_size=1))
                    in_channels = desired_filters
                # Se agrega la capa de atención local
                layers.append(SelfAttention(filters=desired_filters,
                                                 window_size=16,  # Ajusta según la resolución (por ejemplo, 16 para 128x128)
                                                 attention_heads=layer['attention_heads'],
                                                 activation=layer['activation'],
                                                 verbose=self.verbose))
                in_channels = desired_filters
            elif layer['type'] == 'BatchNorm':
                layers.append(nn.BatchNorm2d(in_channels))
            elif layer['type'] == 'MaxPooling':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=layer['strides'], padding=1))
            elif layer['type'] == 'Flatten':
                layers.append(nn.Flatten())
            elif layer['type'] == 'Dense':
                self.linear_layers.append((layer['units'], layer['activation']))
            elif layer['type'] == 'Dropout':
                layers.append(nn.Dropout(p=layer['rate']))
            elif layer['type'] == 'DontCare':
                layers.append(DontCareLayer())

        self.feature_extractor = nn.Sequential(*layers)

    def forward_features(self, x):
        for module in self.feature_extractor:
            x = module(x)
        return x

    def forward(self, x):
        """
        Propagación hacia adelante en el modelo.
        """
        # Aplicar capa de conversión inicial si es necesaria
        if hasattr(self, 'initial_conv'):
            x = self.initial_conv(x)
        for i, module in enumerate(self.feature_extractor):
            # Ajuste dinámico de BatchNorm (según si la entrada es 2D o 4D)
            if isinstance(module, nn.BatchNorm2d):
                if x.dim() == 2:  # (batch, features)
                    num_features = x.shape[1]
                    self.feature_extractor[i] = nn.BatchNorm1d(num_features).to(x.device)
                    module = self.feature_extractor[i]
                else:
                    num_channels = x.shape[1]
                    if module.num_features != num_channels:
                        self.feature_extractor[i] = nn.BatchNorm2d(num_channels).to(x.device)
                        module = self.feature_extractor[i]
            x = module(x)
        # Construcción dinámica de las capas densas
        if not hasattr(self, "fully_connected"):
            in_features = x.shape[1]
            fc_layers = []
            for units, activation in self.linear_layers:
                fc_layers.append(nn.Linear(in_features, units))
                fc_layers.append(nn.ReLU() if activation == "relu" else nn.LeakyReLU())
                in_features = units
            self.fully_connected = nn.Sequential(*fc_layers).to(x.device)
        x = self.fully_connected(x)
        return x


class SelfAttention(nn.Module):
    """
    Implementación de capa de auto-atención para redes neuronales convolucionales.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Proyecciones para query, key, value
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # Calcular matriz de atención
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Calcular salida
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Aplicar residual connection con peso gamma
        out = self.gamma * out + x
        return out

class DontCareLayer(nn.Module):
    """
    Capa que no hace nada, utilizada para representar espacios vacíos en la arquitectura.
    """
    def __init__(self):
        super(DontCareLayer, self).__init__()
        
    def forward(self, x):
        return x