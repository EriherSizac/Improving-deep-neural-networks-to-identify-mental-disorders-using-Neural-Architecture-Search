

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
from pyDOE2 import lhs
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
#sys.stdout = open('/dev/null', 'w')  # Redirigir la salida a /dev/null
import os
import pandas as pd
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader,Dataset
import json
import torch.multiprocessing as mp
#import torchaudio.transforms as T

#torch.cuda.memory_summary()
    
# 🔹 Optimización de cuDNN
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
#torch.set_num_threads(1)  # Prueba con 4, 2 o 1
#torch.set_num_interop_threads(1)

#import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

# Opciones de decodificación para otros parámetros
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

# Función para codificar los parámetros de la capa
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




def fixArch(encoded_model, verbose=False):
    """
    Corrige la arquitectura codificada del modelo, asegurando que:
    - Se evite la presencia de capas incompatibles después de una capa Flatten.
    - En caso de una capa de Repetition, se ajuste el alcance de repetición si no hay suficientes capas anteriores.
    - Limita la arquitectura a una sola capa de SelfAttention.
    
    Parameters:
        encoded_model (list): Lista codificada de la arquitectura del modelo.
        verbose (bool): Si es True, muestra las correcciones realizadas.

    Returns:
        list: Lista con la arquitectura corregida, truncada a un máximo de 48 alelos.
    """

    fixed_layers = []  # Lista que almacenará la arquitectura corregida
    input_is_flattened = False  # Indicador para saber si ya hay una capa Flatten en el modelo
    index = 0  # Índice para recorrer el modelo codificado
    found_self_attention = False  # Flag para rastrear la primera aparición de SelfAttention

    # Procesar cada capa en el modelo sin forzar la primera capa a ser específica
    while index < len(encoded_model) and len(fixed_layers) < 48:
        layer_type = int(encoded_model[index])  # Obtener el tipo de capa actual

        # Procesar la capa de Repetition
        if layer_type == 8:
            repetition_layers = int(encoded_model[index + 1])  # Número de capas a repetir
            repetition_count = min(max(int(encoded_model[index + 2]), 0), 32)  # Cantidad de repeticiones

            # Verificar si hay suficientes capas para la repetición solicitada
            actual_layers_to_repeat = min(repetition_layers, len(fixed_layers) // 4)

            if actual_layers_to_repeat != repetition_layers:
                if verbose:
                    print(f"Ajustando alcance de repetición de {repetition_layers} a {actual_layers_to_repeat} debido a falta de capas.")
                repetition_layers = actual_layers_to_repeat

            # Añadir la capa de repetición sin modificar su estructura
            fixed_layers.extend([layer_type, repetition_layers, repetition_count, 0])
            index += 4
            continue

        # Procesar cada tipo de capa normal con sus restricciones
        if layer_type == 0:  # Conv2D
            if input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])  # DontCare
            else:
                # Limitar el número de filtros entre 4 y 32
                filters = min(max(int(encoded_model[index + 1]), 4), 32)
                stride_idx = min(max(int(encoded_model[index + 2]), 0), 1)
                activation_idx = min(max(int(encoded_model[index + 3]), 0), 3)
                fixed_layers.extend([layer_type, filters, stride_idx, activation_idx])

        elif layer_type == 6:  # SelfAttention
            if input_is_flattened or found_self_attention:
                fixed_layers.extend([7, 0, 0, 0])  # Reemplazar SelfAttention extra con DontCare
                if verbose and found_self_attention:
                    print("Capa SelfAttention adicional reemplazada con DontCare.")
            else:
                # Añadir la primera capa SelfAttention
                filters = min(max(int(encoded_model[index + 1]), 4), 64)  # Limitar filtros [4, 64]
                attention_heads = min(max(int(encoded_model[index + 2]), 1), 4)  # Limitar cabezas [1, 4]
                activation_idx = min(max(int(encoded_model[index + 3]), 0), 3)
                fixed_layers.extend([layer_type, filters, attention_heads, activation_idx])
                found_self_attention = True  # Marcar que ya se añadió una SelfAttention

        elif layer_type == 2:  # MaxPooling
            if input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])  # DontCare
            else:
                stride_idx = min(max(int(encoded_model[index + 1]), 0), 1)
                fixed_layers.extend([layer_type, stride_idx, 0, 0])

        elif layer_type == 3:  # Dropout
            rate_idx = min(max(int(encoded_model[index + 1]), 0), 3)
            fixed_layers.extend([layer_type, rate_idx, 0, 0])

        elif layer_type == 4:  # Dense
            # Limitar el número de neuronas entre 1 y 512
            neurons = min(max(int(encoded_model[index + 1]), 1), 512)
            activation_idx = min(max(int(encoded_model[index + 2]), 0), 3)
            fixed_layers.extend([layer_type, neurons, activation_idx, 0])

        elif layer_type == 1:  # BatchNorm
    # 📌 Asegurar que el número de canales (C) sea el actual en la arquitectura
            if len(fixed_layers) > 0:
                prev_layer = fixed_layers[-4:]  # Última capa agregada
                prev_layer_type = prev_layer[0]  # Tipo de capa anterior
                
                # Obtener número de canales del output de la última capa convolucional o SelfAttention
                if prev_layer_type in [0, 6]:  # Conv2D o SelfAttention
                    num_features = prev_layer[1]  # Número de filtros de la última capa
                
                else:
                    num_features = 4  # Default si no hay capas anteriores relevantes
                
            else:
                num_features = 4  # Si BatchNorm es la primera capa, asignar 4 por defecto

            """ print(f"📌 Configurando BatchNorm con {num_features} canales") """
            fixed_layers.extend([layer_type, num_features, 0, 0])  # Guardar num_features


        elif layer_type == 5:  # Flatten
            if input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])  # Reemplazar Flatten adicional con DontCare
            else:
                # Verificar que la siguiente capa sea una capa densa
                if index + 4 < len(encoded_model):
                    next_layer_type = int(encoded_model[index + 4])     
                    if next_layer_type not in [4, 7]:  # Solo debe ir antes de Dense o DontCare
                        """ print(f"⚠️ WARNING: Flatten seguido de {next_layer_type}, reemplazando con DontCare") """
                        fixed_layers.extend([7, 0, 0, 0])
                    else:
                        fixed_layers.extend([layer_type, 0, 0, 0])
                        input_is_flattened = True  # Marcar que ya hay un Flatten
                else:
                    fixed_layers.extend([layer_type, 0, 0, 0])
                    input_is_flattened = True  # Marcar que ya hay un Flatten


        elif layer_type == 7:  # DontCare
            fixed_layers.extend([layer_type, 0, 0, 0])

        else:  # Cualquier otro tipo de capa desconocida
            fixed_layers.extend([7, 0, 0, 0])  # Reemplazar con DontCare

        index += 4  # Avanzar al siguiente grupo de parámetros

    return fixed_layers[:48]  # Limitar a 48 alelos


def decode_model_architecture(encoded_model):
    """
    Decodifica la arquitectura del modelo a partir de la lista codificada de valores (índices),
    aplicando las reglas de repetición y asegurando la inclusión de una capa convolucional inicial.
    """
    model_dict = {'layers': []}  # Lista de capas decodificadas
    index = 0
    found_self_attention = False  # Flag para asegurar una sola SelfAttention
    print("Encoded model:", encoded_model)
    while index < len(encoded_model):
        print(f"Current index: {index}, Current value: {encoded_model}, len: {len(encoded_model)}")
        print(f"encoded model: {encoded_model}")
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
    def __init__(self, model_dict, input_shape=(1, 64, 552), verbose=False):
        """
        Construye un modelo de PyTorch a partir de un diccionario de arquitectura.
        """
        super(BuildPyTorchModel, self).__init__()
        self.verbose = verbose
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
                layers.append(SelfAttention(filters=in_channels,
                                            attention_heads=layer['attention_heads'],
                                            activation=layer['activation'],
                                            verbose=self.verbose))
            elif layer['type'] == 'BatchNorm':
                # Se inicia con BatchNorm2d, pero se ajustará en forward si es necesario.
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

    def forward(self, x):
        if self.initial_conv is not None:
            x = self.initial_conv(x)

        for i, module in enumerate(self.feature_extractor):
            # Si el módulo es BatchNorm2d pero la entrada es 2D (después del Flatten)
            if isinstance(module, nn.BatchNorm2d):
                if x.dim() == 2:  # Es decir, (batch, features)
                    num_features = x.shape[1]
                    print(f"⚠️ Reemplazando BatchNorm2d por BatchNorm1d para entrada con forma {x.shape}")
                    # Reemplazar la capa por una BatchNorm1d con el número correcto de features
                    self.feature_extractor[i] = nn.BatchNorm1d(num_features).to(x.device)
                    module = self.feature_extractor[i]
                else:
                    # En caso de entrada 4D, se verifica que el número de canales coincida
                    num_channels = x.shape[1]
                    if module.num_features != num_channels:
                        print(f"⚠️ Ajustando BatchNorm2d: esperaba {module.num_features} canales, pero recibió {num_channels}")
                        self.feature_extractor[i] = nn.BatchNorm2d(num_channels).to(x.device)
                        module = self.feature_extractor[i]
            x = module(x)

        # Construcción dinámica de capas densas
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


# Capa de identidad (DontCareLayer)
class DontCareLayer(nn.Module):
    def __init__(self):
        super(DontCareLayer, self).__init__()

    def forward(self, x):
        return x



class SelfAttention(nn.Module):
    def __init__(self, filters, window_size=8, attention_heads=4, activation=nn.ReLU(), verbose=False):
        super(SelfAttention, self).__init__()
        if filters % attention_heads != 0:
            if verbose:
                print(f"Warning: {filters} no es divisible por {attention_heads}. Ajustando filters.")
            filters = filters - (filters % attention_heads)
            filters = max(filters, attention_heads)

        self.filters = filters
        self.attention_heads = attention_heads
        self.window_size = window_size
        self.verbose = verbose
        self.d_head = self.filters // self.attention_heads  # Canales por cabeza
        
        # Inicializar convoluciones como None para configurarlas en forward()
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.projection_conv = None

    def forward(self, x):
        B, C, H, W = x.shape
        ws = min(self.window_size, H, W)  # Asegurar que `window_size` no sea mayor que H o W
        
        # Padding si H o W no son múltiplos de window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]
        
        num_windows_h = H // ws
        num_windows_w = W // ws
        x_windows = x.view(B, C, num_windows_h, ws, num_windows_w, ws)
        x_windows = x_windows.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x_windows.view(-1, C, ws, ws)

        # Ajustar convoluciones dinámicamente
        if self.query_conv is None or self.query_conv.in_channels != C:
            self.query_conv = nn.Conv2d(in_channels=C, out_channels=self.filters, kernel_size=1).to(x.device)
            self.key_conv = nn.Conv2d(in_channels=C, out_channels=self.filters, kernel_size=1).to(x.device)
            self.value_conv = nn.Conv2d(in_channels=C, out_channels=self.filters, kernel_size=1).to(x.device)

        Q = self.query_conv(windows)
        K = self.key_conv(windows)
        V = self.value_conv(windows)

        B_w, C_w, H_w, W_w = Q.shape
        N = H_w * W_w
        Q = Q.view(B_w, self.attention_heads, self.d_head, N).permute(0, 1, 3, 2)
        K = K.view(B_w, self.attention_heads, self.d_head, N).permute(0, 1, 3, 2)
        V = V.view(B_w, self.attention_heads, self.d_head, N).permute(0, 1, 3, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out_window = torch.matmul(attn, V)

        out_window = out_window.permute(0, 1, 3, 2).contiguous()

        # Validación antes de `view()`
        expected_elements_out = B * C * num_windows_h * ws * num_windows_w * ws
        actual_elements_out = out_window.numel()

        if expected_elements_out != actual_elements_out:
            print(f"⚠️ ERROR: Tamaño incompatible en `view()`")
            print(f"Esperado: {expected_elements_out}, Real: {actual_elements_out}")
            print(f"Forma de `out_window` antes de `view()`: {out_window.shape}")
            
            # Ajuste seguro
            out = out_window.reshape(B, C, -1, num_windows_w * ws)
        else:
            out = out_window.view(B, C, num_windows_h * ws, num_windows_w * ws)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H - pad_h, :W - pad_w]
        return out

