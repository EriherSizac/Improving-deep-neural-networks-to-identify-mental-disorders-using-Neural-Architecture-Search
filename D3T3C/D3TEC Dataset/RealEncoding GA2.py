
# %%
# %% [markdown]
# # ERpncoding

# %%
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
import torchaudio.transforms as T

torch.cuda.memory_summary()


    
    # 🔹 Optimización de cuDNN
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
#torch.set_num_threads(1)  # Prueba con 4, 2 o 1
#torch.set_num_interop_threads(1)

#import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

# %%
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




# %% [markdown]
# ## Complete archs
# 

# %%



class SelfAttention(nn.Module):
    def __init__(self, filters, attention_heads=4, activation=nn.ReLU(), verbose=False):
        super(SelfAttention, self).__init__()
        self.filters = max(4, filters)  # Mínimo 4 filtros
        self.attention_heads = min(max(1, attention_heads), 4)  # Limitar entre 1 y 4 cabezas
        self.activation = activation
        self.verbose = verbose

        # Capas convolucionales para generar Q, K y V
        self.query_conv = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=1)

        # Proyección final para ajustar canales
        self.projection_conv = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        if self.verbose:
            print(f"📌 SelfAttention - Input Shape: {x.shape}")

        if channels < self.filters:
            x = F.pad(x, (0, 0, 0, 0, 0, self.filters - channels))
            if self.verbose:
                print(f"📌 SelfAttention - After padding: {x.shape}")

        # Calcular Q, K y V
        query = self.query_conv(x)
        key   = self.key_conv(x)
        value = self.value_conv(x)

        if self.verbose:
            print(f"📌 SelfAttention - Query shape (original): {query.shape}")
            print(f"📌 SelfAttention - Key shape (original):   {key.shape}")
            print(f"📌 SelfAttention - Value shape (original): {value.shape}")

        # Se eliminan los reshape para conservar la forma original.
        if self.verbose:
            print("📌 SelfAttention - No se aplican cambios de forma (reshape/view) para depuración.")
        # Se retorna el input original para depurar; en una versión final se aplicarían operaciones de atención.
        return x




# Capa de identidad (DontCareLayer)
class DontCareLayer(nn.Module):
    def __init__(self):
        super(DontCareLayer, self).__init__()

    def forward(self, x):
        return x

# %%
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


# %%


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

            print(f"📌 Configurando BatchNorm con {num_features} canales")
            fixed_layers.extend([layer_type, num_features, 0, 0])  # Guardar num_features


        elif layer_type == 5:  # Flatten
            if input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])  # Reemplazar Flatten adicional con DontCare
            else:
                # Verificar que la siguiente capa sea una capa densa
                if index + 4 < len(encoded_model):
                    next_layer_type = int(encoded_model[index + 4])
                    if next_layer_type not in [4, 7]:  # Solo debe ir antes de Dense o DontCare
                        print(f"⚠️ WARNING: Flatten seguido de {next_layer_type}, reemplazando con DontCare")
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


# %%
def decode_model_architecture(encoded_model):
    """
    Decodifica la arquitectura del modelo a partir de la lista codificada de valores (índices),
    aplicando las reglas de repetición y asegurando la inclusión de una capa convolucional inicial.
    """
    model_dict = {'layers': []}  # Lista de capas decodificadas
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








# %% [markdown]
# # Testing random generated architectures
# 

# %% [markdown]
# 

# %%
import csv
import numpy as np
from pyDOE2 import lhs

# Función para generar un hipercubo latino con rangos normalizados [0, 1]
def generate_latin_hypercube_samples(num_samples, dimensions):
    return lhs(dimensions, samples=num_samples)

# Validar si los parámetros generados están dentro del rango esperado
def validate_latin_hypercube(num_models=100):
    dimensions = 12 * 3  # 12 capas, 3 parámetros por capa
    latin_samples = generate_latin_hypercube_samples(num_models, dimensions)
    
    # Validar cada muestra generada
    for sample_idx, sample in enumerate(latin_samples):
        reshaped_sample = sample.reshape(12, 3)  # Cada modelo tiene 12 capas
        
        for layer_idx, layer_params in enumerate(reshaped_sample):
            # Validar parámetros individuales
            type_idx = int(layer_params[0] * 9)  # 9 tipos de capas
            param1 = layer_params[1]
            param2 = layer_params[2]

            # Verificar tipo de capa
            if type_idx not in range(9):
                print(f"ERROR en Modelo {sample_idx + 1}, Capa {layer_idx + 1}: Tipo inválido {type_idx}")
                return False

            # Verificar rangos específicos según el tipo de capa
            layer_mapping = ['Conv2D', 'DepthwiseConv2D', 'BatchNorm', 'MaxPooling', 
                             'Dropout', 'Dense', 'Flatten', 'DontCare', 'Repetition']
            layer_type = layer_mapping[type_idx]

            if layer_type in ['Conv2D', 'DepthwiseConv2D']:
                filters = int(param1 * (32 - 4) + 4)  # Filtros entre [4, 32]
                if not (4 <= filters <= 32):
                    print(f"ERROR en Modelo {sample_idx + 1}, Capa {layer_idx + 1}: Filtros fuera de rango {filters}")
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

    print("Validación completada: Todas las muestras están dentro de los rangos esperados.")
    return True

# Guardar el encoding generado en un archivo CSV
def save_encoded_models_to_csv(num_models, filename, max_alleles=48):
    # Generar muestras de hipercubo latino
    latin_samples = generate_latin_hypercube_samples(num_models, 12 * 3)  # 12 capas, 3 parámetros por capa

    # Crear el archivo CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir encabezados
        writer.writerow(["Model", "Encoded Chromosome"])

        for model_idx in range(num_models):
            # Cada modelo tiene 12 capas
            model_samples = latin_samples[model_idx].reshape(12, 3)

            # Mapear cada muestra a un modelo (JSON)
            model_dict = {
                "layers": [
                    map_to_architecture_params(sample) for sample in model_samples
                ]
            }

            # Realizar el encoding del modelo
            encoded_chromosome = encode_model_architecture(model_dict, max_alleles=max_alleles)

            # Guardar en el archivo CSV
            writer.writerow([model_idx + 1, encoded_chromosome])

    print(f"Cromosomas codificados guardados en {filename}")
    

# Mapear valores normalizados a arquitecturas
def map_to_architecture_params(latin_hypercube_sample):
    layer_type = int(latin_hypercube_sample[0] * 9)  # 9 tipos de capas
    layer_mapping = ['Conv2D', 'DepthwiseConv2D', 'BatchNorm', 'MaxPooling', 
                     'Dropout', 'Dense', 'Flatten', 'DontCare', 'Repetition']

    layer_type_name = layer_mapping[layer_type]

    if layer_type_name == 'Conv2D':
        return {
            "type": "Conv2D",
            "filters": int(latin_hypercube_sample[1] * (16 - 4) + 4),  # [4, 16]
            "strides": 1 if latin_hypercube_sample[2] < 0.5 else 2,
            "activation": "relu"
        }
    elif layer_type_name == 'DepthwiseConv2D':
        return {
            "type": "DepthwiseConv2D",
            "filters": int(latin_hypercube_sample[1] * (16 - 4) + 4),
            "strides": 1 if latin_hypercube_sample[2] < 0.5 else 2,
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
            "units": int(latin_hypercube_sample[1] * (128 - 1) + 1),
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
    return {}

# Ejecutar validación
# if validate_latin_hypercube(num_models=100):
#     # Guardar los cromosomas codificados en un archivo CSV si la validación pasa
#     save_encoded_models_to_csv(num_models=1000, filename="EncodedChromosomes.csv")


# %% [markdown]
# # GA
# 
# 

# %%
def int_to_real_dom(num, domain):
  min_i, max_i = domain
  r = (num - min_i) / (max_i - min_i)
  return r

def real_to_int_dom(num, domain):
  min_i, max_i = domain
  value = min_i + num * (max_i - min_i)
  if isinstance(min_i, int) and isinstance(max_i, int):
      value = int(round(value))
  return value

def convert_individual(ind, to_real=True):
    real_rep = []
    N = max(layer_type_options.keys())
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
        if layer_type in ['Conv2D', 'DepthwiseConv2D']:
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [4, 32]))
                real_rep.append(int_to_real_dom(ind[i + 2], [0, 1]))
                real_rep.append(int_to_real_dom(ind[i + 3], [0, 3]))
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [4, 32]))
                real_rep.append(real_to_int_dom(ind[i + 2], [0, 1]))
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
                real_rep.append(int_to_real_dom(ind[i + 1], [1, 512]))
                real_rep.append(int_to_real_dom(ind[i + 2], [0, 3]))
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [1, 512]))
                real_rep.append(real_to_int_dom(ind[i + 2], [0, 3]))
            real_rep.append(0)
        elif layer_type == 'Flatten':
            real_rep.extend([0, 0, 0])
        elif layer_type == 'Repetition':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [1, 4]))
                real_rep.append(int_to_real_dom(ind[i + 2], [1, 32]))
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [1, 4]))
                real_rep.append(real_to_int_dom(ind[i + 2], [1, 32]))
            real_rep.append(0)
        elif layer_type == 'DontCare':
            real_rep.extend([0, 0, 0])
    return real_rep

# %%
import numpy as np

def get_succ_m(parents, children):
  # Count if mutation was successful than the average parent fitness
  parent_fitness = [parent['fitness'] for parent in parents]
  avg_fitness = np.mean(parent_fitness)

  succ_m_count = sum(1 for child in children if child['fitness'] < avg_fitness)

  return succ_m_count

# %%
def get_cr_points(rp, n):
  indexes = list(range(n))
  j_star = random.sample(indexes, n // 2)

  for j in range(n):
    if random.random() < rp and j not in j_star:
      j_star.append(j)

  return j_star

# %%
def generate_latin_hypercube_samples(num_samples, dimensions):
    """
    Genera muestras usando el hipercubo latino.
    """
    # Generar muestras usando pyDOE2
    samples = lhs(dimensions, samples=num_samples)
    return samples

def pop_gen(num_models, max_alleles=48):
    """
    Genera una población inicial utilizando el hipercubo latino y las funciones existentes.

    Args:
        num_models: int - Número de individuos a generar.
        max_alleles: int - Número máximo de alelos en los cromosomas.

    Returns:
        list - Lista de diccionarios con individuos y su fitness inicializado a 0.
    """
    # Padres iniciales predefinidos
    initial_parents = [
        [0, 30, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0, 4, 1, 2, 0],
        [1, 0, 0, 0, 0, 16, 0, 1, 1, 0, 0, 0, 0, 8, 0, 1, 1, 0, 0, 0, 5, 0, 0, 0, 4, 32, 1, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0],
        [0, 32, 0, 1, 1, 0, 0, 0, 2, 1, 0, 0, 8, 3, 31, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 3, 0, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0]
    ]
    
    # Mitad de la población desde los padres iniciales
    archs = [{'individual': fixArch(parent), 'fitness': 0} for parent in initial_parents]
    
    # Generar el resto usando hipercubo latino
    num_random = num_models - len(initial_parents)
    dimensions = 12 * 3  # 12 capas, 3 parámetros por capa
    
    # Generar muestras del hipercubo latino
    latin_samples = generate_latin_hypercube_samples(num_random, dimensions)
    
    for sample in latin_samples:
        # Transformar cada muestra en una arquitectura
        model_samples = sample.reshape(12, 3)
        model_dict = {
            "layers": [map_to_architecture_params(layer_sample) for layer_sample in model_samples]
        }
        
        # Codificar el modelo y repararlo
        encoded_chromosome = encode_model_architecture(model_dict, max_alleles=max_alleles)
        repaired_architecture = fixArch(encoded_chromosome)
        
        # Añadir a la población
        archs.append({'individual': repaired_architecture, 'fitness': 0})
    
    return archs


# %%
def pop_gen_with_initial_parents(num_models, initial_parents, max_alleles=48):
    """
    Genera una población inicial combinando padres predefinidos con una población generada aleatoriamente.

    Args:
        num_models (int): Número total de individuos en la población inicial.
        initial_parents (list): Lista de cromosomas predefinidos como padres iniciales.
        max_alleles (int): Número máximo de alelos en los cromosomas.

    Returns:
        list: Población inicial con individuos y su fitness inicializado a 0.
    """
    # Generar población aleatoria restante
    num_random = num_models - len(initial_parents)
    if num_random < 0:
        raise ValueError("El número de padres iniciales excede el tamaño de la población deseada.")

    # Generar población aleatoria usando el hipercubo latino
    dimensions = max_alleles
    latin_samples = generate_latin_hypercube_samples(num_random, dimensions)

    # Convertir las muestras aleatorias en cromosomas válidos
    random_individuals = []
    for sample in latin_samples:
        repaired = fixArch(sample.tolist())
        random_individuals.append({'individual': repaired, 'fitness': 0})

    # Incluir los padres iniciales en la población
    parent_individuals = [{'individual': fixArch(parent), 'fitness': 0} for parent in initial_parents]

    # Combinar padres iniciales y población aleatoria
    population = parent_individuals + random_individuals

    return population


# %%
import os
import pickle

def save_checkpoint(filename, population, generation, best_fitness_per_gen, Fs):
    """
    Guarda el estado del algoritmo evolutivo en un archivo.
    
    Args:
        filename (str): Nombre del archivo para guardar el checkpoint.
        population (list): Población actual.
        generation (int): Generación actual.
        best_fitness_per_gen (list): Mejor fitness por generación.
        Fs (list): Valores históricos del factor F (si se usa auto-adaptación).
    """
    checkpoint = {
        'population': population,
        'generation': generation,
        'best_fitness_per_gen': best_fitness_per_gen,
        'Fs': Fs
    }
    #print(checkpoint)
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    #print(f"Checkpoint guardado en {filename}")

def load_checkpoint(filename):
    """
    Carga el estado del algoritmo evolutivo desde un archivo.

    Args:
        filename (str): Nombre del archivo del checkpoint.

    Returns:
        dict: Estado del algoritmo evolutivo cargado.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint cargado desde {filename}")
        return checkpoint
    else:
        print(f"No se encontró un checkpoint previo en {filename}. Comenzando desde cero.")
        return None


# %%
import tqdm

def es(target_func, mu=10, lamb=1, F=2, rp=0.5, gens=100, n=10, auto_adapt=False, 
       initial_parents=None, checkpoint_file='checkpoint.pkl', resume=False):
    """
    Estrategia evolutiva con soporte para checkpoints.

    Args:
        target_func: Función objetivo.
        mu: Tamaño de la población.
        lamb: Número de hijos generados por generación.
        F: Factor de escalamiento para mutación.
        rp: Probabilidad de recombinación.
        gens: Número total de generaciones.
        n: Ventana para auto-adaptación.
        auto_adapt: Si se usa auto-adaptación.
        initial_parents: Padres iniciales predefinidos.
        checkpoint_file: Archivo para guardar/cargar checkpoints.
        resume: Si se reanuda desde un checkpoint.
    """
    if resume:
        # Cargar estado desde checkpoint
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            pop = checkpoint['population']
            start_gen = checkpoint['generation']
            best_fitness_per_gen = checkpoint['best_fitness_per_gen']
            Fs = checkpoint['Fs']
        else:
            # Comenzar desde cero si no se puede cargar el checkpoint
             # Generar población inicial
            if initial_parents:
                pop = pop_gen_with_initial_parents(mu, initial_parents)
            else:
                pop = pop_gen(mu)       
            start_gen = 0
            best_fitness_per_gen = []
            Fs = []
    else:
        # Generar población inicial
        if initial_parents:
            pop = pop_gen_with_initial_parents(mu, initial_parents)
        else:
            pop = pop_gen(mu)
        start_gen = 0
        best_fitness_per_gen = []
        Fs = []

    succ_m_count = 0

    # Barra de progreso para generaciones
    with tqdm(total=gens, initial=start_gen, desc="Generations", leave=False) as pbar_gens:
        for gen in range(start_gen, gens):
            children = []

            # Evaluar fitness de los padres
            for parent in pop:
                parent['fitness'] = target_func(parent['individual'])
            best_parent = max(pop, key=lambda x: x['fitness'])

            for i in range(lamb):
                parent = pop[i]['individual']
                parent_unit = convert_individual(parent)
                best_parent_c = convert_individual(best_parent['individual'])

                # Mutación y recombinación
                x2, x3 = random.sample(pop, 2)
                x2 = convert_individual(x2['individual'])
                x3 = convert_individual(x3['individual'])
                u = [best_parent_c[j] + F * (x2[j] - x3[j]) for j in range(len(best_parent_c))]
                u = fixArch(convert_individual(u, to_real=False))
                u = convert_individual(u)
                cr_points = get_cr_points(rp, len(parent))
                child = [u[j] if j in cr_points else parent[j] for j in range(len(u))]
                child = fixArch(convert_individual(child, to_real=False))
                children.append({'individual': child, 'fitness': 0})

            # Evaluar fitness de los hijos
            children = [{'individual': child['individual'], 'fitness': target_func(child['individual'])} for child in children]

            # Seleccionar los mejores para la próxima generación
            complete_pop = pop + children
            pop = sorted(complete_pop, key=lambda x: x['fitness'], reverse=True)[:mu]
            best_fitness_per_gen.append(max(pop, key=lambda x: x['fitness'])['fitness'])

            if auto_adapt:
                # Ajustar F según el éxito de las mutaciones
                succ_m_count += get_succ_m(pop, children)
                if (gen + 1) % n == 0:
                    Fs.append(F)
                    ps = succ_m_count / (n * lamb)
                    F = F / 0.817 if ps > 1/5 else F * 0.817
                    succ_m_count = 0

            # Guardar checkpoint
            save_checkpoint(checkpoint_file, pop, gen + 1, best_fitness_per_gen, Fs)

            # Actualizar barra de progreso
            pbar_gens.update(1)

    best_element = max(pop, key=lambda x: x['fitness'])
    return best_element, best_fitness_per_gen, Fs


# %%
import copy
import joblib

def eval_arch(ind):
  ind_c = copy.deepcopy(ind)
  reshaped_ind_c = np.array(ind_c).reshape(1, -1)
  model = joblib.load("F1_SVM_optimized_model.pkl")
  acc = model.predict(reshaped_ind_c)
  return acc[0]

# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Función para cargar supracheckpoint
def load_super_checkpoint(super_checkpoint_file):
    if os.path.exists(super_checkpoint_file):
        with open(super_checkpoint_file, 'rb') as f:
            super_checkpoint = pickle.load(f)
        print(f"Supracheckpoint cargado desde {super_checkpoint_file}")
        return super_checkpoint
    else:
        print(f"No se encontró un supracheckpoint en {super_checkpoint_file}. Comenzando desde el experimento 0.")
        return {'last_experiment': 0}

# Guardar supracheckpoint
def save_super_checkpoint(super_checkpoint_file, last_experiment):
    super_checkpoint = {'last_experiment': last_experiment}
    with open(super_checkpoint_file, 'wb') as f:
        pickle.dump(super_checkpoint, f)
    print(f"Supracheckpoint guardado en {super_checkpoint_file}")

# Archivo de supracheckpoint
super_checkpoint_file = 'super_checkpoint.pkl'
super_checkpoint = load_super_checkpoint(super_checkpoint_file)
last_experiment = super_checkpoint['last_experiment']

# Experimentos
from tqdm import tqdm
initial_parents = [
    [0, 30, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0, 4, 1, 2, 0],
    [1, 0, 0, 0, 0, 16, 0, 1, 1, 0, 0, 0, 0, 8, 0, 1, 1, 0, 0, 0, 5, 0, 0, 0, 4, 32, 1, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0],
    [0, 32, 0, 1, 1, 0, 0, 0, 2, 1, 0, 0, 8, 3, 31, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 3, 0, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0]
]

best_models = []
all_best_fitness = []
F_arr = []
NUMBER_EXPERIMENTS = 30

# Retomar experimentos desde el último registrado
for i in tqdm(range(last_experiment, NUMBER_EXPERIMENTS), desc="Running Experiments"):
    best_model, best_fitness_per_gen, Fs = es(
        eval_arch,
        gens=500,  # 1000
        F=0.5,
        mu=500,  # 5000
        auto_adapt=True,
        initial_parents=initial_parents,
        checkpoint_file=f"checkpoint_{i}.pkl",
        resume=True
    )
    F_arr.append(Fs)
    best_models.append(best_model)
    all_best_fitness.append(best_fitness_per_gen)

    # Actualizar y guardar supracheckpoint
    save_super_checkpoint(super_checkpoint_file, i + 1)

# %%
# Cargar y combinar resultados de todos los checkpoints
all_fitness_from_checkpoints = []
for i in range(NUMBER_EXPERIMENTS):
    checkpoint_file = f"checkpoint_{i}.pkl"
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        all_fitness_from_checkpoints.append(checkpoint['best_fitness_per_gen'])

# Convertir los valores de fitness en una matriz para las gráficas
max_generations = max(len(fitness) for fitness in all_fitness_from_checkpoints)
generations = np.arange(1, max_generations + 1)
fitness_matrix = np.full((len(all_fitness_from_checkpoints), max_generations), np.nan)

for i, fitness in enumerate(all_fitness_from_checkpoints):
    fitness_matrix[i, :len(fitness)] = fitness

# Graficar el progreso
plt.figure(figsize=(12, 8))
for i in range(fitness_matrix.shape[0]):
    plt.plot(
        generations,
        fitness_matrix[i],
        linestyle='-',
        color='red',
        alpha=0.5
    )

# Calcular estadísticas
mean_fitness = np.nanmean(fitness_matrix, axis=0)
std_fitness = np.nanstd(fitness_matrix, axis=0)

# Graficar la media del fitness
plt.plot(
    generations,
    mean_fitness,
    linestyle='-',
    color='blue',
    label='Mean Fitness'
)

# Rellenar entre la media y la desviación estándar
plt.fill_between(
    generations,
    mean_fitness - std_fitness,
    mean_fitness + std_fitness,
    color='blue',
    alpha=0.2,
    label='Standard Deviation'
)

plt.title('Convergence Plot of Fitness per Generation')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()

# Graficar Box Plot
plt.figure(figsize=(12, 8))
plt.boxplot(
    fitness_matrix.T,  # Transponer para que cada generación sea una caja
    patch_artist=True,
    showmeans=True,
    boxprops=dict(facecolor='lightblue', color='blue'),
    meanprops=dict(marker='o', markerfacecolor='red', markersize=5),
    medianprops=dict(color='green')
)
plt.title('Box Plot of Fitness per Generation')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.grid(True)
plt.show()


# %%
# Cargar y combinar resultados de todos los checkpoints
all_fitness_from_checkpoints = []
best_individuals = []

for i in range(30):  # Ajusta el rango según el número de experimentos realizados
    checkpoint_file = f"checkpoint_{i}.pkl"
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        all_fitness_from_checkpoints.append(checkpoint['best_fitness_per_gen'])
        best_individuals.append(max(checkpoint['population'], key=lambda x: x['fitness']))

# Encontrar el mejor individuo entre todos los experimentos
best_overall_individual = max(best_individuals, key=lambda x: x['fitness'])
print(f"El mejor individuo tiene un fitness de: {best_overall_individual['fitness']}")
print(f"Codificación del mejor individuo: {best_overall_individual['individual']}")


# %%
import os
import copy
import json
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import torchaudio
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Resizing, Conv2D, Dropout, BatchNormalization, MaxPooling2D, MaxPool2D, Flatten, Dense, Input, LeakyReLU
from tqdm import tqdm
import tensorflow_addons as tfa

# Configuración de parámetros
class Config:
    def __init__(self, architecture='random', epochs=50, sample_rate=None, time=5, n_splits=5, window_size=5, checkpoint_file="training_checkpoint.json"):
        self.architecture = architecture
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.time = time
        self.n_splits = n_splits
        self.window_size = window_size
        self.checkpoint_file = checkpoint_file

# Crear o cargar checkpoint
def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            checkpoint = json.load(f)
            print(f"Checkpoint cargado: {checkpoint}")
            return checkpoint
    return {"last_completed": -1}

def save_checkpoint(file_path, architecture_index):
    checkpoint = {"last_completed": architecture_index}
    with open(file_path, 'w') as f:
        json.dump(checkpoint, f)
    print(f"Checkpoint guardado: {checkpoint}")

# Cargar datos de audio
def load_audio_data(directory, window_size, sample_rate):
    audio_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(directory, file_name))
            if sample_rate is None:
                sample_rate = sr
            num_windows = int(waveform.shape[1] / (window_size * sample_rate))
            for i in range(num_windows):
                start = i * window_size * sample_rate
                end = (i + 1) * window_size * sample_rate
                audio_dict[f"{file_name}_{i}"] = waveform[:, start:end].numpy()
    return audio_dict, sample_rate

# Preprocesar datos de audio
def preprocess_audio(audio_dict, sample_rate):
    audio_dict = copy.deepcopy(audio_dict)
    n_mels = 128
    n_fft = int(sample_rate * 0.029)
    hop_length = int(sample_rate * 0.010)
    win_length = int(sample_rate * 0.025)

    for filename, waveform in tqdm(audio_dict.items(), desc='MELSPECTROGRAM'):
        waveform = torch.from_numpy(waveform)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length)(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        spec = spec.numpy()
        spec = (spec - spec.min()) / (spec.max() - spec.min())
        audio_dict[filename] = spec
    return audio_dict

# Padding de los espectrogramas
def pad_and_crop_spectrograms(spectrograms, target_shape=(128, 128)):
    padded_spectrograms = []
    for spec in spectrograms:
        if spec.shape[0] > target_shape[0]:
            spec = spec[:target_shape[0], :]
        if spec.shape[1] > target_shape[1]:
            spec = spec[:, :target_shape[1]]
        
        pad_width = [(0, max(0, target_shape[0] - spec.shape[0])), 
                     (0, max(0, target_shape[1] - spec.shape[1]))]
        
        padded_spec = np.pad(spec, pad_width, mode='constant')
        padded_spectrograms.append(padded_spec)
    return np.array(padded_spectrograms)

# Split de audio en train y test
def train_test_split_audio(audio_dict):
    df = pd.read_csv('Dataset.csv', usecols=['Participant_ID', 'PHQ-9 Score'], dtype={1: str})
    df['labels'] = np.zeros([len(df),], dtype=int)
    df.loc[df['PHQ-9 Score'] < 10, 'labels'] = 0
    df.loc[df['PHQ-9 Score'] >= 10, 'labels'] = 1

    labels = df.set_index('Participant_ID').to_dict()['labels']

    X, Y = [], []
    for filename, data in tqdm(audio_dict.items(), 'LABEL'):
        ID = filename[:3]
        if ID in labels:
            dep = 0 if labels[ID] == 0 else 1
            [X.append(x) for x in data]
            [Y.append(dep) for x in data]

    X = pad_and_crop_spectrograms(X)
    Y = np.array(Y)

    X = X[..., np.newaxis]
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y

# Guardar resultados en CSV (append)
def append_results_to_csv(file_path, model_results):
    columns = ["Encoded Architecture", "Loss", "Accuracy", "Precision", "Recall", "F1", "Specificity"]

    # Convertir arquitectura a cadena para guardar
    model_results = [str(model_results[0])] + model_results[1:]

    # Verificar si el archivo ya existe
    if not os.path.exists(file_path):
        # Crear archivo con encabezados si no existe
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    # Escribir resultados en el archivo
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(model_results)
    print(f"Resultados guardados en: {file_path}")

# Función de especificidad
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Entrenar y evaluar modelo
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config):
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy", 'Precision', 'Recall'])
    model.fit(X_train, Y_train, epochs=config.epochs, validation_data=(X_val, Y_val), verbose=0)
    results = model.evaluate(X_test, Y_test, verbose=0)

    # Obtener predicciones para métricas adicionales
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = results[1]
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    specificity = specificity_score(Y_test, Y_pred)

    return [results[0], accuracy, precision, recall, f1, specificity]

# Evaluar y almacenar resultados
def evaluate_and_store_model(architecture, X_train_val, X_test, Y_train_val, Y_test, config, use_kfold, stratified_kfold, target_shape, results_file):
    repaired_architecture = fixArch(architecture)
    decoded_model_dict = decode_model_architecture(repaired_architecture)
    model_results = [repaired_architecture]

    if use_kfold:
        fold_results = []
        for fold, (train_index, val_index) in enumerate(stratified_kfold.split(X_train_val, Y_train_val)):
            print(f"Entrenando fold {fold + 1}/{config.n_splits}...")
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
            tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=(target_shape[0], target_shape[1], 1))
            fold_results.append(train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config))
            print(f"Fold {fold + 1} completado.")

        avg_results = np.mean(fold_results, axis=0)
        model_results.extend(avg_results)

    else:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)
        tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=(target_shape[0], target_shape[1], 1))
        single_run_results = train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config)
        model_results.extend(single_run_results)
        print("Modelo evaluado sin K-Fold Cross Validation.")

    # Guardar métricas de la arquitectura actual en el archivo CSV
    append_results_to_csv(results_file, model_results)



# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import torchaudio
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import copy
import os

# Función de especificidad
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Entrenar y evaluar modelo
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs):
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy", 'Precision', 'Recall'])
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), verbose=0)
    results = model.evaluate(X_test, Y_test, verbose=0)

    # Obtener predicciones para métricas adicionales
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = results[1]
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    specificity = specificity_score(Y_test, Y_pred)

    return [results[0], accuracy, precision, recall, f1, specificity]

# Evaluar y almacenar resultados
def evaluate_best_individual(best_individual, X_train_val, X_test, Y_train_val, Y_test, target_shape, epochs=50, n_splits=5):
    # Decodificar la arquitectura
    repaired_architecture = fixArch(best_individual)
    decoded_model_dict = decode_model_architecture(repaired_architecture)

    fold_results = []
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(stratified_kfold.split(X_train_val, Y_train_val)):
        print(f"Entrenando fold {fold + 1}/{n_splits}...")
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        
        # Crear el modelo a partir de la arquitectura
        tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=(target_shape[0], target_shape[1], 1))
        fold_results.append(train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs))
        print(f"Fold {fold + 1} completado.")

    avg_results = np.mean(fold_results, axis=0)
    print(f"Resultados promedio en {n_splits} folds: Loss={avg_results[0]:.4f}, Accuracy={avg_results[1]:.4f}, Precision={avg_results[2]:.4f}, Recall={avg_results[3]:.4f}, F1={avg_results[4]:.4f}, Specificity={avg_results[5]:.4f}")

# Preprocesamiento de datos de audio
def load_audio_data(directory, window_size, sample_rate):
    audio_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(directory, file_name))
            if sample_rate is None:
                sample_rate = sr
            num_windows = int(waveform.shape[1] / (window_size * sample_rate))
            for i in range(num_windows):
                start = i * window_size * sample_rate
                end = (i + 1) * window_size * sample_rate
                audio_dict[f"{file_name}_{i}"] = waveform[:, start:end].numpy()
    return audio_dict, sample_rate

def preprocess_audio(audio_dict, sample_rate):
    audio_dict = copy.deepcopy(audio_dict)
    n_mels = 128
    n_fft = int(sample_rate * 0.029)
    hop_length = int(sample_rate * 0.010)
    win_length = int(sample_rate * 0.025)

    for filename, waveform in tqdm(audio_dict.items(), desc='MELSPECTROGRAM'):
        waveform = torch.from_numpy(waveform)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length)(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        spec = spec.numpy()
        spec = (spec - spec.min()) / (spec.max() - spec.min())
        audio_dict[filename] = spec
    return audio_dict

def pad_and_crop_spectrograms(spectrograms, target_shape=(128, 128)):
    padded_spectrograms = []
    for spec in spectrograms:
        if spec.shape[0] > target_shape[0]:
            spec = spec[:target_shape[0], :]
        if spec.shape[1] > target_shape[1]:
            spec = spec[:, :target_shape[1]]
        
        pad_width = [(0, max(0, target_shape[0] - spec.shape[0])), 
                     (0, max(0, target_shape[1] - spec.shape[1]))]
        
        padded_spec = np.pad(spec, pad_width, mode='constant')
        padded_spectrograms.append(padded_spec)
    return np.array(padded_spectrograms)

def train_test_split_audio(audio_dict):
    df = pd.read_csv('Dataset.csv', usecols=['Participant_ID', 'PHQ-9 Score'], dtype={1: str})
    df['labels'] = np.zeros([len(df),], dtype=int)
    df.loc[df['PHQ-9 Score'] < 10, 'labels'] = 0
    df.loc[df['PHQ-9 Score'] >= 10, 'labels'] = 1

    labels = df.set_index('Participant_ID').to_dict()['labels']

    X, Y = [], []
    for filename, data in tqdm(audio_dict.items(), 'LABEL'):
        ID = filename[:3]
        if ID in labels:
            dep = 0 if labels[ID] == 0 else 1
            [X.append(x) for x in data]
            [Y.append(dep) for x in data]

    X = pad_and_crop_spectrograms(X)
    Y = np.array(Y)

    X = X[..., np.newaxis]
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y

# Cargar datos
directory = './SM-27'  # Ruta a los archivos de audio
window_size = 5  # Tamaño de la ventana en segundos
sample_rate = None  # Se determinará en la carga

print("Cargando y preprocesando datos de audio...")
audio_dict, sample_rate = load_audio_data(directory, window_size, sample_rate)
audio_dict = preprocess_audio(audio_dict, sample_rate)
X, Y = train_test_split_audio(audio_dict)

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Evaluar el mejor individuo (sustituye 'best_overall_individual' con la arquitectura específica)
evaluate_best_individual(
    best_overall_individual['individual'],
    X_train_val,
    X_test,
    Y_train_val,
    Y_test,
    target_shape=(128, 128),
    epochs=100,
    n_splits=5
)


# %%
best_individual = best_overall_individual['individual']
constructed_model = build_tf_model_from_dict(decode_model_architecture(best_individual))
print(constructed_model.summary())


# %%
import os

def delete_checkpoints(folder_path, checkpoint_prefix="checkpoint_", super_checkpoint_file="super_checkpoint.pkl"):
    """
    Elimina todos los archivos de checkpoint y supracheckpoint.

    Args:
        folder_path (str): Ruta del directorio donde se encuentran los checkpoints.
        checkpoint_prefix (str): Prefijo de los archivos de checkpoint.
        super_checkpoint_file (str): Nombre del archivo de supracheckpoint.
    """
    # Borrar checkpoints individuales
    for filename in os.listdir(folder_path):
        if filename.startswith(checkpoint_prefix):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Eliminado: {file_path}")
            except Exception as e:
                print(f"Error al eliminar {file_path}: {e}")
    
    # Borrar supracheckpoint
    super_checkpoint_path = os.path.join(folder_path, super_checkpoint_file)
    if os.path.exists(super_checkpoint_path):
        try:
            os.remove(super_checkpoint_path)
            print(f"Eliminado: {super_checkpoint_path}")
        except Exception as e:
            print(f"Error al eliminar {super_checkpoint_path}: {e}")

# Ejecutar la función
delete_checkpoints(".", checkpoint_prefix="checkpoint_", super_checkpoint_file="super_checkpoint.pkl")



