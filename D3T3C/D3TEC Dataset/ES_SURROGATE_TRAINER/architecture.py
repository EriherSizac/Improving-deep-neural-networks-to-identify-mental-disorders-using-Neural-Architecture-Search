"""
Architecture Module

Este módulo contiene funciones para codificar, decodificar y manipular 
arquitecturas de redes neuronales para el proceso de búsqueda de arquitectura.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Opciones de tipos de capas
LAYER_TYPE_OPTIONS = {
    0: 'Conv2D',
    1: 'SelfAttention',
    2: 'BatchNorm',
    3: 'MaxPooling',
    4: 'Dropout',
    5: 'Dense',
    6: 'Flatten',
    7: 'DontCare'
}

# Opciones de stride
STRIDE_OPTIONS = {
    0: 1,
    1: 2,
    2: 3,
    3: 4
}

# Opciones de dropout
DROPOUT_OPTIONS = {
    0: 0.1,
    1: 0.2,
    2: 0.3,
    3: 0.5
}

# Opciones de activación
ACTIVATION_OPTIONS = {
    0: 'relu',
    1: 'sigmoid',
    2: 'tanh',
    3: 'linear'
}

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

def encode_layer_params(layer_type_idx, param1=0, param2=0, param3=0):
    """
    Codifica los parámetros de una capa en una lista de enteros.
    
    Args:
        layer_type_idx: Índice del tipo de capa
        param1, param2, param3: Parámetros específicos de la capa
        
    Returns:
        list: Lista con los parámetros codificados
    """
    return [layer_type_idx, param1, param2, param3]

def encode_model_architecture(model_dict):
    """
    Codifica una arquitectura de modelo en una lista de enteros.
    
    Args:
        model_dict: Diccionario con la definición del modelo
        
    Returns:
        list: Arquitectura codificada como lista de enteros
    """
    encoded_model = []
    
    for layer in model_dict['layers']:
        layer_type = layer['type']
        
        if layer_type == 'Conv2D':
            # Conv2D: [0, stride, kernel_size, activation]
            stride_idx = next((k for k, v in STRIDE_OPTIONS.items() if v == layer.get('stride', 1)), 0)
            kernel_idx = min(layer.get('kernel_size', 3) - 1, 3)  # 1->0, 2->1, 3->2, 4->3
            activation_idx = next((k for k, v in ACTIVATION_OPTIONS.items() if v == layer.get('activation', 'relu')), 0)
            encoded_model.extend(encode_layer_params(0, stride_idx, kernel_idx, activation_idx))
            
        elif layer_type == 'SelfAttention':
            # SelfAttention: [1, 0, 0, 0]
            encoded_model.extend(encode_layer_params(1, 0, 0, 0))
            
        elif layer_type == 'BatchNorm':
            # BatchNorm: [2, 0, 0, 0]
            encoded_model.extend(encode_layer_params(2, 0, 0, 0))
            
        elif layer_type == 'MaxPooling':
            # MaxPooling: [3, pool_size, 0, 0]
            pool_idx = min(layer.get('pool_size', 2) - 1, 3)  # 1->0, 2->1, 3->2, 4->3
            encoded_model.extend(encode_layer_params(3, pool_idx, 0, 0))
            
        elif layer_type == 'Dropout':
            # Dropout: [4, rate_idx, 0, 0]
            rate = layer.get('rate', 0.2)
            rate_idx = next((k for k, v in DROPOUT_OPTIONS.items() if abs(v - rate) < 0.05), 1)
            encoded_model.extend(encode_layer_params(4, rate_idx, 0, 0))
            
        elif layer_type == 'Dense':
            # Dense: [5, units_idx, 0, activation]
            units = layer.get('units', 64)
            units_idx = 0  # Por defecto
            if units <= 32:
                units_idx = 0
            elif units <= 64:
                units_idx = 1
            elif units <= 128:
                units_idx = 2
            else:
                units_idx = 3
                
            activation_idx = next((k for k, v in ACTIVATION_OPTIONS.items() if v == layer.get('activation', 'relu')), 0)
            encoded_model.extend(encode_layer_params(5, units_idx, 0, activation_idx))
            
        elif layer_type == 'Flatten':
            # Flatten: [6, 0, 0, 0]
            encoded_model.extend(encode_layer_params(6, 0, 0, 0))
            
        else:  # DontCare o cualquier otro tipo no reconocido
            # DontCare: [7, 0, 0, 0]
            encoded_model.extend(encode_layer_params(7, 0, 0, 0))
    
    # Asegurar que la longitud es exactamente 80 (20 capas * 4 parámetros)
    if len(encoded_model) < 80:
        # Rellenar con capas DontCare
        padding = [7, 0, 0, 0] * ((80 - len(encoded_model)) // 4)
        encoded_model.extend(padding)
    
    # Truncar si es más largo
    encoded_model = encoded_model[:80]
    
    return encoded_model

def fixArch(encoded_arch):
    """
    Corrige una arquitectura codificada para asegurar que es válida.
    
    Args:
        encoded_arch: Arquitectura codificada como lista o array
        
    Returns:
        numpy.ndarray: Arquitectura corregida
    """
    # Convertir a numpy array si es necesario
    if not isinstance(encoded_arch, np.ndarray):
        encoded_arch = np.array(encoded_arch)
    
    # Asegurar que la longitud es correcta
    if len(encoded_arch) != 80:
        # Crear array de tamaño correcto
        fixed_arch = np.zeros(80, dtype=int)
        
        # Copiar los valores disponibles
        fixed_arch[:min(len(encoded_arch), 80)] = encoded_arch[:min(len(encoded_arch), 80)]
        
        # Rellenar con DontCare si es necesario
        for i in range(min(len(encoded_arch), 80), 80, 4):
            fixed_arch[i:i+4] = [7, 0, 0, 0]  # DontCare
    else:
        fixed_arch = encoded_arch.copy()
    
    # Corregir valores fuera de rango
    for i in range(0, 80, 4):
        # Corregir tipo de capa
        if fixed_arch[i] < 0 or fixed_arch[i] > 7:
            fixed_arch[i] = 7  # DontCare si está fuera de rango
        
        # Corregir parámetros según el tipo de capa
        for j in range(1, 4):
            if fixed_arch[i+j] < 0 or fixed_arch[i+j] > 3:
                fixed_arch[i+j] = 0
    
    # Asegurar que hay al menos una capa Flatten antes de Dense
    has_flatten = False
    has_dense = False
    dense_positions = []
    
    for i in range(0, 80, 4):
        if fixed_arch[i] == 6:  # Flatten
            has_flatten = True
        elif fixed_arch[i] == 5:  # Dense
            has_dense = True
            dense_positions.append(i)
    
    # Si hay Dense pero no hay Flatten, insertar Flatten antes de la primera Dense
    if has_dense and not has_flatten and dense_positions:
        first_dense = dense_positions[0]
        
        # Buscar una capa DontCare para reemplazar con Flatten
        for i in range(0, first_dense, 4):
            if fixed_arch[i] == 7:  # DontCare
                fixed_arch[i:i+4] = [6, 0, 0, 0]  # Flatten
                break
        else:
            # Si no hay DontCare, reemplazar la capa justo antes de Dense
            if first_dense >= 4:
                fixed_arch[first_dense-4:first_dense] = [6, 0, 0, 0]  # Flatten
    
    return fixed_arch

def select_group_for_repetition(groups, n_repetitions):
    """
    Selecciona un grupo para repetir basado en pesos.
    
    Args:
        groups: Lista de grupos disponibles
        n_repetitions: Número de repeticiones deseadas
        
    Returns:
        list: Índices de grupos seleccionados
    """
    weights = {
        'Conv2D': 0.4,
        'SelfAttention': 0.1,
        'BatchNorm': 0.1,
        'MaxPooling': 0.1,
        'Dropout': 0.05,
        'Dense': 0.2,
        'Flatten': 0.05,
        'DontCare': 0.0
    }
    
    # Calcular pesos para cada grupo
    group_weights = [weights.get(g['type'], 0.0) for g in groups]
    
    # Si todos los pesos son cero, usar distribución uniforme
    if sum(group_weights) == 0:
        group_weights = [1.0 / len(groups) for _ in groups]
    
    # Normalizar pesos
    total_weight = sum(group_weights)
    if total_weight > 0:
        group_weights = [w / total_weight for w in group_weights]
    
    # Seleccionar grupos
    selected_indices = random.choices(range(len(groups)), weights=group_weights, k=n_repetitions)
    return selected_indices

def decode_model_architecture(encoded_model):
    """
    Decodifica una arquitectura codificada en un formato legible.
    
    Args:
        encoded_model: Arquitectura codificada como lista o array
        
    Returns:
        dict: Diccionario con la arquitectura decodificada
    """
    # Asegurar que la arquitectura es válida
    encoded_model = fixArch(encoded_model)
    
    model_dict = {'layers': []}
    
    for i in range(0, len(encoded_model), 4):
        layer_type_idx = encoded_model[i]
        param1 = encoded_model[i+1]
        param2 = encoded_model[i+2]
        param3 = encoded_model[i+3]
        
        layer_type = LAYER_TYPE_OPTIONS.get(layer_type_idx, 'DontCare')
        
        if layer_type == 'Conv2D':
            # Conv2D: [0, stride, kernel_size, activation]
            stride = STRIDE_OPTIONS.get(param1, 1)
            kernel_size = param2 + 1  # 0->1, 1->2, 2->3, 3->4
            activation = ACTIVATION_OPTIONS.get(param3, 'relu')
            
            model_dict['layers'].append({
                'type': 'Conv2D',
                'filters': 32,  # Valor por defecto
                'kernel_size': kernel_size,
                'stride': stride,
                'activation': activation
            })
            
        elif layer_type == 'SelfAttention':
            # SelfAttention: [1, 0, 0, 0]
            model_dict['layers'].append({
                'type': 'SelfAttention'
            })
            
        elif layer_type == 'BatchNorm':
            # BatchNorm: [2, 0, 0, 0]
            model_dict['layers'].append({
                'type': 'BatchNorm'
            })
            
        elif layer_type == 'MaxPooling':
            # MaxPooling: [3, pool_size, 0, 0]
            pool_size = param1 + 1  # 0->1, 1->2, 2->3, 3->4
            
            model_dict['layers'].append({
                'type': 'MaxPooling',
                'pool_size': pool_size
            })
            
        elif layer_type == 'Dropout':
            # Dropout: [4, rate_idx, 0, 0]
            rate = DROPOUT_OPTIONS.get(param1, 0.2)
            
            model_dict['layers'].append({
                'type': 'Dropout',
                'rate': rate
            })
            
        elif layer_type == 'Dense':
            # Dense: [5, units_idx, 0, activation]
            units_mapping = {0: 32, 1: 64, 2: 128, 3: 256}
            units = units_mapping.get(param1, 64)
            activation = ACTIVATION_OPTIONS.get(param3, 'relu')
            
            model_dict['layers'].append({
                'type': 'Dense',
                'units': units,
                'activation': activation
            })
            
        elif layer_type == 'Flatten':
            # Flatten: [6, 0, 0, 0]
            model_dict['layers'].append({
                'type': 'Flatten'
            })
            
        elif layer_type == 'DontCare':
            # DontCare: [7, 0, 0, 0]
            model_dict['layers'].append({
                'type': 'DontCare'
            })
    
    return model_dict

class BuildPyTorchModel(nn.Module):
    """
    Construye un modelo PyTorch a partir de una arquitectura codificada.
    """
    def __init__(self, encoded_arch):
        super(BuildPyTorchModel, self).__init__()
        
        # Decodificar la arquitectura
        self.arch_dict = decode_model_architecture(encoded_arch)
        
        # Construir el modelo
        self.layers = nn.ModuleList()
        
        # Parámetros iniciales
        in_channels = 1  # Asumiendo entrada de 1 canal
        current_size = (64, 552)  # Tamaño de entrada (asumido)
        is_conv = True  # Flag para seguir el estado (convolucional o fully connected)
        
        for layer in self.arch_dict['layers']:
            layer_type = layer['type']
            
            if layer_type == 'DontCare':
                self.layers.append(DontCareLayer())
                
            elif layer_type == 'Conv2D' and is_conv:
                filters = 32  # Valor fijo para simplificar
                kernel_size = layer.get('kernel_size', 3)
                stride = layer.get('stride', 1)
                
                # Verificar que el kernel no es más grande que la entrada
                kernel_size = min(kernel_size, min(current_size))
                
                # Añadir capa convolucional
                self.layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2  # Same padding
                ))
                
                # Actualizar parámetros
                in_channels = filters
                current_size = (
                    (current_size[0] + 2 * (kernel_size // 2) - kernel_size) // stride + 1,
                    (current_size[1] + 2 * (kernel_size // 2) - kernel_size) // stride + 1
                )
                
                # Añadir activación
                activation = layer.get('activation', 'relu')
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                
            elif layer_type == 'SelfAttention' and is_conv:
                self.layers.append(SelfAttention(in_channels))
                
            elif layer_type == 'BatchNorm' and is_conv:
                self.layers.append(nn.BatchNorm2d(in_channels))
                
            elif layer_type == 'MaxPooling' and is_conv:
                pool_size = layer.get('pool_size', 2)
                
                # Verificar que el pool_size no es más grande que la entrada
                pool_size = min(pool_size, min(current_size))
                
                self.layers.append(nn.MaxPool2d(kernel_size=pool_size))
                
                # Actualizar tamaño
                current_size = (
                    current_size[0] // pool_size,
                    current_size[1] // pool_size
                )
                
            elif layer_type == 'Flatten':
                self.layers.append(nn.Flatten())
                is_conv = False
                in_features = in_channels * current_size[0] * current_size[1]
                
            elif layer_type == 'Dropout':
                rate = layer.get('rate', 0.2)
                self.layers.append(nn.Dropout(rate))
                
            elif layer_type == 'Dense' and not is_conv:
                units = layer.get('units', 64)
                
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
                
                # Añadir activación
                activation = layer.get('activation', 'relu')
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
    
    def forward(self, x):
        """
        Pasa los datos a través del modelo.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        for layer in self.layers:
            try:
                x = layer(x)
            except Exception as e:
                print(f"Error en capa {layer}: {e}")
                # Continuar con la siguiente capa en caso de error
                continue
        
        return x
