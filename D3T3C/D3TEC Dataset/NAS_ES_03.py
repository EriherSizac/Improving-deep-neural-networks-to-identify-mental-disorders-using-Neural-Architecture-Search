# %%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer,  Softmax, Multiply, Add, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, DepthwiseConv2D
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


torch.set_num_threads(1)  # Prueba con 4, 2 o 1


# %%
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
    #print(f"Final Encoded Model: {final_encoding}")

    return final_encoding


# %%


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

    print("Validación completada: Todas las muestras están dentro de los rangos esperados.")
    return True

# Guardar el encoding generado en un archivo CSV
def save_encoded_models_to_csv(num_models, filename, max_alleles=48):
    latin_samples = generate_latin_hypercube_samples(num_models, 12 * 3)  # 12 capas, 3 parámetros por capa

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Encoded Chromosome"])

        for model_idx in range(num_models):
            model_samples = latin_samples[model_idx].reshape(12, 3)

            model_dict = {
                "layers": [
                    map_to_architecture_params(sample) for sample in model_samples
                ]
            }

            encoded_chromosome = fixArch(encode_model_architecture(model_dict, max_alleles=max_alleles))
            writer.writerow([model_idx + 1, encoded_chromosome])

    print(f"Cromosomas codificados guardados en {filename}")

# Mapear valores normalizados a arquitecturas
def map_to_architecture_params(latin_hypercube_sample):
    layer_type = int(latin_hypercube_sample[0] * 9)  # 9 tipos de capas
    layer_mapping = ['Conv2D', 'SelfAttention', 'BatchNorm', 'MaxPooling',
                     'Dropout', 'Dense', 'Flatten', 'DontCare', 'Repetition']

    layer_type_name = layer_mapping[layer_type]

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
    return {}

# Ejecutar validación y guardado en CSV
""" if validate_latin_hypercube(num_models=5000):
    save_encoded_models_to_csv(num_models=5000, filename="EncodedChromosomes_V3.csv")
 """

# %%
import tensorflow as tf

# Clase para capas neutrales 'DontCare'
class DontCareLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DontCareLayer, self).__init__()

    def call(self, inputs):
        return inputs

# %% [markdown]
# # Busqueda del ga

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
    N = max(layer_type_options.keys())  # Obtener el máximo índice de capa en el diccionario

    for i in range(0, len(ind), 4):
        layer_type_idx = ind[i]
        domain_layer_type = [0, N]

        if to_real:
            real_rep.append(int_to_real_dom(layer_type_idx, domain_layer_type))
            layer_type = layer_type_options.get(layer_type_idx, 'DontCare')
        else:
            real_rep.append(real_to_int_dom(layer_type_idx, domain_layer_type))
            layer_type = layer_type_options.get(real_rep[i], 'DontCare')

        # Decodificación basada en el tipo de capa
        if layer_type == 'Conv2D':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [4, 32]))  # Filtros
                real_rep.append(int_to_real_dom(ind[i + 2], [0, 1]))  # Stride
                real_rep.append(int_to_real_dom(ind[i + 3], [0, 3]))  # Activación
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [4, 32]))
                real_rep.append(real_to_int_dom(ind[i + 2], [0, 1]))
                real_rep.append(real_to_int_dom(ind[i + 3], [0, 3]))

        elif layer_type == 'SelfAttention':  # 🚀 Reemplazo de DepthwiseConv2D por SelfAttention
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [4, 64]))  # Filtros
                real_rep.append(int_to_real_dom(ind[i + 2], [1, 8]))  # Número de cabezas de atención
                real_rep.append(int_to_real_dom(ind[i + 3], [0, 3]))  # Activación
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
                real_rep.append(int_to_real_dom(ind[i + 1], [1, 512]))  # Neuronas
                real_rep.append(int_to_real_dom(ind[i + 2], [0, 3]))  # Activación
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [1, 512]))
                real_rep.append(real_to_int_dom(ind[i + 2], [0, 3]))
            real_rep.append(0)

        elif layer_type == 'Flatten':
            real_rep.extend([0, 0, 0])

        elif layer_type == 'Repetition':
            if to_real:
                real_rep.append(int_to_real_dom(ind[i + 1], [1, 4]))  # Número de capas a repetir
                real_rep.append(int_to_real_dom(ind[i + 2], [1, 32]))  # Cantidad de repeticiones
            else:
                real_rep.append(real_to_int_dom(ind[i + 1], [1, 4]))
                real_rep.append(real_to_int_dom(ind[i + 2], [1, 32]))
            real_rep.append(0)

        elif layer_type == 'DontCare':
            real_rep.extend([0, 0, 0])

    return real_rep


# %%
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
def pop_gen(num_models, max_alleles=48):
    """
    Genera una población inicial utilizando el hipercubo latino y las funciones existentes.

    Args:
        num_models: int - Número de individuos a generar.
        max_alleles: int - Número máximo de alelos en los cromosomas.

    Returns:
        list - Lista de diccionarios con individuos y su fitness inicializado a 0.
    """
    archs = []
    dimensions = 12 * 3  # 12 capas, 3 parámetros por capa

    # Generar muestras del hipercubo latino
    latin_samples = generate_latin_hypercube_samples(num_models, dimensions)

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
from tqdm import tqdm

def es(target_func, mu=10, lamb=1, F=2, rp=0.5, gens=100, n=10, auto_adapt=False):
    # Initialize parent population
    pop = pop_gen(mu)
    succ_m_count = 0
    best_fitness_per_gen = []
    Fs = []

    # Barra de progreso para generaciones
    with tqdm(total=gens, desc="Generations", leave=False) as pbar_gens:
        for gen in range(gens):
            children = []

            # Calc fitness and select best element
            for parent in pop:
                parent['fitness'] = target_func(parent['individual'])
            best_parent = max(pop, key=lambda x: x['fitness'])

            for i in range(lamb):
                # print(f"📌 Generación {gen} - Individuo {i}/{lamb}")
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

                # Calc fitness of children
                children = [{'individual': child['individual'], 'fitness': target_func(child['individual'])} for child in children]

            # Select the best mu elements for next gen
            complete_pop = pop + children

            if(auto_adapt):
                # Count succesful mutations
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

            # Actualizar la barra de progreso de generaciones
            pbar_gens.update(1)

    best_element = max(pop, key=lambda x: x['fitness'])
    Fs.append(F)
    return best_element, best_fitness_per_gen, Fs


# %%
import os
import copy
import joblib
import numpy as np
import tensorflow as tf

def load_model_by_extension(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.pkl':
        return joblib.load(model_path)
    elif ext == '.h5':
        # Carga el modelo sin compilar (ignora las métricas guardadas)
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        raise ValueError(f"Extensión de modelo no soportada: {ext}")

# Especifica la ruta del modelo (puede ser .pkl o .h5)
model_path = "./DeepNN_model.h5"  # o, por ejemplo, "./F1_RandomForest_optimized.pkl"
model = load_model_by_extension(model_path)

def eval_arch(ind):
    ind_c = copy.deepcopy(ind)
    reshaped_ind_c = np.array(ind_c).reshape(1, -1)
    acc = model.predict(reshaped_ind_c, verbose=0)
    return acc[0]


# %%
from google.colab import drive
drive.mount('/content/drive')

# %%


# %%
# prompt: disable  sklearn warnings /usr/local/lib/python3.11/dist-packages/sklearn/base.py:380: InconsistentVersionWarning: Trying to unpickle estimator DecisionTreeRegressor from version 1.5.1 when using version 1.6.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# %%
from tqdm import tqdm

best_models = []
all_best_fitness = []
F_arr = []
# Agregar tqdm para mostrar progreso
for i in tqdm(range(30), desc="Running Experiments"):

    best_model, best_fitness_per_gen, Fs = es(
        eval_arch,
        gens=500,
        F=0.5,
        mu=700,
        auto_adapt=True
    )
    F_arr.append(Fs)
    best_models.append(best_model)
    all_best_fitness.append(best_fitness_per_gen)


# %%
import matplotlib.pyplot as plt

# Determine the maximum number of generations across all runs
max_generations = max(len(fitness) for fitness in all_best_fitness)
generations = np.arange(1, max_generations + 1)

# Initialize a matrix to hold all fitness values, handling varying lengths
fitness_matrix = np.full((30, max_generations), np.nan)

for i, best_fitness_per_gen in enumerate(all_best_fitness):
    fitness_length = len(best_fitness_per_gen)
    fitness_matrix[i, :fitness_length] = best_fitness_per_gen

# Plot all runs in light gray
plt.figure(figsize=(12, 8))
for i in range(30):
    plt.plot(
        generations,
        fitness_matrix[i],
        linestyle='-',
        color='red',
        alpha=0.5
    )

# Compute the mean and standard deviation across runs at each generation
mean_fitness = np.nanmean(fitness_matrix, axis=0)
std_fitness = np.nanstd(fitness_matrix, axis=0)

# Plot the mean best fitness
plt.plot(
    generations,
    mean_fitness,
    linestyle='-',
    color='blue',
    label='Mean Accuracy'
)

# Fill the area between (mean - std) and (mean + std)
plt.fill_between(
    generations,
    mean_fitness - std_fitness,
    mean_fitness + std_fitness,
    color='blue',
    alpha=0.2,
    label='Standard Deviation'
)

plt.title('Convergence Plot of Accuracy per Generation over 30 Runs')
plt.xlabel('Generations')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# %%


# %%
from tqdm import tqdm

best_models = []
all_best_fitness = []
F_arr = []
# Agregar tqdm para mostrar progreso
for i in tqdm(range(30), desc="Running Experiments"):

    best_model, best_fitness_per_gen, Fs = es(
        eval_arch,
        gens=1000,
        F=0.5,
        mu=5000,
        auto_adapt=True
    )
    F_arr.append(Fs)
    best_models.append(best_model)
    all_best_fitness.append(best_fitness_per_gen)


# %%
import matplotlib.pyplot as plt

# Determine the maximum number of generations across all runs
max_generations = max(len(fitness) for fitness in all_best_fitness)
generations = np.arange(1, max_generations + 1)

# Initialize a matrix to hold all fitness values, handling varying lengths
fitness_matrix = np.full((30, max_generations), np.nan)

for i, best_fitness_per_gen in enumerate(all_best_fitness):
    fitness_length = len(best_fitness_per_gen)
    fitness_matrix[i, :fitness_length] = best_fitness_per_gen

# Plot all runs in light gray
plt.figure(figsize=(12, 8))
for i in range(30):
    plt.plot(
        generations,
        fitness_matrix[i],
        linestyle='-',
        color='red',
        alpha=0.5
    )

# Compute the mean and standard deviation across runs at each generation
mean_fitness = np.nanmean(fitness_matrix, axis=0)
std_fitness = np.nanstd(fitness_matrix, axis=0)

# Plot the mean best fitness
plt.plot(
    generations,
    mean_fitness,
    linestyle='-',
    color='blue',
    label='Mean Accuracy'
)

# Fill the area between (mean - std) and (mean + std)
plt.fill_between(
    generations,
    mean_fitness - std_fitness,
    mean_fitness + std_fitness,
    color='blue',
    alpha=0.2,
    label='Standard Deviation'
)

plt.title('Convergence Plot of Accuracy per Generation over 30 Runs')
plt.xlabel('Generations')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Parameters
n = 10       # Adaptation interval (every n generations)
gens = 1000  # Total number of generations
num_runs = 30  # Number of runs

# Plotting
plt.figure(figsize=(12, 6))

for Fs in F_arr:
    # Generations where F was recorded
    generations = [n * i for i in range(1, len(Fs))]
    generations.append(generations[-1] + 1)
    # Adjust the last generation if it doesn't align with n
    if generations[-1] != gens:
        generations[-1] = gens
    plt.plot(generations, Fs, alpha=0.3, color='blue')

# Calculate and plot the mean F value at each adaptation point
max_length = max(len(Fs) for Fs in F_arr)

# Pad shorter Fs lists if necessary
Fs_padded = []
for Fs in F_arr:
    padding_length = max_length - len(Fs)
    Fs_padded.append(Fs + [Fs[-1]] * padding_length)

mean_Fs = [sum(Fs[i] for Fs in Fs_padded) / num_runs for i in range(max_length)]
mean_generations = [n * i for i in range(1, max_length)]
mean_generations.append(mean_generations[-1] + 1)
# Adjust the last generation if necessary
if mean_generations[-1] != gens:
    mean_generations[-1] = gens

plt.plot(mean_generations, mean_Fs, color='red', linewidth=2, label='Mean F Value')

# Label the axes and add a title
plt.xlabel('Generation')
plt.ylabel('F Value')
plt.title('Progress of F over Generations across 30 Runs')
plt.grid(True)
plt.legend(loc='upper right')

# Display the plot
plt.show()

# %%
best_model = max(best_models, key=lambda x: x['fitness'])
best_model['fitness']

# %%
print(best_model['individual'])

# %%
model_dict = decode_model_architecture(best_model['individual'])
model_dict

# %%
cnn = build_tf_model_from_dict(model_dict)
cnn

# %%
cnn.save('/drive/MyDrive/NAS/EC_Project/best_model_mu100_gens1000_F0p5_5000mu_autoadapt.h5')

# %%
cnn.save('/drive/MyDrive/NAS/EC_Project/best_model_mu100_gens1000_F0p5_5000mu_autoadapt.keras')


