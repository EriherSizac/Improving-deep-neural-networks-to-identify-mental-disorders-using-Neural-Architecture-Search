from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, DepthwiseConv2D
import tensorflow as tf
import numpy as np

# Opciones de decodificación
layer_type_options = {
    0: 'Conv2D', 
    1: 'BatchNorm', 
    2: 'MaxPooling', 
    3: 'Dropout', 
    4: 'Dense', 
    5: 'Flatten',
    6: 'DepthwiseConv2D',  
    7: 'DontCare',  
    8: 'Repetition'
}
stride_options = {0: 1, 1: 2}
dropout_options = {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.5}
activation_options = {0: 'relu', 1: 'leaky_relu', 2: 'sigmoid', 3: 'tanh'}

# Codificación de capa
def encode_layer_params(layer_type_idx, param1=0, param2=0, param3=0):
    return [layer_type_idx, param1, param2, param3]

# Decodificación de capa
def decode_layer_params(encoded_params):
    layer_type_idx = encoded_params[0]
    layer_type = layer_type_options.get(layer_type_idx, 'DontCare')
    if layer_type in ['Conv2D', 'DepthwiseConv2D']:
        filters = max(4, min(encoded_params[1], 32))
        strides = stride_options.get(encoded_params[2], 1)
        activation = activation_options.get(encoded_params[3], 'relu')
        return {'type': layer_type, 'filters': filters, 'strides': strides, 'activation': activation}
    elif layer_type == 'Dense':
        units = max(1, min(encoded_params[1], 512))
        activation = activation_options.get(encoded_params[2], 'relu')
        return {'type': 'Dense', 'units': units, 'activation': activation}
    elif layer_type == 'Dropout':
        rate = dropout_options.get(encoded_params[1], 0.2)
        return {'type': 'Dropout', 'rate': rate}
    elif layer_type in ['Flatten', 'BatchNorm', 'MaxPooling', 'DontCare']:
        return {'type': layer_type}
    elif layer_type == 'Repetition':
        return {'type': 'Repetition', 'repetition_layers': int(encoded_params[1]), 'repetition_count': int(encoded_params[2])}
    return None

# Reparación de arquitectura
def fixArch(encoded_model, max_alleles=48, verbose=False):
    fixed_layers = []
    input_is_flattened = False
    index = 0
    while index < len(encoded_model) and len(fixed_layers) < max_alleles:
        layer_type = int(encoded_model[index])
        if layer_type == 5:  # Flatten
            if not input_is_flattened:
                fixed_layers.extend([layer_type, 0, 0, 0])
                input_is_flattened = True
            else:
                fixed_layers.extend([7, 0, 0, 0])
        elif layer_type == 4:  # Dense
            if not input_is_flattened:
                fixed_layers.extend([7, 0, 0, 0])
            else:
                fixed_layers.extend([layer_type, max(1, min(encoded_model[index + 1], 512)), encoded_model[index + 2], 0])
        else:
            fixed_layers.extend(encoded_model[index:index + 4])
        index += 4
    return fixed_layers[:max_alleles]

# Decodificar arquitectura completa
def decode_model_architecture(encoded_model):
    model_dict = {'layers': [{'type': 'Conv2D', 'filters': 32, 'strides': 1, 'activation': 'relu'}]}
    index = 0
    while index < len(encoded_model):
        layer_type = int(encoded_model[index])
        param1, param2, param3 = encoded_model[index + 1:index + 4]
        decoded_layer = decode_layer_params([layer_type, param1, param2, param3])
        if decoded_layer:
            model_dict['layers'].append(decoded_layer)
        index += 4
    return model_dict

# Construir modelo en TensorFlow
def build_tf_model_from_dict(model_dict, input_shape=(28, 28, 3)):
    model = Sequential([tf.keras.Input(shape=input_shape)])
    for layer in model_dict['layers']:
        if layer['type'] == 'Conv2D':
            model.add(Conv2D(filters=layer['filters'], kernel_size=(3, 3), strides=layer['strides'], padding='same', activation=layer['activation']))
        elif layer['type'] == 'Dense':
            model.add(Dense(units=layer['units'], activation=layer['activation']))
        elif layer['type'] == 'Dropout':
            model.add(Dropout(rate=layer['rate']))
        elif layer['type'] == 'Flatten':
            model.add(Flatten())
        elif layer['type'] == 'BatchNorm':
            model.add(BatchNormalization())
    return model
