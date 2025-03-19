import torch
import torch.nn as nn
from .layers import DontCareLayer, SelfAttention
from ..utils.encoding import decode_model_architecture

class BuildPyTorchModel(nn.Module):
    def __init__(self, model_dict, input_shape=(1, 64, 552), verbose=False):
        """
        Construye un modelo de PyTorch a partir de un diccionario de arquitectura.
        """
        super(BuildPyTorchModel, self).__init__()
        self.verbose = verbose
        model_dict = decode_model_architecture(model_dict)
        
        if self.verbose:
            print(model_dict)

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
        """Procesa solo las características sin las capas completamente conectadas."""
        for module in self.feature_extractor:
            x = module(x)
        return x

    def forward(self, x):
        """Proceso completo de forward pass."""
        if self.initial_conv is not None:
            x = self.initial_conv(x)
            
        for i, module in enumerate(self.feature_extractor):
            # Ajuste dinámico de BatchNorm (según si la entrada es 2D o 4D)
            if isinstance(module, nn.BatchNorm2d):
                if x.dim() == 2:  # (batch, features)
                    num_features = x.shape[1]
                    if self.verbose:
                        print(f"⚠️ Reemplazando BatchNorm2d por BatchNorm1d para entrada con forma {x.shape}")
                    self.feature_extractor[i] = nn.BatchNorm1d(num_features).to(x.device)
                    module = self.feature_extractor[i]
                else:
                    num_channels = x.shape[1]
                    if module.num_features != num_channels:
                        if self.verbose:
                            print(f"⚠️ Ajustando BatchNorm2d: esperaba {module.num_features} canales, pero recibió {num_channels}")
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