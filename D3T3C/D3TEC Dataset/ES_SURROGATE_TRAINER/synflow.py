"""
SynFlow Module

Este módulo implementa la métrica SynFlow para evaluar arquitecturas neuronales
sin necesidad de entrenamiento, basándose en el flujo de señal a través de la red.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Importar funciones necesarias del módulo de arquitectura
from architecture import fixArch, BuildPyTorchModel

def compute_synflow(model):
    """
    Calcula la métrica SynFlow asegurando que todos los parámetros tengan gradientes.
    
    Args:
        model: Modelo PyTorch para evaluar
        
    Returns:
        float: Puntuación SynFlow
    """
    model.eval()
    
    # Habilitar gradientes para todos los parámetros
    for p in model.parameters():
        p.requires_grad = True
    
    # Crear tensor de entrada (todos unos)
    input_shape = (1, 1, 64, 552)  # Dimensión de entrada (1 canal, 64x552)
    input_data = torch.ones(input_shape, requires_grad=True)
    
    # Calcular SynFlow
    try:
        output = model(input_data)  # Pasar el input por la red
        loss = output.sum()  # Generar una suma de salida
        loss.backward(retain_graph=True)  # Retropropagación para obtener los gradientes
        
        # Calcular SynFlow sumando |gradiente * peso| en cada parámetro
        synflow_score = sum(
            (p.grad * p).abs().sum().item() if p.grad is not None else 0 
            for p in model.parameters()
        )
        
        return synflow_score
    except Exception as e:
        print(f"⚠️ Error en el cálculo de SynFlow: {e}")
        return 0  # Si el modelo falla, devolver SynFlow = 0

def calculate_synflow_scores(file_path):
    """
    Calcula SynFlow para cada arquitectura en el archivo CSV y guarda los resultados.
    
    Args:
        file_path: Ruta al archivo CSV con arquitecturas codificadas
        
    Returns:
        DataFrame: Datos originales con columna SynFlow añadida
    """
    # Cargar datos
    data = pd.read_csv(file_path)
    
    # Convertir 'Encoded Architecture' en listas de enteros
    data['Encoded Architecture'] = data['Encoded Architecture'].apply(eval)
    
    # Calcular SynFlow para cada arquitectura
    print(f"📊 Calculando SynFlow para {len(data)} arquitecturas...")
    synflow_values = []
    
    for i, arch in enumerate(data['Encoded Architecture']):
        if i % 50 == 0:
            print(f"   Progreso: {i}/{len(data)} arquitecturas")
            
        # Aplicar correcciones a la arquitectura
        encoded_fixed_arch = fixArch(arch)
        
        # Construir el modelo en PyTorch
        pytorch_model = BuildPyTorchModel(encoded_fixed_arch)
        
        # Calcular SynFlow
        synflow_score = compute_synflow(pytorch_model)
        synflow_values.append(synflow_score)
    
    # Agregar los valores de SynFlow al DataFrame
    data['SynFlow'] = synflow_values
    
    # Guardar el nuevo CSV con SynFlow incluido
    output_path = "EncodedChromosomes_with_SynFlow.csv"
    data.to_csv(output_path, index=False)
    print(f"✅ SynFlow calculado y guardado en '{output_path}'")
    
    return data

def plot_synflow_correlation(data, file_path=None):
    """
    Analiza y visualiza la correlación entre SynFlow y F1 Score.
    
    Args:
        data: DataFrame con columnas 'SynFlow' y 'F1', o None para cargar desde file_path
        file_path: Ruta al archivo CSV con datos de SynFlow y F1 Score
        
    Returns:
        tuple: Coeficiente de correlación y valor p
    """
    if data is None and file_path:
        data = pd.read_csv(file_path)
        
    # Verificar si SynFlow está en los datos
    if 'SynFlow' not in data.columns:
        raise ValueError("⚠️ La columna 'SynFlow' no está en el DataFrame.")
    
    # Calcular la correlación de Pearson entre SynFlow y F1 Score
    corr, p_value = pearsonr(data['SynFlow'], data['F1'])
    
    print(f"📊 Correlación de Pearson entre SynFlow y F1 Score: {corr:.4f}")
    print(f"📊 Valor p: {p_value:.4f}")
    
    # Generar gráfico de dispersión con tendencia
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=data['SynFlow'], y=data['F1'], alpha=0.5)
    sns.regplot(x=data['SynFlow'], y=data['F1'], scatter=False, color='red', label='Tendencia')
    plt.xlabel("SynFlow")
    plt.ylabel("F1 Score")
    plt.title(f"Correlación SynFlow vs. F1 Score\nPearson: {corr:.4f}")
    plt.legend()
    plt.show()
    
    return corr, p_value
