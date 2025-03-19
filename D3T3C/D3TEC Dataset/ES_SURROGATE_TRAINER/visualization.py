"""
Visualization Module

Este módulo contiene funciones para visualizar resultados de modelos surrogate,
incluyendo gráficos de predicciones y evolución de fitness.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_predictions(predictions, y_test, metrics):
    """
    Genera gráficos individuales por modelo comparando predicciones vs valores reales.
    
    Args:
        predictions: Diccionario con predicciones de cada modelo
        y_test: Valores reales de prueba
        metrics: Diccionario con métricas (MSE, MAE, MAPE) de cada modelo
    """
    for model_name, y_pred in predictions.items():
        mse_val, mae_val, mape_val = metrics[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, label=model_name)
        ax.plot(
            [min(y_test.values), max(y_test.values)],
            [min(y_test.values), max(y_test.values)],
            'r--', label="Ideal"
        )
        ax.set_xlabel("F1 Score Real")
        ax.set_ylabel("F1 Score Predicho")
        ax.set_title(f"{model_name}\nMSE: {mse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_synflow_correlation(data, corr, p_value):
    """
    Genera un gráfico de dispersión con tendencia para visualizar la correlación entre SynFlow y F1 Score.
    
    Args:
        data: DataFrame con columnas 'SynFlow' y 'F1'
        corr: Coeficiente de correlación de Pearson
        p_value: Valor p de la correlación
    """
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=data['SynFlow'], y=data['F1'], alpha=0.5)
    sns.regplot(x=data['SynFlow'], y=data['F1'], scatter=False, color='red', label='Tendencia')
    plt.xlabel("SynFlow")
    plt.ylabel("F1 Score")
    plt.title(f"Correlación SynFlow vs. F1 Score\nPearson: {corr:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_es_convergence(all_best_fitness):
    """
    Visualiza la convergencia del algoritmo de evolución estratégica a lo largo de múltiples ejecuciones.
    
    Args:
        all_best_fitness: Lista de listas con el mejor fitness por generación para cada ejecución
    """
    # Determinar el número máximo de generaciones
    max_generations = max(len(fitness) for fitness in all_best_fitness)
    generations = np.arange(1, max_generations + 1)
    
    # Inicializar matriz para almacenar valores de fitness
    fitness_matrix = np.full((len(all_best_fitness), max_generations), np.nan)
    
    for i, best_fitness_per_gen in enumerate(all_best_fitness):
        fitness_length = len(best_fitness_per_gen)
        fitness_matrix[i, :fitness_length] = best_fitness_per_gen
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    # Graficar todas las ejecuciones en gris claro
    for i in range(len(all_best_fitness)):
        plt.plot(
            generations,
            fitness_matrix[i],
            linestyle='-',
            color='red',
            alpha=0.5
        )
    
    # Calcular y graficar media y desviación estándar
    mean_fitness = np.nanmean(fitness_matrix, axis=0)
    std_fitness = np.nanstd(fitness_matrix, axis=0)
    
    plt.plot(
        generations,
        mean_fitness,
        linestyle='-',
        color='blue',
        label='Media de Fitness'
    )
    
    plt.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        color='blue',
        alpha=0.2,
        label='Desviación Estándar'
    )
    
    plt.title('Convergencia de Fitness por Generación')
    plt.xlabel('Generaciones')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
