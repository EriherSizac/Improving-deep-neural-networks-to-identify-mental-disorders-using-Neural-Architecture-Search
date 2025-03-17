import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_history(fitness_history, title="Fitness Evolution", save_path=None):
    """Plot the evolution of fitness across generations.

    Args:
        fitness_history (list or numpy.ndarray): List of fitness values per generation or list of lists for multiple experiments
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    # Asegurar que fitness_history es una lista o array
    if not isinstance(fitness_history, (list, np.ndarray)):
        print(f"Error: fitness_history debe ser una lista o array, pero es {type(fitness_history)}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Verificar si tenemos múltiples experimentos
    if isinstance(fitness_history[0], (list, np.ndarray)):
        # Múltiples experimentos
        n_experiments = len(fitness_history)
        print(f"Visualizando {n_experiments} experimentos")
        
        # Calcular estadísticas entre experimentos
        max_len = max(len(exp) for exp in fitness_history)
        aligned_history = np.zeros((n_experiments, max_len))
        
        for i, exp in enumerate(fitness_history):
            aligned_history[i, :len(exp)] = exp
            # Para generaciones no ejecutadas en experimentos más cortos, repetir el último valor
            if len(exp) < max_len:
                aligned_history[i, len(exp):] = exp[-1]
        
        # Calcular media y desviación estándar entre experimentos
        mean_fitness = np.mean(aligned_history, axis=0)
        std_fitness = np.std(aligned_history, axis=0)
        generations = np.arange(max_len)
        
        # Graficar media con banda de confianza
        plt.plot(generations, mean_fitness, 'b-', linewidth=2, label='Media entre experimentos')
        plt.fill_between(generations, 
                         mean_fitness - std_fitness,
                         mean_fitness + std_fitness,
                         alpha=0.2, color='blue')
        
        # Graficar cada experimento con líneas más finas
        for i, exp in enumerate(fitness_history):
            plt.plot(np.arange(len(exp)), exp, 'b-', alpha=0.5, linewidth=0.8)
        
        plt.title(f"{title} ({n_experiments} experimentos)")
    else:
        # Un solo experimento
        fitness_array = np.array(fitness_history)
        generations = np.arange(len(fitness_array))
        plt.plot(generations, fitness_array, 'b-', linewidth=2, label='Fitness')
        plt.title(f"{title} (1 experimento)")
    
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_f_adaptation(f_history, n_window=1, title="F Parameter Adaptation", save_path=None):
    """Plot the adaptation of the F parameter over generations.

    Args:
        f_history (list or numpy.ndarray): History of F values or list of lists for multiple experiments
        n_window (int): Window size used for adaptation
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    # Asegurar que f_history es una lista o array
    if not isinstance(f_history, (list, np.ndarray)):
        print(f"Error: f_history debe ser una lista o array, pero es {type(f_history)}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Verificar si tenemos múltiples experimentos
    if isinstance(f_history[0], (list, np.ndarray)):
        # Múltiples experimentos
        n_experiments = len(f_history)
        print(f"Visualizando {n_experiments} experimentos")
        
        # Calcular estadísticas entre experimentos
        max_len = max(len(exp) for exp in f_history)
        aligned_history = np.zeros((n_experiments, max_len))
        
        for i, exp in enumerate(f_history):
            aligned_history[i, :len(exp)] = exp
            # Para generaciones no ejecutadas en experimentos más cortos, repetir el último valor
            if len(exp) < max_len:
                aligned_history[i, len(exp):] = exp[-1]
        
        # Calcular media y desviación estándar entre experimentos
        mean_f = np.mean(aligned_history, axis=0)
        std_f = np.std(aligned_history, axis=0)
        generations = np.arange(max_len) * n_window
        
        # Graficar media con banda de confianza
        plt.plot(generations, mean_f, 'r-', linewidth=2, label='Media entre experimentos')
        plt.fill_between(generations, 
                         mean_f - std_f,
                         mean_f + std_f,
                         alpha=0.2, color='red')
        
        # Graficar cada experimento con líneas más finas
        for i, exp in enumerate(f_history):
            plt.plot(np.arange(len(exp)) * n_window, exp, 'k-', alpha=0.3, linewidth=0.5)
        
        plt.title(f"{title} ({n_experiments} experimentos)")
    else:
        # Un solo experimento
        f_array = np.array(f_history)
        generations = np.arange(len(f_array)) * n_window
        plt.plot(generations, f_array, 'r-', linewidth=2, label='Valor F')
        plt.title(f"{title} (1 experimento)")
    
    plt.xlabel('Generación')
    plt.ylabel('Parámetro F')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_top_architectures(top_models, title="Top Architectures", save_path=None):
    """Plot the fitness of top architectures from multiple experiments.

    Args:
        top_models (dict): Dictionary with experiment data containing top models
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    if not top_models:
        print("Error: No hay modelos para visualizar")
        return
    
    # Extraer datos
    experiment_ids = []
    top1_fitness = []
    top2_fitness = []
    top3_fitness = []
    
    for exp_key, exp_data in top_models.items():
        # Extraer el número de experimento del nombre de la clave
        exp_num = exp_key.split('_')[-1]
        experiment_ids.append(exp_num)
        
        models = exp_data['top_3_models']
        
        if len(models) >= 1:
            top1_fitness.append(models[0]['fitness'])
        else:
            top1_fitness.append(0)
            
        if len(models) >= 2:
            top2_fitness.append(models[1]['fitness'])
        else:
            top2_fitness.append(0)
            
        if len(models) >= 3:
            top3_fitness.append(models[2]['fitness'])
        else:
            top3_fitness.append(0)
    
    # Crear gráfico de barras
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.25
    index = np.arange(len(experiment_ids))
    
    plt.bar(index, top1_fitness, bar_width, label='Top 1', color='gold')
    plt.bar(index + bar_width, top2_fitness, bar_width, label='Top 2', color='silver')
    plt.bar(index + 2*bar_width, top3_fitness, bar_width, label='Top 3', color='#CD7F32')  # Bronze color
    
    plt.xlabel('Experimento')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.xticks(index + bar_width, experiment_ids)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Añadir línea horizontal con el promedio del top 1
    mean_top1 = np.mean(top1_fitness)
    plt.axhline(y=mean_top1, color='r', linestyle='--', alpha=0.8, 
                label=f'Media Top 1: {mean_top1:.4f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()