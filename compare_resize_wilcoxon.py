import json
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata
import Orange
from Orange.evaluation import compute_CD, graph_ranks

# Función auxiliar para graficar forzando índices enteros y mostrando logs
def my_graph_ranks(avranks, cd, names, width=10, textspace=1.5, title=""):
    # Convertir avranks a lista de floats nativos
    avranks = list(map(float, avranks))
    print("avranks:", avranks)
    # Calcular índices ordenados basados en avranks
    sortidx = sorted(range(len(avranks)), key=lambda i: avranks[i])
    print("sortidx (raw):", sortidx)
    # Forzar que cada índice sea entero
    sortidx = [int(x) for x in sortidx]
    print("sortidx (int):", sortidx)
    try:
        nnames = [names[x] for x in sortidx]
    except Exception as e:
        print("Error al indexar names con sortidx:", e)
        print("names:", names)
        print("sortidx:", sortidx)
        raise e
    print("nnames:", nnames)
    cd = float(cd)
    print("cd:", cd)
    # Llamar a graph_ranks con el orden correcto: avranks, names, cd, ...
    graph_ranks(avranks, nnames, cd, width=width, textspace=textspace, title=title)

# Función para visualizar comparación entre resize=True y resize=False
def plot_comparison(data_true, data_false, metric_name, higher_is_better=True, config_info=None):
    """
    Visualiza la comparación entre resize=True y resize=False para una métrica específica.
    
    data_true: lista de valores para resize=True
    data_false: lista de valores para resize=False
    metric_name: nombre de la métrica
    higher_is_better: True si mayor es mejor; False si menor es mejor
    config_info: información adicional para incluir en el título
    """
    # Convertir a arrays numpy
    data_true = np.array(data_true)
    data_false = np.array(data_false)
    
    # Calcular diferencias para el test de Wilcoxon
    differences = data_true - data_false
    
    # Realizar test de Wilcoxon
    try:
        stat, pvalue = wilcoxon(differences)
        print(f"Wilcoxon test for {metric_name} ({config_info}): statistic = {stat:.4f}, p-value = {pvalue:.4f}")
        
        # Determinar qué configuración es mejor según la métrica
        if higher_is_better:
            better_config = "Resize=True" if np.mean(data_true) > np.mean(data_false) else "Resize=False"
        else:
            better_config = "Resize=True" if np.mean(data_true) < np.mean(data_false) else "Resize=False"
        
        print(f"Mean {metric_name} for Resize=True: {np.mean(data_true):.4f}")
        print(f"Mean {metric_name} for Resize=False: {np.mean(data_false):.4f}")
        print(f"Better configuration: {better_config}")
        
        # Visualizar con boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot([data_true, data_false], labels=["Resize=True", "Resize=False"])
        plt.title(f"{metric_name} comparison ({config_info})\np-value: {pvalue:.4f}")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir indicador de significancia estadística
        if pvalue < 0.05:
            plt.text(1.5, max(np.max(data_true), np.max(data_false)) * 1.05, 
                    "* Significant difference (p<0.05)", 
                    horizontalalignment='center',
                    color='red')
        
        plt.show()
    except ValueError as e:
        print(f"Error en el test de Wilcoxon para {metric_name}: {e}")
        print("Esto puede ocurrir si todas las diferencias son cero o hay muy pocas muestras.")

# -------------------------------------------------------------------
# Cargar el archivo de resultados
with open('./experiment_results.json', 'r') as f:
    results = json.load(f)

# Agrupar resultados por modelo, splits y epochs, separando por resize
def organize_data_by_resize():
    organized = {}
    for key, metrics in results.items():
        # Ejemplo: CNN_LF_splits_5_epochs_50_resize_True
        m = re.match(r"(.+?)_splits_(\d+)_epochs_(\d+)_resize_(True|False)", key)
        if m:
            architecture, splits, epochs, resize = m.groups()
            splits = int(splits)
            epochs = int(epochs)
            config = f"{architecture}_splits_{splits}_epochs_{epochs}"
            
            if config not in organized:
                organized[config] = {"True": None, "False": None}
            
            organized[config][resize] = metrics
    
    return organized

# Organizar los datos
organized_data = organize_data_by_resize()

# Verificar configuraciones disponibles
print("\n==== Configuraciones disponibles ====")
for config, resize_data in organized_data.items():
    has_true = resize_data["True"] is not None
    has_false = resize_data["False"] is not None
    if has_true and has_false:
        print(f"Configuración completa: {config}")
    else:
        missing = "Resize=True" if not has_true else "Resize=False"
        print(f"Configuración incompleta: {config} (falta {missing})")

# Filtrar solo configuraciones que tienen ambos resize=True y resize=False
complete_configs = {config: data for config, data in organized_data.items() 
                   if data["True"] is not None and data["False"] is not None}

# Especificar para cada métrica si mayor es mejor
metric_higher = {
    "Loss": False,
    "Accuracy": True,
    "Precision": True,
    "Recall": True,
    "F1-score": True,
    "Specificity": True
}

# Agrupar por modelo para análisis más detallado
models = {}
for config in complete_configs:
    model = config.split("_splits_")[0]
    if model not in models:
        models[model] = []
    models[model].append(config)

print("\n==== Modelos disponibles ====")
for model, configs in models.items():
    print(f"Modelo: {model}, Configuraciones: {len(configs)}")

# Realizar prueba de Wilcoxon para cada métrica, comparando resize=True vs resize=False
# Primero, para todas las configuraciones juntas
print("\n==== Análisis global: Todas las configuraciones ====")
for metric in metric_higher.keys():
    # Preparar datos para la prueba de Wilcoxon
    data_true = []
    data_false = []
    
    for config in complete_configs:
        data_true.append(complete_configs[config]["True"][metric])
        data_false.append(complete_configs[config]["False"][metric])
    
    # Realizar prueba de Wilcoxon y visualizar
    plot_comparison(data_true, data_false, 
                   metric_name=metric, 
                   higher_is_better=metric_higher[metric],
                   config_info="All Configurations")

# Luego, análisis por modelo
for model, configs in models.items():
    if len(configs) >= 2:  # Necesitamos al menos 2 configuraciones para un análisis significativo
        print(f"\n==== Análisis por modelo: {model} ====")
        for metric in metric_higher.keys():
            # Preparar datos para la prueba de Wilcoxon
            data_true = []
            data_false = []
            
            for config in configs:
                data_true.append(complete_configs[config]["True"][metric])
                data_false.append(complete_configs[config]["False"][metric])
            
            # Realizar prueba de Wilcoxon y visualizar
            plot_comparison(data_true, data_false, 
                           metric_name=metric, 
                           higher_is_better=metric_higher[metric],
                           config_info=f"Model {model}")

# Análisis por número de epochs
epochs_values = sorted(set([int(config.split("_epochs_")[1]) for config in complete_configs]))
for epochs in epochs_values:
    configs_with_epochs = [config for config in complete_configs if f"_epochs_{epochs}" in config]
    if len(configs_with_epochs) >= 2:
        print(f"\n==== Análisis por epochs: {epochs} ====")
        for metric in metric_higher.keys():
            # Preparar datos para la prueba de Wilcoxon
            data_true = []
            data_false = []
            
            for config in configs_with_epochs:
                data_true.append(complete_configs[config]["True"][metric])
                data_false.append(complete_configs[config]["False"][metric])
            
            # Realizar prueba de Wilcoxon y visualizar
            plot_comparison(data_true, data_false, 
                           metric_name=metric, 
                           higher_is_better=metric_higher[metric],
                           config_info=f"Epochs {epochs}")

# Análisis por número de splits
splits_values = sorted(set([int(config.split("_splits_")[1].split("_epochs_")[0]) for config in complete_configs]))
for splits in splits_values:
    configs_with_splits = [config for config in complete_configs if f"_splits_{splits}_" in config]
    if len(configs_with_splits) >= 2:
        print(f"\n==== Análisis por splits: {splits} ====")
        for metric in metric_higher.keys():
            # Preparar datos para la prueba de Wilcoxon
            data_true = []
            data_false = []
            
            for config in configs_with_splits:
                data_true.append(complete_configs[config]["True"][metric])
                data_false.append(complete_configs[config]["False"][metric])
            
            # Realizar prueba de Wilcoxon y visualizar
            plot_comparison(data_true, data_false, 
                           metric_name=metric, 
                           higher_is_better=metric_higher[metric],
                           config_info=f"Splits {splits}")
