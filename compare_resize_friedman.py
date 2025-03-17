import json
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import rankdata, friedmanchisquare
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

# Función para calcular y plotear el diagrama CD para una métrica, e imprimir el p-value de Friedman
def plot_cd_for_metric(data, group_names, metric_name, higher_is_better=True, config_info=None):
    """
    data: matriz de shape (n_replicates, n_groups) – cada fila es una configuración.
    group_names: lista de nombres (strings) para cada grupo (ej.: ["Resize True", "Resize False"]).
    higher_is_better: True si mayor es mejor; False si menor es mejor.
    config_info: información adicional para incluir en el título.
    """
    if higher_is_better:
        ranks = np.array([rankdata(-row) for row in data])
    else:
        ranks = np.array([rankdata(row) for row in data])
    avg_ranks = np.mean(ranks, axis=0)
    avranks = avg_ranks.tolist()
    n_replicates = data.shape[0]
    cd = compute_CD(avranks, n_replicates, alpha="0.05", test="nemenyi")
    
    # Calcular el p-value de Friedman (cada columna es un grupo)
    friedman_stat, pvalue = friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    print(f"Friedman test for {metric_name} ({config_info}): statistic = {friedman_stat:.4f}, p-value = {pvalue:.4f}")
    
    title_str = f"Critical Difference Diagram for {metric_name}"
    if config_info is not None:
        title_str += f" ({config_info})"
    print(f"\n--- Plotting CD diagram for {metric_name} ({config_info}) ---")
    print("Group names:", group_names)
    my_graph_ranks(avranks, cd, group_names, width=10, textspace=1.5, title=title_str)
    plt.show()

# -------------------------------------------------------------------
# Cargar el archivo de resultados
with open('D3T3C/D3TEC Dataset/old_surr/experiment_results.json', 'r') as f:
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

# Realizar prueba de Friedman para cada métrica, comparando resize=True vs resize=False
# Primero, para todas las configuraciones juntas
print("\n==== Análisis global: Todas las configuraciones ====")
for metric in metric_higher.keys():
    # Preparar datos para la prueba de Friedman
    data_true = []
    data_false = []
    
    for config in complete_configs:
        data_true.append(complete_configs[config]["True"][metric])
        data_false.append(complete_configs[config]["False"][metric])
    
    # Convertir a array numpy
    data_matrix = np.column_stack((data_true, data_false))
    
    # Realizar prueba de Friedman y graficar diagrama CD
    plot_cd_for_metric(data_matrix, ["Resize True", "Resize False"], 
                       metric_name=metric, 
                       higher_is_better=metric_higher[metric],
                       config_info="All Configurations")

# Luego, análisis por modelo
for model, configs in models.items():
    if len(configs) >= 2:  # Necesitamos al menos 2 configuraciones para la prueba de Friedman
        print(f"\n==== Análisis por modelo: {model} ====")
        for metric in metric_higher.keys():
            # Preparar datos para la prueba de Friedman
            data_true = []
            data_false = []
            
            for config in configs:
                data_true.append(complete_configs[config]["True"][metric])
                data_false.append(complete_configs[config]["False"][metric])
            
            # Convertir a array numpy
            data_matrix = np.column_stack((data_true, data_false))
            
            # Realizar prueba de Friedman y graficar diagrama CD
            plot_cd_for_metric(data_matrix, ["Resize True", "Resize False"], 
                              metric_name=metric, 
                              higher_is_better=metric_higher[metric],
                              config_info=f"Model {model}")
