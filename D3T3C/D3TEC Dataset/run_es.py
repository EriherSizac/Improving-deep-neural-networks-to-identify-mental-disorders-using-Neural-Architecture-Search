import sys
import os
import argparse
from ES_ALG.search import unified_search, load_surrogate_model
from ES_ALG.utils.visualization import plot_fitness_history, plot_f_adaptation, plot_top_architectures
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Run Evolutionary Strategy for Neural Architecture Search using surrogate models'
    )
    parser.add_argument(
        'surrogate_model',
        type=str,
        help='Path to the surrogate model file (.h5)'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=50,
        help='Population size for evolutionary search'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Number of generations for evolutionary search'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./best_architecture.json',
        help='Output file to save the best architecture found'
    )
    parser.add_argument(
        '--auto-adapt',
        action='store_true',
        help='Enable auto-adaptation of mutation factor F'
    )
    parser.add_argument(
        '--n-window',
        type=int,
        default=10,
        help='Window size for F adaptation'
    )
    parser.add_argument(
        '--n-experiments',
        type=int,
        default=30,
        help='Number of experiments to run'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save experiment checkpoints'
    )
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Only visualize results from previous runs without running new experiments'
    )
    
    args = parser.parse_args()
    
    # Crear directorio de checkpoints si no existe
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.visualize_only:
        # Solo visualizar resultados de ejecuciones anteriores
        print("Modo de visualización: cargando resultados previos...")
        
        # Verificar si existe el archivo de resultados finales
        final_results_path = os.path.join(args.checkpoint_dir, 'best_architectures.json')
        if os.path.exists(final_results_path):
            with open(final_results_path, 'r') as f:
                final_results = json.load(f)
            
            # Visualizar top arquitecturas
            plot_top_architectures(final_results['experiments'], 
                                  title=f"Top Arquitecturas ({len(final_results['experiments'])} experimentos)",
                                  save_path=os.path.join(args.checkpoint_dir, 'top_architectures.png'))
            
            print(f"Mejor arquitectura global: {final_results['best_model']['fitness']}")
            print(f"Gráfico guardado en {os.path.join(args.checkpoint_dir, 'top_architectures.png')}")
        else:
            print(f"No se encontró el archivo de resultados finales en {final_results_path}")
            
        # Cargar historiales de fitness y F de experimentos individuales
        all_fitness_histories = []
        all_f_histories = []
        
        for i in range(1, 100):  # Buscar hasta 100 experimentos
            exp_path = os.path.join(args.checkpoint_dir, f'experiment_{i}_top3.json')
            if os.path.exists(exp_path):
                with open(exp_path, 'r') as f:
                    exp_data = json.load(f)
                all_fitness_histories.append(exp_data['fitness_history'])
                all_f_histories.append(exp_data['F_history'])
            else:
                if i > 1:  # Si al menos encontramos un experimento
                    break
        
        if all_fitness_histories:
            # Visualizar historiales
            plot_fitness_history(all_fitness_histories, 
                               title="Evolución del Fitness",
                               save_path=os.path.join(args.checkpoint_dir, 'fitness_history.png'))
            
            plot_f_adaptation(all_f_histories, 
                            n_window=args.n_window,
                            title="Adaptación del parámetro F",
                            save_path=os.path.join(args.checkpoint_dir, 'f_adaptation.png'))
            
            print(f"Gráficos guardados en {args.checkpoint_dir}")
        else:
            print("No se encontraron datos de experimentos previos")
        
        return
    
    # Ejecutar búsqueda evolutiva
    print(f"Ejecutando búsqueda con:")
    print(f"  - Modelo surrogate: {args.surrogate_model}")
    print(f"  - Tamaño de población: {args.population_size}")
    print(f"  - Generaciones: {args.generations}")
    print(f"  - Archivo de salida: {args.output}")
    print(f"  - Auto-adaptación: {args.auto_adapt}")
    print(f"  - Número de experimentos: {args.n_experiments}")
    
    # Cargar el modelo surrogate
    surrogate_model = load_surrogate_model(args.surrogate_model)
    
    # Ejecutar búsqueda unificada con modelo surrogate
    results = unified_search(
        surrogate_model=surrogate_model,
        population_size=args.population_size,
        generations=args.generations,
        n_experiments=args.n_experiments,
        F=0.5,
        cr_rate=0.5,
        auto_adaptation=args.auto_adapt,
        checkpoint_dir=args.checkpoint_dir
    )
    
    best_model = results['best_model']
    fitness_histories = results['all_fitness_histories']
    f_histories = results['all_F_histories']
    
    # Convertir numpy arrays a listas para JSON
    individual = best_model['individual']
    if isinstance(individual, np.ndarray):
        individual = individual.tolist()
    
    fitness = best_model['fitness']
    if isinstance(fitness, np.ndarray) and fitness.size == 1:
        fitness = float(fitness.item())  # Convertir correctamente a escalar
    elif isinstance(fitness, np.number):
        fitness = float(fitness)
    
    # Guardar resultados
    with open(args.output, 'w') as f:
        json.dump({
            'individual': individual,
            'fitness': fitness
        }, f, indent=2)
    
    # Cargar resultados finales para visualización
    final_results_path = os.path.join(args.checkpoint_dir, 'best_architectures.json')
    if os.path.exists(final_results_path):
        with open(final_results_path, 'r') as f:
            final_results = json.load(f)
        
        # Visualizar top arquitecturas
        plot_top_architectures(final_results['experiments'], 
                              title=f"Top Arquitecturas ({len(final_results['experiments'])} experimentos)",
                              save_path=os.path.join(args.checkpoint_dir, 'top_architectures.png'))
    
    # Visualizar historiales
    try:
        plot_fitness_history(fitness_histories, 
                           title="Evolución del Fitness",
                           save_path=os.path.join(args.checkpoint_dir, 'fitness_history.png'))
        
        plot_f_adaptation(f_histories, 
                        n_window=args.n_window,
                        title="Adaptación del parámetro F",
                        save_path=os.path.join(args.checkpoint_dir, 'f_adaptation.png'))
    except Exception as e:
        print(f"Error al generar gráficos: {e}")
    
    print(f"Mejor arquitectura guardada en {args.output}")
    print(f"Gráficos guardados en {args.checkpoint_dir}")

if __name__ == '__main__':
    main()
