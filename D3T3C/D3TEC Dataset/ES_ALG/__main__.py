import argparse
import json
import os
import sys

# Ajustar la ruta para importaciones relativas
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from ES_ALG.search import unified_search
from ES_ALG.utils.visualization import plot_fitness_history, plot_f_adaptation

def setup_argparse():
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
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Run unified search with surrogate model
    best_model, fitness_history, f_history = unified_search(
        surrogate_model_path=args.surrogate_model,
        population_size=args.population_size,
        gens=args.generations,
        auto_adapt=args.auto_adapt,
        n_window=args.n_window,
        n_experiments=args.n_experiments,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save results and visualizations
    with open(args.output, 'w') as f:
        json.dump({'individual': best_model['individual'], 'fitness': float(best_model['fitness'])}, f, indent=2)
    
    plot_fitness_history(fitness_history)
    plot_f_adaptation(f_history, args.n_window)
    
    print(f"Best architecture saved to {args.output}")

if __name__ == '__main__':
    main()