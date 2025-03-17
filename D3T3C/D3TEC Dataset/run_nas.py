#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar la búsqueda de arquitectura neuronal (NAS) con sistema de checkpoints.
Permite iniciar una nueva búsqueda o continuar desde el último checkpoint.

Uso:
    python run_nas.py [ruta_modelo_surrogate] [opciones]

Argumentos:
    ruta_modelo_surrogate: Ruta al modelo surrogate a utilizar (opcional)
    --new-run: Si se especifica, inicia una nueva búsqueda ignorando checkpoints existentes.
               Si no se especifica, intenta reanudar desde el último checkpoint.
    --population-size: Tamaño de la población (default: 20)
    --generations: Número de generaciones (default: 100)
    --n-experiments: Número de experimentos (default: 5)
    --checkpoint-dir: Directorio para guardar los checkpoints (default: ./checkpoints)
    --auto-adaptation: Habilitar adaptación automática del factor F (default: True)
"""

import argparse
import os
import sys
from ES_ALG.search import unified_search, load_surrogate_model, find_latest_checkpoint

def main():
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Ejecutar búsqueda de arquitectura neuronal')
    parser.add_argument('surrogate_model_path', nargs='?', default="./DeepNN_model.h5",
                        help='Ruta al modelo surrogate')
    parser.add_argument('--new-run', action='store_true', default=False,
                        help='Iniciar una nueva búsqueda ignorando checkpoints existentes')
    parser.add_argument('--population-size', type=int, default=20,
                        help='Tamaño de la población (default: 20)')
    parser.add_argument('--generations', type=int, default=100,
                        help='Número de generaciones (default: 100)')
    parser.add_argument('--n-experiments', type=int, default=5,
                        help='Número de experimentos (default: 5)')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                        help='Directorio para guardar los checkpoints (default: ./checkpoints)')
    parser.add_argument('--auto-adaptation', action='store_true', default=True,
                        help='Habilitar adaptación automática del factor F (default: True)')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Configurar parámetros de búsqueda
    checkpoint_dir = args.checkpoint_dir
    population_size = args.population_size
    generations = args.generations
    n_experiments = args.n_experiments
    surrogate_model_path = args.surrogate_model_path
    
    print(f"Configuración:")
    print(f"- Modelo surrogate: {surrogate_model_path}")
    print(f"- Tamaño de población: {population_size}")
    print(f"- Generaciones: {generations}")
    print(f"- Experimentos: {n_experiments}")
    print(f"- Directorio de checkpoints: {checkpoint_dir}")
    print(f"- Auto-adaptación: {args.auto_adaptation}")
    print(f"- Nuevo inicio: {args.new_run}")
    
    # Cargar el modelo surrogate
    print("\nCargando modelo surrogate...")
    surrogate_model = load_surrogate_model(surrogate_model_path)
    
    if not surrogate_model:
        print(f"Error: No se pudo cargar el modelo surrogate en {surrogate_model_path}.")
        sys.exit(1)
    
    # Ejecutar la búsqueda
    if args.new_run:
        print("Iniciando nueva búsqueda...")
        results = unified_search(
            surrogate_model=surrogate_model,
            population_size=population_size,
            generations=generations,
            n_experiments=n_experiments,
            auto_adaptation=args.auto_adaptation,
            checkpoint_dir=checkpoint_dir,
            new_run=True
        )
    else:
        print("Buscando el último checkpoint para reanudar...")
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint:
            print(f"Reanudando desde checkpoint: {latest_checkpoint}")
            results = unified_search(
                surrogate_model=surrogate_model,
                population_size=population_size,
                generations=generations,
                n_experiments=n_experiments,
                auto_adaptation=args.auto_adaptation,
                checkpoint_dir=checkpoint_dir,
                resume_from=latest_checkpoint,
                new_run=False
            )
        else:
            print("No se encontraron checkpoints anteriores. Iniciando nueva búsqueda...")
            results = unified_search(
                surrogate_model=surrogate_model,
                population_size=population_size,
                generations=generations,
                n_experiments=n_experiments,
                auto_adaptation=args.auto_adaptation,
                checkpoint_dir=checkpoint_dir,
                new_run=True
            )
    
    print("\nBúsqueda completada.")
    print(f"Mejor fitness general: {results['best_model']['fitness']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
