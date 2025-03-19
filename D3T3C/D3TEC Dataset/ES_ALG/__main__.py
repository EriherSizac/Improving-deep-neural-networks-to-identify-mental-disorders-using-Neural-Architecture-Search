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
    --population-size: Tamaño de la población (default: 500)
    --generations: Número de generaciones (default: 200)
    --n-experiments: Número de experimentos (default: 5)
    --checkpoint-dir: Directorio para guardar los checkpoints (default: ./checkpoints)
    --auto-adaptation: Habilitar adaptación automática del factor F (default: False)
    --selection-method: Método de selección: random (clásico DE), tournament o sus (Stochastic Universal Sampling) (default: random)
"""

import argparse
import os
import sys
from ES_ALG.search import unified_search, load_surrogate_model, find_latest_checkpoint
import json

def setup_argparse():
    """Configura el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Búsqueda de arquitectura neural usando Evolución Diferencial")
    
    # Argumentos obligatorios
    parser.add_argument("surrogate_model_path", type=str, help="Ruta al modelo surrogate (.h5 o .pkl)")
    
    # Argumentos opcionales
    parser.add_argument("--population-size", type=int, default=500, help="Tamaño de la población")
    parser.add_argument("--generations", type=int, default=200, help="Número de generaciones")
    parser.add_argument("--n-experiments", type=int, default=5, help="Número de experimentos")
    parser.add_argument("--f", type=float, default=0.5, help="Factor de mutación")
    parser.add_argument("--cr", type=float, default=0.5, help="Tasa de cruce")
    parser.add_argument("--auto-adapt", action="store_true", help="Habilitar auto-adaptación del factor F")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directorio para guardar checkpoints")
    parser.add_argument("--output", type=str, default="best_model.json", help="Archivo de salida para el mejor modelo")
    parser.add_argument("--tournament", action="store_true", help="Usar selección por torneo (obsoleto, usar --selection-method)")
    parser.add_argument("--selection-method", choices=["random", "tournament", "sus"], default="random",
                       help="Método de selección: random (clásico DE), tournament o sus (Stochastic Universal Sampling)")
    parser.add_argument("--new-run", action="store_true", help="Iniciar una nueva búsqueda, ignorando checkpoints existentes")
    parser.add_argument("--resume-from", type=str, help="Ruta a un checkpoint específico para reanudar la búsqueda")
    
    return parser

def main():
    parser = setup_argparse()
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
    print(f"- Auto-adaptación: {args.auto_adapt}")
    print(f"- Nuevo inicio: {args.new_run}")
    print(f"- Selección por torneo: {args.tournament}")
    print(f"- Método de selección: {args.selection_method}")
    
    # Cargar el modelo surrogate
    print("\nCargando modelo surrogate...")
    surrogate_model = load_surrogate_model(surrogate_model_path)
    
    if not surrogate_model:
        print(f"Error: No se pudo cargar el modelo surrogate en {surrogate_model_path}.")
        sys.exit(1)
    
    # Ejecutar la búsqueda
    resume_from = None  # Inicializar con valor por defecto
    
    if args.resume_from:
        resume_from = args.resume_from
    elif not args.new_run:
        # Buscar el checkpoint más reciente
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Encontrado checkpoint anterior: {latest_checkpoint}")
            resume_from = latest_checkpoint
        else:
            print("No se encontraron checkpoints anteriores. Iniciando nueva búsqueda...")
    else:
        print("Iniciando nueva búsqueda...")
    
    # Ejecutar la búsqueda unificada
    results = unified_search(
        surrogate_model=surrogate_model,
        population_size=population_size,
        generations=generations,
        n_experiments=n_experiments,
        auto_adaptation=args.auto_adapt,
        F=args.f,
        cr_rate=args.cr,
        checkpoint_dir=checkpoint_dir,
        new_run=args.new_run,
        selection_method=args.selection_method if not args.tournament else "tournament",
        resume_from=resume_from
    )
    
    # Extraer los resultados
    best_model = results.get('best_model', {})
    
    # Save results and visualizations
    with open(args.output, 'w') as f:
        json.dump({'individual': best_model.get('individual', []), 'fitness': float(best_model.get('fitness', 0))}, f, indent=2)
    
    print(f"Mejor arquitectura guardada en {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
