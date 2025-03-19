import argparse
import os
import sys

# Ajustar la ruta para importaciones relativas
from ES_TRAIN.trainer import train_models
from ES_TRAIN.models_config import models_to_train

def setup_argparse():
    parser = argparse.ArgumentParser(
        description='Entrenar modelos de redes neuronales con datos de audio'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Dataset.csv',
        help='Ruta al archivo CSV del dataset'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='./SM-27',
        help='Directorio que contiene los archivos de audio'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=400,
        help='Tamaño del batch para entrenamiento'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número de épocas para entrenamiento'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Final_EncodedChromosomes_V3_results.csv',
        help='Archivo de salida para guardar los resultados del entrenamiento'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Activar salida detallada durante el entrenamiento'
    )
    
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Entrenar modelos usando las arquitecturas predefinidas
    print("Iniciando entrenamiento de modelos de redes neuronales...")
    train_models(
        models_to_train,
        args.dataset,
        args.audio_dir,
        save_file=args.output,
        verbose=args.verbose,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    print(f"Entrenamiento completado. Resultados guardados en {args.output}")

if __name__ == '__main__':
    main()