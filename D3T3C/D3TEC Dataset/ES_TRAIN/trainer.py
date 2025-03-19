import os
import json
import csv
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Importar la clase BuildPyTorchModel desde el módulo principal
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from TrainFinalModels import BuildPyTorchModel

class Config:
    def __init__(self, epochs=20, window_size=5, sample_rate=None, checkpoint_file="./checkpoint.json"):
        self.epochs = epochs
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.checkpoint_file = checkpoint_file

# Dataset personalizado para cargar audios en tiempo de ejecución
class AudioDataset(Dataset):
    def __init__(self, directory, dataset_csv, window_size):
        self.directory = directory
        self.window_size = window_size
        self.audio_segments = self._load_audio_segments(dataset_csv)

    def _load_audio_segments(self, dataset_csv):
        """Carga los nombres de los archivos y sus etiquetas desde el CSV, dividiendo en segmentos de `window_size` segundos."""
        df = pd.read_csv(dataset_csv, usecols=['Participant_ID', 'PHQ-9 Score'])
        df['label'] = (df['PHQ-9 Score'] >= 10).astype(int)
        labels = df.set_index('Participant_ID')['label'].to_dict()

        audio_segments = []

        for file_name in os.listdir(self.directory):
            if file_name.endswith(".wav"):
                participant_id = int(file_name.split("_")[0].split('.')[0])
                if participant_id not in labels:
                    continue

                label = labels[participant_id]
                file_path = os.path.join(self.directory, file_name)
                waveform, sample_rate = torchaudio.load(file_path)

                min_samples = self.window_size * sample_rate
                total_samples = waveform.shape[1]

                if total_samples < min_samples:
                    print(f"⚠️ OMITIENDO: {file_name} - Duración insuficiente ({total_samples/sample_rate:.2f} s)")
                    continue

                num_windows = total_samples // min_samples  # Dividir en segmentos de `window_size`
                for i in range(num_windows):
                    start = i * min_samples
                    end = start + min_samples
                    segment = waveform[:, start:end]
                    audio_segments.append((segment, label))

        return audio_segments

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        waveform, label = self.audio_segments[idx]
        spectrogram = self._generate_spectrogram(waveform)
        return spectrogram, label
    
    def _generate_spectrogram(self, waveform):
        """Convierte audio en espectrograma Mel, lo normaliza y lo redimensiona a 128x128."""
        import torch.nn.functional as F
        
        n_mels = 64
        sample_rate = 16000  # Aseguramos que sea consistente
        n_fft = int(sample_rate * 0.029)
        hop_length = int(sample_rate * 0.010)
        win_length = int(sample_rate * 0.025)

        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length
        )(waveform)

        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        spec = (spec - spec.min()) / (spec.max() - spec.min())
        
        # Evita el warning clonando y detach:
        tensor_spec = spec.clone().detach().float()  # Esperamos forma: (C, 64, tiempo)
        
        # Si el tensor tiene 3 dimensiones, agregamos la dimensión de batch:
        if tensor_spec.dim() == 3:
            tensor_spec = tensor_spec.unsqueeze(0)  # Ahora: (1, C, 64, tiempo)
        
        # Redimensionar a 128x128:
        tensor_spec = F.interpolate(tensor_spec, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Si se desea, quitar la dimensión de batch:
        tensor_spec = tensor_spec.squeeze(0)  # Resultado final: (C, 128, 128)
        
        print(f"📌 Espectrograma generado - Shape: {tensor_spec.shape}")
        return tensor_spec

def save_checkpoint(file_path, architecture_index):
    """
    Guarda el checkpoint con la última arquitectura completada.

    Args:
        file_path (str): Ruta donde se guarda el checkpoint.
        architecture_index (int): Índice de la última arquitectura entrenada.
    """
    checkpoint = {"last_completed": architecture_index}
    with open(file_path, 'w') as f:
        json.dump(checkpoint, f)
    print(f"📌 Checkpoint guardado: {checkpoint}")

def load_checkpoint(file_path):
    """
    Carga el checkpoint guardado para continuar el entrenamiento desde la última arquitectura.

    Args:
        file_path (str): Ruta del archivo de checkpoint.

    Returns:
        dict: Diccionario con el índice de la última arquitectura entrenada.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            checkpoint = json.load(f)
            return checkpoint
    return {"last_completed": -1}  # Si no hay checkpoint, empezar desde el inicio

def save_results_to_csv(file_path, architecture, results):
    """
    Guarda los resultados de la arquitectura en un archivo CSV.
    
    Parameters:
        - file_path (str): Ruta del archivo CSV donde se guardarán los resultados.
        - architecture (list): Representación codificada de la arquitectura.
        - results (list): Métricas del modelo (loss, accuracy, precision, recall, f1, specificity).
    """
    columns = ["Encoded Architecture", "Loss", "Accuracy", "Precision", "Recall", "F1", "Specificity"]

    # Si el archivo no existe, crear con encabezados
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    # Escribir los resultados
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(architecture)] + results)  # Guardar arquitectura y métricas
    
    print(f"📊 Resultados guardados en {file_path}")

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de evaluación: precisión, recall, F1-score y especificidad.
    
    Args:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Predicciones del modelo.
        
    Returns:
        tuple: (precision, recall, f1, specificity)
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calcular la especificidad
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return precision, recall, f1, specificity

def train_models(architectures, dataset_csv, directory, epochs=20, batch_size=1, save_file="final_results.csv", verbose=False):
    """
    Entrena modelos de redes neuronales con las arquitecturas especificadas.
    
    Args:
        architectures (list): Lista de arquitecturas codificadas para entrenar.
        dataset_csv (str): Ruta al archivo CSV con los datos del dataset.
        directory (str): Directorio que contiene los archivos de audio.
        epochs (int): Número de épocas para el entrenamiento.
        batch_size (int): Tamaño del batch para el entrenamiento.
        save_file (str): Archivo donde guardar los resultados.
        verbose (bool): Si es True, muestra información detallada durante el entrenamiento.
    """
    print("📌 Iniciando entrenamiento con 10-Fold Cross-Validation...")

    config = Config(epochs=epochs, window_size=2, checkpoint_file="final_checkpoint.json")
    checkpoint = load_checkpoint(config.checkpoint_file)

    print("📌 Cargando y procesando audios en tiempo de ejecución...")
    dataset = AudioDataset(directory, dataset_csv, config.window_size)
    print(f"📌 Total de muestras cargadas: {len(dataset)}")
    dataset = [d for d in dataset if d is not None]  # Filtrar posibles valores None

    print(f"📌 Total de muestras antes del balanceo: {len(dataset)}")

    # Balanceo de clases: cortar al tamaño de la clase minoritaria
    spectrograms, labels = zip(*dataset)
    spectrograms = torch.stack(spectrograms)
    labels = torch.tensor(labels)

    num_class_0 = (labels == 0).sum().item()
    num_class_1 = (labels == 1).sum().item()
    min_class_count = min(num_class_0, num_class_1)

    print(f"📊 Ajustando ambas clases a {min_class_count} muestras.")
    idx_class_0 = torch.where(labels == 0)[0][:min_class_count]
    idx_class_1 = torch.where(labels == 1)[0][:min_class_count]
    balanced_indices = torch.cat((idx_class_0, idx_class_1))

    spectrograms = spectrograms[balanced_indices]
    labels = labels[balanced_indices]

    print(f"📌 Total de muestras después del balanceo: {spectrograms.shape[0]}")

    X_np = spectrograms.numpy()
    y_np = labels.numpy()
    print("Forma de datos para KFold:")
    print("X_np:", X_np.shape)
    print("y_np:", y_np.shape)

    kfold = StratifiedKFold(n_splits=5)#, shuffle=False, random_state=42

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("❌ Error: Los datos de entrada y las etiquetas tienen tamaños diferentes.")

    print(f"📌 Total de arquitecturas a evaluar: {len(architectures)}")

    for i, architecture in enumerate(architectures):
        if i <= checkpoint["last_completed"]:
            print(f"⏭️ Saltando arquitectura {i+1}/{len(architectures)} (ya entrenada)...")
            continue

        print(f"\n🚀 Evaluando arquitectura {i + 1}/{len(architectures)} con 10-Fold Cross-Validation...")

        model = BuildPyTorchModel(architecture, input_shape=(1, 128, 128), verbose=verbose)
        fold_results = []  # Lista para almacenar resultados de cada fold

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_np, y_np)):
            print(f"\n📌 Fold {fold+1}/10 - Entrenando modelo...")

            fold_model = BuildPyTorchModel(architecture, input_shape=(1, 128, 128), verbose=verbose)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            fold_model = fold_model.to(device)

            X_train, X_val = spectrograms[train_idx], spectrograms[val_idx]
            Y_train, Y_val = labels[train_idx], labels[val_idx]

            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

            optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(config.epochs):
                fold_model.train()
                running_loss = 0.0
                for inputs, batch_labels in train_loader:
                    inputs, batch_labels = inputs.to(device), batch_labels.float().to(device)
                    optimizer.zero_grad()
                    outputs = fold_model(inputs)
                    batch_labels = batch_labels.view(-1, 1)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                print(f"🔹 Fold {fold+1} - Epoch {epoch+1}/{config.epochs} - Loss: {running_loss / len(train_loader):.4f}")

            fold_model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for inputs, batch_labels in val_loader:
                    inputs, batch_labels = inputs.to(device), batch_labels.float().to(device)
                    batch_labels = batch_labels.view(-1, 1)
                    outputs = fold_model(inputs).squeeze()
                    predictions = (torch.sigmoid(outputs) > 0.5).int()
                    y_true.extend(batch_labels.cpu().numpy().tolist())
                    y_pred.extend(predictions.cpu().numpy().tolist())

            accuracy = (np.array(y_true) == np.array(y_pred)).mean()
            precision, recall, f1, specificity = calculate_metrics(y_true, y_pred)
            fold_result = [running_loss / len(train_loader), accuracy, precision, recall, f1, specificity]
            fold_results.append(fold_result)

        # Al finalizar todos los folds, imprimir los resultados individuales
        print("\n📌 Resultados individuales por fold:")
        for idx, result in enumerate(fold_results):
            print(f"Fold {idx+1}:")
            print(f"  Loss: {result[0]:.4f}")
            print(f"  Accuracy: {result[1]:.4f}")
            print(f"  Precision: {result[2]:.4f}")
            print(f"  Recall: {result[3]:.4f}")
            print(f"  F1: {result[4]:.4f}")
            print(f"  Specificity: {result[5]:.4f}\n")

        avg_results = np.mean(fold_results, axis=0).tolist()
        print(f"📊 Resultados Promediados - Accuracy: {avg_results[1]:.4f}, F1: {avg_results[4]:.4f}")

        save_results_to_csv(save_file, architecture, avg_results)

        print(f"📌 Arquitectura {i+1} evaluada con éxito. Guardando checkpoint...")
        save_checkpoint(config.checkpoint_file, i)

    print("✅ Entrenamiento completado con éxito.")
