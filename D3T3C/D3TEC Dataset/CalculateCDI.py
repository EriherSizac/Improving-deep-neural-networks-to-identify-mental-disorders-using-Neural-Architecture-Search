# %%
import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuración de parámetros extendida
class Config:
    def __init__(self, architecture='CNN_LF', epochs=50, sample_rate=None, time=5, n_splits=5, window_size=5,
                 pad_short_audio=False, resize_input=True):
        self.architecture = architecture
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.time = time
        self.n_splits = n_splits
        self.window_size = window_size
        self.pad_short_audio = pad_short_audio    # Si True, se rellena audios cortos; si False, se omiten.
        self.resize_input = resize_input          # Si True, se redimensiona la imagen; de lo contrario, se deja intacta.

# Funciones de Preprocesamiento
def load_audio_data(directory, window_size, sample_rate, pad_short_audio=False):
    """
    Carga los archivos .wav desde un directorio, dividiéndolos en ventanas de duración 'window_size'.
    Si el audio tiene menos muestras que el requerido y pad_short_audio es True, se le hace padding.
    """
    audio_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(directory, file_name))
            if sample_rate is None:
                sample_rate = sr
            total_samples = waveform.shape[1]
            required_samples = int(window_size * sample_rate)
            if total_samples < required_samples:
                if pad_short_audio:
                    pad_amount = required_samples - total_samples
                    waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                    num_windows = 1
                else:
                    continue
            else:
                num_windows = int(total_samples / required_samples)
            for i in range(num_windows):
                start = int(i * required_samples)
                end = int((i + 1) * required_samples)
                audio_dict[f"{file_name}_{i}"] = waveform[:, start:end].numpy()
    return audio_dict, sample_rate

def preprocess_audio(audio_dict, sample_rate):
    """
    Calcula el melspectrograma de cada ventana y lo normaliza en [0,1].
    """
    audio_dict = copy.deepcopy(audio_dict)
    n_mels = 128
    n_fft = int(sample_rate * 0.029)
    hop_length = int(sample_rate * 0.010)
    win_length = int(sample_rate * 0.025)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    for filename, waveform in tqdm(audio_dict.items(), desc='MELSPECTROGRAM'):
        waveform_tensor = torch.from_numpy(waveform)
        spec = mel_transform(waveform_tensor)
        spec = db_transform(spec)
        spec = spec.numpy()
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        audio_dict[filename] = spec
    return audio_dict

def pad_and_crop_spectrograms(spectrograms, target_shape=(128, 128)):
    """
    Recorta o agrega padding a cada espectrograma para que tenga el tamaño objetivo.
    Si el espectrograma tiene más de 2 dimensiones, se toma el primer canal.
    """
    padded_spectrograms = []
    for spec in spectrograms:
        if spec.ndim > 2:
            spec = spec[0]
        if spec.shape[0] > target_shape[0]:
            spec = spec[:target_shape[0], :]
        if spec.shape[1] > target_shape[1]:
            spec = spec[:, :target_shape[1]]
        pad_width = [
            (0, max(0, target_shape[0] - spec.shape[0])),
            (0, max(0, target_shape[1] - spec.shape[1]))
        ]
        padded_spec = np.pad(spec, pad_width, mode='constant')
        padded_spectrograms.append(padded_spec)
    return np.array(padded_spectrograms)

def train_test_split_audio(audio_dict):
    """
    Lee 'Dataset.csv' para obtener etiquetas (usando PHQ-9 Score) y asocia cada espectrograma a su etiqueta.
    Se asume que el ID se extrae de los tres primeros caracteres del nombre del archivo.
    """
    df = pd.read_csv('./Dataset.csv', usecols=['Participant_ID', 'PHQ-9 Score'])
    print(df.head())
    df['labels'] = 0
    df.loc[df['PHQ-9 Score'] >= 10, 'labels'] = 1
    labels = df.set_index('Participant_ID')['labels'].to_dict()

    X, Y = [], []
    for filename, data in tqdm(audio_dict.items(), desc='LABEL'):
        ID = filename[:3]
        try:
            int_ID = int(ID)
            if int_ID in labels:
                dep = 0 if int(labels[int_ID]) == 0 else 1
                X.append(data)
                Y.append(dep)
            else:
                print(f"ID no encontrado: {int_ID} en {filename}")
        except ValueError:
            print(f"ID inválido: {ID} en {filename}")
    print(f"Total muestras extraídas: {len(X)}")
    X = pad_and_crop_spectrograms(X)
    Y = np.array(Y)
    X = X[..., np.newaxis]
    return X, Y


# Definición de modelos en PyTorch

class CNN_LF(nn.Module):
    def __init__(self, input_shape=(128, 128, 1), resize_input=True):
        super(CNN_LF, self).__init__()
        if resize_input:
            self.resizing = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        else:
            self.resizing = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 32 * 32, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        x = self.resizing(x)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class ReducedModel(nn.Module):
    def __init__(self, input_shape=(128, 128, 1)):
        super(ReducedModel, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.01)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.01)
        self.bn3 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)  # 🔹 Agregar pooling para reducir tamaño
        self.flatten = nn.Flatten()

        # 🔹 Calcular tamaño correcto de entrada para fc1
        self.feature_size = 8 * (128 // 2) * (128 // 2)  # 8 canales después del último Conv, con pooling
        self.fc1 = nn.Linear(self.feature_size, 32)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)  # 🔹 Reorganizar dimensiones

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.bn3(x)
        x = self.pool(x)  # 🔹 Aplicar pooling para reducir tamaño
        print(f"Shape de la entrada antes de Flatten: {x.shape}")
        x = self.flatten(x)
        print(f"Shape después de Flatten: {x.shape}")


        x = self.fc1(x)
        x = self.leaky3(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class SpectroCNN(nn.Module):
    def __init__(self, input_shape=(128, 128, 1)):
        super(SpectroCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=0.01)
            ) for _ in range(31)
        ])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky1(x)
        x = self.pool1(x)
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

def build_model(config):
    if config.architecture == 'CNN_LF':
        return CNN_LF(resize_input=config.resize_input)
    elif config.architecture == 'reduced_model':
        return ReducedModel()
    elif config.architecture == 'Spectro_CNN':
        return SpectroCNN()
    else:
        raise ValueError("Arquitectura no soportada")

# Plot de Espectrogramas
def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

# Función para calcular la especificidad
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-8)

def main(architecture='CNN_LF', epochs=50, n_splits=5, window_size=10, pad_short_audio=False, resize_input=True):
    config = Config(architecture=architecture, epochs=epochs, n_splits=n_splits, window_size=window_size,
                    pad_short_audio=pad_short_audio, resize_input=resize_input)
    directory = './SM-27'
    
    # Carga y preprocesamiento de audio (se pasa el parámetro pad_short_audio)
    audio_dict, sample_rate = load_audio_data(directory, config.window_size, config.sample_rate, config.pad_short_audio)
    audio_dict = preprocess_audio(audio_dict, sample_rate)
    X, Y = train_test_split_audio(audio_dict)
    
    # División en entrenamiento+validación y test (80%-20%)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Mostrar el primer y último espectrograma
    plot_spectrogram(X[0].squeeze(), "Primer Espectrograma")
    plot_spectrogram(X[-1].squeeze(), "Último Espectrograma")
    
    # Validación cruzada KFold en el conjunto de entrenamiento+validación
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, (train_index, val_index) in enumerate(kfold.split(X_train_val), 1):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        
        model = build_model(config)
        print(f"\nIniciando Fold {fold}")
        fold_results = train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config, device)
        results.append(fold_results)
    
    results = np.array(results)
    avg_results = np.mean(results, axis=0)
    
    print("\nResultados por fold:")
    for i, result in enumerate(results, 1):
        print(f"Fold {i} - Loss: {result[0]:.4f}, Accuracy: {result[1]:.4f}, Precision: {result[2]:.4f}, "
              f"Recall: {result[3]:.4f}, F1-score: {result[4]:.4f}, Specificity: {result[5]:.4f}")
    
    print("\nResultados Promedio:")
    print(f"Loss: {avg_results[0]:.4f}, Accuracy: {avg_results[1]:.4f}, Precision: {avg_results[2]:.4f}, "
          f"Recall: {avg_results[3]:.4f}, F1-score: {avg_results[4]:.4f}, Specificity: {avg_results[5]:.4f}")
    return avg_results


# Entrenamiento y Evaluación (con DataParallel y DataLoader en paralelo)
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config, device):
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Utilizando {torch.cuda.device_count()} GPUs")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    batch_size = 300

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    print(f"Shape de X_train: {X_train.shape}")
    print(f"Shape de Y_train: {Y_train.shape}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).int().cpu().numpy().flatten()
            labels = batch_y.int().cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    test_loss /= len(test_loader.dataset)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    spec = specificity_score(all_labels, all_preds)
    
    return test_loss, accuracy, precision, recall, f1, spec

# Grupo 1: Experimentar variando solo el tamaño de la ventana (para 50 y 100 épocas)
def run_experiments_group1():
    architectures = ['CNN_LF', 'reduced_model', 'Spectro_CNN']
    n_splits_options = [10]
    epochs_options = [50, 100]
    window_sizes = [2, 5, 10, 15, 20, 30]
    # Para este grupo, usamos los parámetros por defecto:
    pad_audio = False
    resize_input = True

    results = {}
    json_file = 'experiment_results_group1.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)

    total_experiments = (len(architectures) * len(n_splits_options) *
                         len(epochs_options) * len(window_sizes))
    experiment_count = 0

    total_done_experiments = len(results.keys())
    print(f"\nExperimentos ya realizados: {total_done_experiments}/{total_experiments}")

    for architecture in architectures:
        for n_splits in n_splits_options:
            for epochs in epochs_options:
                for window_size in window_sizes:
                    key = (f"group1_{architecture}_splits_{n_splits}_epochs_{epochs}_window_{window_size}"
                           f"_pad_{pad_audio}_resize_{resize_input}")
                    if key not in results.keys():
                        print(f"\nEjecutando experimento {experiment_count + 1 + total_done_experiments}/{total_experiments}: {key}")
                        avg_results = main(architecture=architecture, epochs=epochs, n_splits=n_splits,
                                           window_size=window_size, pad_short_audio=pad_audio,
                                           resize_input=resize_input)
                        results[key] = {
                            'Loss': avg_results[0],
                            'Accuracy': avg_results[1],
                            'Precision': avg_results[2],
                            'Recall': avg_results[3],
                            'F1-score': avg_results[4],
                            'Specificity': avg_results[5]
                        }
                        with open(json_file, 'w') as f:
                            json.dump(results, f, indent=4)
                        experiment_count += 1

                    else:
                        print(f"Experimento {key} ya realizado. Saltando...")   
    return results

# Grupo 2: Usar la "mejor ventana" (por ejemplo, 10) y variar el parámetro resize_input (True/False)
def run_experiments_group2(best_window=10):
    architectures = ['CNN_LF', 'reduced_model', 'Spectro_CNN']
    n_splits_options = [10]
    epochs_options = [50, 100]
    resize_options = [True, False]
    # Para este grupo, fijamos la ventana en la "mejor" (best_window) y mantenemos pad_audio = False.
    pad_audio = False

    results = {}
    json_file = 'experiment_results_group2.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)

    total_experiments = (len(architectures) * len(n_splits_options) *
                         len(epochs_options) * len(resize_options))
    experiment_count = 0

    for architecture in architectures:
        for n_splits in n_splits_options:
            for epochs in epochs_options:
                for resize in resize_options:
                    key = (f"group2_{architecture}_splits_{n_splits}_epochs_{epochs}_window_{best_window}"
                           f"_pad_{pad_audio}_resize_{resize}")
                    if key not in results:
                        print(f"\nEjecutando experimento {experiment_count + 1}/{total_experiments}: {key}")
                        avg_results = main(architecture=architecture, epochs=epochs, n_splits=n_splits,
                                           window_size=best_window, pad_short_audio=pad_audio,
                                           resize_input=resize)
                        results[key] = {
                            'Loss': avg_results[0],
                            'Accuracy': avg_results[1],
                            'Precision': avg_results[2],
                            'Recall': avg_results[3],
                            'F1-score': avg_results[4],
                            'Specificity': avg_results[5]
                        }
                        with open(json_file, 'w') as f:
                            json.dump(results, f, indent=4)
                        experiment_count += 1
    return results

def plot_experiment_results(results):
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        x_labels = list(results.keys())
        y_values = [results[k][metric] for k in x_labels]
        plt.plot(x_labels, y_values, marker='o', linestyle='-', label=metric)
        plt.title(f'{metric} across different experiments')
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        plt.xticks(rotation=90)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

def run_experiments():
    print("Ejecutando Grupo 1: Variando la ventana y épocas...")
    results_group1 = run_experiments_group1()
    print("\nEjecutando Grupo 2: Fijando la mejor ventana y variando el redimensionamiento...")
    # Supongamos que la mejor ventana es 10 (se puede ajustar según los resultados de Grupo 1)
    best_window = 10
    #results_group2 = run_experiments_group2(best_window=best_window)
    
    # Aquí se pueden graficar los resultados de cada grupo por separado o combinados
    print("\nResultados Grupo 1:")
    plot_experiment_results(results_group1)
    print("\nResultados Grupo 2:")
    #plot_experiment_results(results_group2)

# %%
if __name__ == '__main__':
    run_experiments()
