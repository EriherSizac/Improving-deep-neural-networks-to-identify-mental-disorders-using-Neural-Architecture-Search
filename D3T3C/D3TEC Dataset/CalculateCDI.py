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

# Configuración de parámetros
class Config:
    def __init__(self, architecture='CNN_LF', epochs=50, sample_rate=None, time=5, n_splits=5, window_size=5):
        self.architecture = architecture
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.time = time
        self.n_splits = n_splits
        self.window_size = window_size

# Funciones de Preprocesamiento
def load_audio_data(directory, window_size, sample_rate):
    """
    Carga los archivos .wav desde un directorio, dividiéndolos en ventanas de duración 'window_size'
    """
    audio_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(directory, file_name))
            if sample_rate is None:
                sample_rate = sr
            # Número de ventanas que se pueden extraer
            num_windows = int(waveform.shape[1] / (window_size * sample_rate))
            for i in range(num_windows):
                start = int(i * window_size * sample_rate)
                end = int((i + 1) * window_size * sample_rate)
                # Se guarda cada ventana con una clave única
                audio_dict[f"{file_name}_{i}"] = waveform[:, start:end].numpy()
    return audio_dict, sample_rate

def preprocess_audio(audio_dict, sample_rate):
    """
    Calcula el melspectrograma de cada ventana y lo normaliza en el rango [0,1]
    """
    audio_dict = copy.deepcopy(audio_dict)
    n_mels = 128
    n_fft = int(sample_rate * 0.029)
    hop_length = int(sample_rate * 0.010)
    win_length = int(sample_rate * 0.025)

    # Se crean los transformadores (estos se podrían definir fuera del bucle para eficiencia)
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
        # Normalización min-max
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        audio_dict[filename] = spec
    return audio_dict

def pad_and_crop_spectrograms(spectrograms, target_shape=(128, 128)):
    """
    Recorta o agrega padding a cada espectrograma para que tenga el tamaño objetivo.
    """
    padded_spectrograms = []
    for spec in spectrograms:
        # Recorte si es mayor al tamaño objetivo
        if spec.shape[0] > target_shape[0]:
            spec = spec[:target_shape[0], :]
        if spec.shape[1] > target_shape[1]:
            spec = spec[:, :target_shape[1]]
        
        pad_width = [(0, max(0, target_shape[0] - spec.shape[0])), 
                     (0, max(0, target_shape[1] - spec.shape[1]))]
        
        padded_spec = np.pad(spec, pad_width, mode='constant')
        padded_spectrograms.append(padded_spec)
    return np.array(padded_spectrograms)

def train_test_split_audio(audio_dict):
    """
    Lee el archivo 'Dataset.csv' para obtener las etiquetas (umbral en PHQ-9 Score) y asocia cada
    espectrograma a su etiqueta según el ID del participante.
    """
    df = pd.read_csv('./Dataset.csv', usecols=['Participant_ID', 'PHQ-9 Score'])
    print(df.head())
    df['labels'] = 0
    df.loc[df['PHQ-9 Score'] >= 10, 'labels'] = 1
    labels = df.set_index('Participant_ID')['labels'].to_dict()

    X, Y = [], []
    for filename, data in tqdm(audio_dict.items(), desc='LABEL'):
        # Verifica y depura cómo se extrae el ID
        ID = filename[:3]
        if ID in labels:
            dep = 0 if labels[ID] == 0 else 1
            X.append(data)
            Y.append(dep)
        else:
            print(type(ID))
            print(type(df['Participant_ID']))
            print(f"ID no encontrado: {ID} para el archivo {filename}")

    print(f"Total muestras extraídas: {len(X)}")
    X = pad_and_crop_spectrograms(X)
    Y = np.array(Y)
    X = X[..., np.newaxis]
    return X, Y


# Definición de Modelos en PyTorch

# Modelo CNN_LF similar al de Keras
class CNN_LF(nn.Module):
    def __init__(self, input_shape=(128, 128, 1)):
        super(CNN_LF, self).__init__()
        # Se usa nn.Upsample para “redimensionar” la imagen a 128x128 (aunque ya esté en ese tamaño)
        self.resizing = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        # Suponiendo que la entrada tiene 1 canal
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # reduce dimensiones a la mitad
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calcular dimensión de salida: entrada 128x128 -> después de 2 maxpools = 32x32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 32 * 32, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Se espera que x tenga forma [batch, 128, 128, 1], se reordena a [batch, 1, 128, 128]
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
        x = torch.sigmoid(x)
        return x

# Modelo reduced_model similar al de Keras
class ReducedModel(nn.Module):
    def __init__(self, input_shape=(128, 128, 1)):
        super(ReducedModel, self).__init__()
        # Se aplica BatchNorm a la entrada (1 canal)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.01)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.01)
        self.bn3 = nn.BatchNorm2d(8)
        self.flatten = nn.Flatten()
        # Para una imagen de 128x128 y 8 canales
        self.fc1 = nn.Linear(8 * 128 * 128, 32)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky3(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Modelo Spectro_CNN similar al de Keras
class SpectroCNN(nn.Module):
    def __init__(self, input_shape=(128, 128, 1)):
        super(SpectroCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 31 bloques de convolución
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=0.01)
            ) for _ in range(31)
        ])
        self.flatten = nn.Flatten()
        # Tras el primer pooling, la dimensión es 64x64 con 32 canales
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
        x = torch.sigmoid(x)
        return x

def build_model(config):
    if config.architecture == 'CNN_LF':
        return CNN_LF()
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

# Entrenamiento y Evaluación en PyTorch
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config, device):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    batch_size = 32

    # Crear DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Bucle de entrenamiento
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            # Asegurarse que batch_y tenga forma [batch, 1]
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluación en validación
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
    
    # Evaluación en test
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

# Función principal de ejecución (con validación cruzada)
def main(architecture='CNN_LF', epochs=50, n_splits=5, window_size=10):
    config = Config(architecture=architecture, epochs=epochs, n_splits=n_splits, window_size=window_size)
    directory = './SM-27'
    
    # Carga y preprocesamiento de audio
    audio_dict, sample_rate = load_audio_data(directory, config.window_size, config.sample_rate)
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

# %%
def plot_experiment_results(results):
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']
    
    # Gráfica individual por métrica
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

    # Gráfica comparativa de todas las métricas
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        metric_values = [results[k][metric] for k in results.keys()]
        plt.plot(list(results.keys()), metric_values, marker='o', label=metric)
    plt.title('Comparison of all metrics across different experiments')
    plt.xlabel('Experiment')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Gráfica de línea para diferentes window_size (filtrado por arquitectura, splits y epochs)
    for architecture in ['CNN_LF', 'reduced_model', 'Spectro_CNN']:
        for n_splits in [5, 10]:
            for epochs in [50, 100]:
                filtered_results = {k: v for k, v in results.items() if f"{architecture}_splits_{n_splits}_epochs_{epochs}" in k}
                if filtered_results:
                    plt.figure(figsize=(12, 6))
                    for metric in metrics:
                        metric_values = [v[metric] for v in filtered_results.values()]
                        window_sizes = [int(k.split('_')[-1]) for k in filtered_results.keys()]
                        plt.plot(window_sizes, metric_values, marker='o', label=metric)
                    plt.title(f'{architecture} - {n_splits} splits - {epochs} epochs across window sizes')
                    plt.xlabel('Window Size')
                    plt.ylabel('Metric Value')
                    plt.legend(loc='best')
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.show()

def run_experiments():
    architectures = ['CNN_LF', 'reduced_model', 'Spectro_CNN']
    n_splits_options = [5]
    epochs_options = [100]
    window_sizes = [2, 5, 10, 15, 20, 30]

    results = {}
    json_file = 'experiment_results.json'

    # Cargar resultados previos si existen
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)

    total_experiments = len(architectures) * len(n_splits_options) * len(epochs_options) * len(window_sizes)
    experiment_count = 0

    for architecture in architectures:
        for n_splits in n_splits_options:
            for epochs in epochs_options:
                for window_size in window_sizes:
                    experiment_key = f"{architecture}_splits_{n_splits}_epochs_{epochs}_window_{window_size}"
                    if experiment_key not in results:
                        print(f"\nRunning experiment {experiment_count + 1}/{total_experiments}: {experiment_key}")
                        avg_results = main(architecture=architecture, epochs=epochs, n_splits=n_splits, window_size=window_size)
                        results[experiment_key] = {
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

    plot_experiment_results(results)

# %%
# Ejecución de los experimentos
if __name__ == '__main__':
    run_experiments()
