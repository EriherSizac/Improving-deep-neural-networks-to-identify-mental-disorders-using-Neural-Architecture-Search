# %%
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torchaudio
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Resizing, Conv2D, Dropout, BatchNormalization, MaxPooling2D, MaxPool2D, Flatten, Dense, Input, LeakyReLU
from tqdm import tqdm
import tensorflow_addons as tfa

F1 = tfa.metrics.F1Score(num_classes=1, threshold=0.5)

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
    audio_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(directory, file_name))
            if sample_rate is None:
                sample_rate = sr
            num_windows = int(waveform.shape[1] / (window_size * sample_rate))
            for i in range(num_windows):
                start = i * window_size * sample_rate
                end = (i + 1) * window_size * sample_rate
                audio_dict[f"{file_name}_{i}"] = waveform[:, start:end].numpy()
    return audio_dict, sample_rate

def preprocess_audio(audio_dict, sample_rate):
    audio_dict = copy.deepcopy(audio_dict)
    n_mels = 128
    n_fft = int(sample_rate * 0.029)
    hop_length = int(sample_rate * 0.010)
    win_length = int(sample_rate * 0.025)

    for filename, waveform in tqdm(audio_dict.items(), desc='MELSPECTROGRAM'):
        waveform = torch.from_numpy(waveform)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, win_length=win_length)(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        spec = spec.numpy()
        spec = (spec - spec.min()) / (spec.max() - spec.min())
        audio_dict[filename] = spec
    return audio_dict

def pad_and_crop_spectrograms(spectrograms, target_shape=(128, 128)):
    padded_spectrograms = []
    for spec in spectrograms:
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
    df = pd.read_csv('Dataset.csv', usecols=['Participant_ID', 'PHQ-9 Score'], dtype={1: str})
    df['labels'] = np.zeros([len(df),], dtype=int)
    df.loc[df['PHQ-9 Score'] < 10, 'labels'] = 0
    df.loc[df['PHQ-9 Score'] >= 10, 'labels'] = 1

    labels = df.set_index('Participant_ID').to_dict()['labels']

    X, Y = [], []
    for filename, data in tqdm(audio_dict.items(), 'LABEL'):
        ID = filename[:3]
        if ID in labels:
            dep = 0 if labels[ID] == 0 else 1
            [X.append(x) for x in data]
            [Y.append(dep) for x in data]

    X = pad_and_crop_spectrograms(X)
    Y = np.array(Y)

    X = X[..., np.newaxis]
    return X, Y

# Definición de Modelos
def build_model(config):
    if config.architecture == 'CNN_LF':
        return build_CNN_LF_model()
    elif config.architecture == 'reduced_model':
        return build_reduced_model()
    elif config.architecture == 'Spectro_CNN':
        return build_Spectro_CNN_model()
    else:
        raise ValueError("Arquitectura no soportada")

def build_CNN_LF_model(input_shape=(128, 128, 1)):
    model = Sequential()
    model.add(Resizing(128, 128, input_shape=input_shape))
    model.add(Conv2D(30, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(15, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model

def build_reduced_model(input_shape=(128, 128, 1)):
    model = Sequential([
        BatchNormalization(name='batch_normalization_9'),
        Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=input_shape, name='conv2d_6'),
        LeakyReLU(alpha=0.01, name='leaky_re_lu_9'),
        BatchNormalization(name='batch_normalization_10'),
        Conv2D(8, (3, 3), padding='same', name='conv2d_7'),
        LeakyReLU(alpha=0.01, name='leaky_re_lu_10'),
        BatchNormalization(name='batch_normalization_11'),
        Flatten(name='flatten_6'),
        Dense(32, name='dense_6'),
        LeakyReLU(alpha=0.01, name='leaky_re_lu_11'),
        Dense(1, activation='sigmoid', name='dense_7')
    ])
    return model

def build_Spectro_CNN_model(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    for _ in range(31):
        x = Conv2D(32, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

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
    return tn / (tn + fp)

# Entrenamiento y Evaluación
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config):
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy", 'Precision', 'Recall', F1])
    model.fit(X_train, Y_train, epochs=config.epochs, validation_data=(X_val, Y_val))
    results = model.evaluate(X_test, Y_test)
    
    # Predicciones para métricas adicionales
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = results[1]
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    specificity = specificity_score(Y_test, Y_pred)
    
    return results[0], accuracy, precision, recall, f1, specificity

# Ejecución Principal
def main(architecture='CNN_LF', epochs=50, n_splits=5, window_size=10):
    config = Config(architecture=architecture, epochs=epochs, n_splits=n_splits, window_size=window_size)
    directory = './SM-27'
    
    audio_dict, sample_rate = load_audio_data(directory, config.window_size, config.sample_rate)
    audio_dict = preprocess_audio(audio_dict, sample_rate)
    X, Y = train_test_split_audio(audio_dict)

    # División inicial en conjuntos de entrenamiento+validación y prueba
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Plot del primer y último espectrograma
    plot_spectrogram(X[0].squeeze(), "Primer Espectrograma")
    plot_spectrogram(X[-1].squeeze(), "Último Espectrograma")
    
    kfold = KFold(n_splits=config.n_splits, shuffle=True)

    results = []
    
    for fold, (train_index, val_index) in enumerate(kfold.split(X_train_val), 1):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        
        model = build_model(config)
        fold_results = train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config)
        results.append(fold_results)
    
    results = np.array(results)
    avg_results = np.mean(results, axis=0)
    
    print("\nResults per fold:")
    for i, result in enumerate(results, 1):
        print(f"Fold {i} - Loss: {result[0]}, Accuracy: {result[1]}, Precision: {result[2]}, Recall: {result[3]}, F1-score: {result[4]}, Specificity: {result[5]}")
    
    print("\nAverage results:")
    print(f"Loss: {avg_results[0]}, Accuracy: {avg_results[1]}, Precision: {avg_results[2]}, Recall: {avg_results[3]}, F1-score: {avg_results[4]}, Specificity: {avg_results[5]}")
    return avg_results




# %%
#main()

# %%
#main(architecture='reduced_model')

# %%
#main(architecture='Spectro_CNN')

# %%
import json

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
                        print(f"Running experiment {experiment_count + 1}/{total_experiments}: {experiment_key}")
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

def plot_experiment_results(results):
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity']
    plt.figure(figsize=(20, 10))

    for metric in metrics:
        plt.figure(figsize=(20, 10))
        for key, value in results.items():
            plt.plot(key, value[metric], marker='o', label=key)
        plt.title(f'{metric} across different experiments')
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        plt.legend(loc='best')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # Grafica comparativa de todas las métricas
    plt.figure(figsize=(20, 10))
    for metric in metrics:
        metric_values = [value[metric] for key, value in results.items()]
        plt.plot(list(results.keys()), metric_values, marker='o', label=metric)
    
    plt.title('Comparison of all metrics across different experiments')
    plt.xlabel('Experiment')
    plt.ylabel('Metric Value')
    plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Gráfica de línea para diferentes window_size
    for architecture in ['CNN_LF', 'reduced_model', 'Spectro_CNN']:
        for n_splits in [5, 10]:
            for epochs in [50, 100]:
                filtered_results = {k: v for k, v in results.items() if f"{architecture}_splits_{n_splits}_epochs_{epochs}" in k}
                if filtered_results:
                    plt.figure(figsize=(20, 10))
                    for metric in metrics:
                        metric_values = [value[metric] for key, value in filtered_results.items()]
                        window_sizes = [int(key.split('_')[-1].replace('window_', '')) for key in filtered_results.keys()]
                        plt.plot(window_sizes, metric_values, marker='o', label=metric)
                    
                    plt.title(f'Comparison of {architecture} with {n_splits} splits and {epochs} epochs across different window sizes')
                    plt.xlabel('Window Size')
                    plt.ylabel('Metric Value')
                    plt.legend(loc='best')
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.show()

# Ejecución de los experimentos
run_experiments()

