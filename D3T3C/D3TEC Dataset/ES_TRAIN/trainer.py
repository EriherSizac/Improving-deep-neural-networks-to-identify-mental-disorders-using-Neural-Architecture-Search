import os
import json
import csv
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T # Make sure T is imported
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix   
from .utils.encoding import BuildPyTorchModel
from torch.utils.data.dataloader import default_collate

class Config:
    def __init__(self, epochs=20, window_size=5, sample_rate=None, checkpoint_file="./checkpoint.json"):
        self.epochs = epochs
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.checkpoint_file = checkpoint_file

# Dataset personalizado para cargar audios en tiempo de ejecución
# In SurrogatesGenerator.py



# Placeholder for load/save normalization params if they are methods of the class
# Otherwise, define them as standalone functions if needed.

class AudioDataset(Dataset):
    # Modified __init__ to accept file_label_list and optional scaler_file
    def __init__(self, file_label_list, window_size, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512, scaler_file="normalization_params.json"):
        """
        Dataset para cargar y procesar archivos de audio de forma diferida (lazy loading).
        
        Args:
            file_label_list (list): Lista de tuplas (filepath, label).
            window_size (int): Tamaño de la ventana en segundos.
            sample_rate (int): Tasa de muestreo esperada.
            n_mels (int): Número de bandas Mel.
            n_fft (int): Tamaño FFT.
            hop_length (int): Salto de ventana FFT.
            scaler_file (str): Ruta al archivo JSON para guardar/cargar parámetros de normalización.
        """
        self.file_label_list = file_label_list
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler_file = scaler_file
        self.scaler = None  # Para almacenar parámetros de normalización global {'mean': mean, 'std': std}

        # Intentar cargar parámetros de normalización existentes
        self._load_normalization_params()

        # Si no se cargaron, calcularlos y guardarlos
        if self.scaler is None:
            print("🔍 No se encontraron parámetros de normalización. Calculando nuevos...")
            self._calculate_and_save_global_normalization()

    def _load_normalization_params(self):
        """Carga los parámetros de normalización desde el archivo JSON."""
        if self.scaler_file and os.path.exists(self.scaler_file):
            try:
                with open(self.scaler_file, 'r') as f:
                    params = json.load(f)
                    # Convertir a tensores si se guardaron como listas/números
                    self.scaler = {
                        'mean': torch.tensor(params['mean']),
                        'std': torch.tensor(params['std'])
                    }
                print(f"✅ Parámetros de normalización cargados desde: {self.scaler_file}")
                print(f"   Mean: {self.scaler['mean']:.4f}, Std: {self.scaler['std']:.4f}")
            except Exception as e:
                print(f"⚠️ Error al cargar parámetros de normalización desde {self.scaler_file}: {e}")
                self.scaler = None
        else:
             print(f"ℹ️ Archivo de normalización {self.scaler_file} no encontrado.")

    def _save_normalization_params(self):
        """Guarda los parámetros de normalización en un archivo JSON."""
        if self.scaler and self.scaler_file:
            try:
                # Convertir tensores a números/listas para serialización JSON
                params_to_save = {
                    'mean': self.scaler['mean'].item() if torch.is_tensor(self.scaler['mean']) else self.scaler['mean'],
                    'std': self.scaler['std'].item() if torch.is_tensor(self.scaler['std']) else self.scaler['std']
                 }
                with open(self.scaler_file, 'w') as f:
                    json.dump(params_to_save, f, indent=4)
                print(f"💾 Parámetros de normalización guardados en: {self.scaler_file}")
            except Exception as e:
                print(f"⚠️ Error al guardar parámetros de normalización en {self.scaler_file}: {e}")

    def _calculate_and_save_global_normalization(self):
        """Calcula los parámetros de normalización global (media y std) para todo el dataset."""
        if not self.file_label_list:
            print("⚠️ No hay archivos en la lista para calcular la normalización global.")
            return

        print(f"🧮 Calculando parámetros de normalización global de {len(self.file_label_list)} archivos...")

        all_specs_sum = 0.0
        all_specs_sq_sum = 0.0
        total_elements = 0
        processed_files = 0

        # Iterar sobre los archivos para calcular estadísticas acumuladas
        for filepath, _ in self.file_label_list:
            try:
                waveform, sr = torchaudio.load(filepath)

                # Resample si es necesario
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Solo procesar el primer segmento válido por archivo para el cálculo de estadísticas
                # (esto es una aproximación, calcular sobre todos los segmentos sería más preciso pero más lento)
                segment_samples = int(self.window_size * self.sample_rate)
                if waveform.shape[1] >= segment_samples:
                    segment = waveform[:, :segment_samples]

                    # Generar espectrograma sin normalizar
                    mel_transform = T.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length
                    )
                    mel_spec = mel_transform(segment)
                    mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

                    # Acumular sumas para calcular media y varianza online
                    all_specs_sum += torch.sum(mel_spec_db)
                    all_specs_sq_sum += torch.sum(mel_spec_db ** 2)
                    total_elements += mel_spec_db.numel()
                    processed_files += 1
                # else: # Opcional: imprimir si se omite un archivo corto
                #     print(f"  -> Omitiendo {os.path.basename(filepath)} para stats (demasiado corto)")

            except Exception as e:
                print(f"⚠️ Error procesando {filepath} durante cálculo de normalización: {e}")
                continue # Saltar al siguiente archivo

        if total_elements > 0:
            global_mean = all_specs_sum / total_elements
            global_var = (all_specs_sq_sum / total_elements) - (global_mean ** 2)
            global_std = torch.sqrt(global_var)

             # Evitar std cero o muy pequeño
            if global_std < 1e-10:
                 print(f"⚠️ Desviación estándar global muy baja ({global_std:.4e}), usando 1.0 en su lugar.")
                 global_std = torch.tensor(1.0)
                 
            self.scaler = {'mean': global_mean, 'std': global_std}
            print(f"📊 Cálculo completado ({processed_files} archivos procesados): Mean={global_mean:.4f}, Std={global_std:.4f}")
            self._save_normalization_params() # Guardar los parámetros calculados
        else:
            print("❌ No se pudieron procesar archivos para calcular la normalización.")
            # Asignar valores por defecto si falla el cálculo
            self.scaler = {'mean': torch.tensor(0.0), 'std': torch.tensor(1.0)}


    def __len__(self):
        """Devuelve el número total de muestras en el dataset."""
        return len(self.file_label_list)

    def __getitem__(self, idx):
        """
        Carga, procesa y devuelve un único espectrograma y su etiqueta.
        """
        filepath, label = self.file_label_list[idx]

        try:
            waveform, sr = torchaudio.load(filepath)

            # Resample si es necesario
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Extraer el primer segmento de 'window_size' segundos
            # Nota: Podríamos querer seleccionar segmentos aleatorios o todos los segmentos.
            # Por simplicidad, tomamos el primero que sea suficientemente largo.
            segment_samples = int(self.window_size * self.sample_rate)
            if waveform.shape[1] < segment_samples:
                 # Si el audio es demasiado corto, podríamos devolver None o un tensor vacío.
                 # Devolver None es más fácil de manejar en el bucle de entrenamiento.
                 # print(f"⚠️ Audio {os.path.basename(filepath)} demasiado corto ({waveform.shape[1]/self.sample_rate:.2f}s), devolviendo None.")
                 return None, None # Indicar que esta muestra no es válida

            # Tomar el primer segmento
            segment = waveform[:, :segment_samples]

            # Generar espectrograma de Mel
            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            mel_spec = mel_transform(segment)

            # Convertir a escala logarítmica (dB)
            mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

            # Normalizar usando los parámetros globales calculados/cargados
            if self.scaler and self.scaler['std'] > 1e-10:
                normalized_spec = (mel_spec_db - self.scaler['mean']) / self.scaler['std']
            else:
                # Si no hay scaler o std es cero, devolver sin normalizar (o manejar de otra forma)
                normalized_spec = mel_spec_db
                if not self.scaler: print("⚠️ Scaler no disponible, devolviendo espectrograma sin normalizar.")
                elif self.scaler['std'] <= 1e-10: print(f"⚠️ Std <= 1e-10 ({self.scaler['std']}), devolviendo espectrograma sin normalizar.")
            
            return normalized_spec, torch.tensor(label, dtype=torch.float)

        except Exception as e:
            print(f"❌ Error cargando o procesando {filepath} en __getitem__: {e}")
            # Devolver None para indicar un error con esta muestra específica
            return None, None


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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset # Added Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import os
# Assuming other necessary imports like AudioDataset, BuildPyTorchModel, Config, etc. are present

# Placeholder for calculate_metrics if not defined elsewhere
def calculate_metrics(y_true, y_pred):
    y_true_np = np.array(y_true).flatten() # Ensure 1D
    y_pred_np = np.array(y_pred).flatten() # Ensure 1D
    
    # Ensure binary classification if not already
    y_true_np = y_true_np.astype(int)
    y_pred_np = y_pred_np.astype(int)

    tp = np.sum((y_true_np == 1) & (y_pred_np == 1))
    tn = np.sum((y_true_np == 0) & (y_pred_np == 0))
    fp = np.sum((y_true_np == 0) & (y_pred_np == 1))
    fn = np.sum((y_true_np == 1) & (y_pred_np == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Handle potential NaN or Inf if denominators were zero, although checks should prevent this
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    specificity = np.nan_to_num(specificity)
    
    return precision, recall, f1, specificity

# Placeholder for save_results_to_csv if not defined
def save_results_to_csv(filename, architecture, results):
     # Implementation needed: Append architecture and results to the CSV
     print(f"DUMMY: Saving results for architecture to {filename}")
     pass

# Placeholder for checkpoint functions
def load_checkpoint(filepath):
    if os.path.exists(filepath):
        # Implementation needed: Load JSON checkpoint
        print(f"DUMMY: Loading checkpoint from {filepath}")
        return {"last_completed": -1} # Example default
    return {"last_completed": -1}

def save_checkpoint(filepath, index):
     # Implementation needed: Save JSON checkpoint
     print(f"DUMMY: Saving checkpoint {index} to {filepath}")
     pass
     
# Placeholder for Config class if not defined
class Config:
    def __init__(self, epochs=20, window_size=2, checkpoint_file="checkpoint.json", sample_rate=16000): # Added sample_rate default
        self.epochs = epochs
        self.window_size = window_size
        self.checkpoint_file = checkpoint_file
        self.sample_rate = sample_rate # AudioDataset might need this

def train_models(architectures, dataset_csv, directory, epochs=20, batch_size=16, save_file="final_results.csv", verbose=False): # Increased default batch_size
    """
    Entrena modelos de redes neuronales con las arquitecturas especificadas usando K-Fold CV.
    
    Args:
        architectures (list): Lista de arquitecturas codificadas para entrenar.
        dataset_csv (str): Ruta al archivo CSV con los datos del dataset.
        directory (str): Directorio que contiene los archivos de audio.
        epochs (int): Número de épocas para el entrenamiento por fold.
        batch_size (int): Tamaño del batch para el entrenamiento.
        save_file (str): Archivo donde guardar los resultados promediados por arquitectura.
        verbose (bool): Si es True, muestra información detallada.
    """
    print("📌 Iniciando entrenamiento con 10-Fold Cross-Validation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")
    torch.backends.cudnn.benchmark = True # Enable cuDNN benchmark

    config = Config(epochs=epochs, window_size=2, checkpoint_file="final_checkpoint.json")
    checkpoint = load_checkpoint(config.checkpoint_file)
    print(f"📌 Total de arquitecturas a procesar: {len(architectures)}")

    # --- 1. Carga inicial de metadatos y balanceo de archivos ---
    print("📊 Cargando metadatos y balanceando archivos...")
    try:
        df = pd.read_csv(dataset_csv, usecols=['Participant_ID', 'PHQ-9 Score'])
        df['label'] = (df['PHQ-9 Score'] >= 10).astype(int)
        labels_dict = df.set_index('Participant_ID')['label'].to_dict()
    except Exception as e:
        print(f"❌ Error cargando o procesando {dataset_csv}: {e}")
        return

    all_files = []
    class_0_files = []
    class_1_files = []

    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            try:
                # Asegurarse que el ID se extrae correctamente
                participant_id_str = file_name.split("_")[0]
                if '.' in participant_id_str: # Manejar casos como '001.wav'
                     participant_id_str = participant_id_str.split('.')[0]
                participant_id = int(participant_id_str)
                
                if participant_id in labels_dict:
                    label = labels_dict[participant_id]
                    file_path = os.path.join(directory, file_name)
                    if label == 0:
                        class_0_files.append((file_path, label))
                    else:
                        class_1_files.append((file_path, label))
                else:
                    if verbose:
                        print(f"🤔 ID {participant_id} del archivo {file_name} no encontrado en {dataset_csv}.")
            except ValueError:
                 if verbose:
                      print(f"⚠️ No se pudo extraer ID numérico de {file_name}")
            except Exception as e:
                 if verbose:
                      print(f"❓ Error procesando archivo {file_name}: {e}")

    min_class_count = min(len(class_0_files), len(class_1_files))
    if min_class_count == 0:
        print("❌ Error: No se encontraron suficientes archivos para una de las clases.")
        return
        
    print(f"⚖️ Balanceando clases a {min_class_count} archivos por clase.")
    # Consider shuffling before slicing for randomness if needed
    # random.shuffle(class_0_files)
    # random.shuffle(class_1_files)
    balanced_files = class_0_files[:min_class_count] + class_1_files[:min_class_count]
    # Shuffle the final balanced list
    # random.shuffle(balanced_files) 
    
    print(f"💾 Total de archivos de audio balanceados: {len(balanced_files)}")

    # --- 2. Crear Dataset principal (carga diferida) ---
    # AudioDataset debe poder manejar una lista de (filepath, label)
    # y cargar/procesar el audio cuando se le pide un índice (__getitem__)
    print("🎧 Creando Dataset principal (carga diferida)...")
    # Asegúrate que AudioDataset acepta 'balanced_files' y 'config.window_size'
    # y tiene un método __len__ y __getitem__
    # __getitem__(self, idx) debería devolver (spectrogram_tensor, label_tensor)
    try:
        # Asumiendo que AudioDataset puede inicializarse así:
        full_dataset = AudioDataset(balanced_files, config.window_size, scaler_file="normalization_params.json") 
        # Si AudioDataset necesita directory/dataset_csv, ajusta la inicialización
        # O podrías necesitar una clase Dataset personalizada aquí
        # class BalancedAudioFileDataset(Dataset):
        #     def __init__(self, file_label_list, window_size, scaler_file): ...
        #     def __len__(self): return len(self.file_label_list)
        #     def __getitem__(self, idx):
        #         filepath, label = self.file_label_list[idx]
        #         # Cargar audio, segmentar, crear espectrograma, normalizar...
        #         # spectrogram = ... (código de AudioDataset)
        #         return spectrogram, torch.tensor(label, dtype=torch.float) 
        # full_dataset = BalancedAudioFileDataset(balanced_files, config.window_size, "norm_params.json")

    except NameError:
         print("❌ Error: La clase AudioDataset no está definida o importada.")
         return
    except Exception as e:
         print(f"❌ Error inicializando AudioDataset: {e}")
         return

    # --- 3. Preparar K-Fold ---
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # K=10, shuffle=True
    # Necesitamos las etiquetas para StratifiedKFold
    y_labels = [label for _, label in balanced_files] 
    # Usaremos los índices para dividir el dataset
    indices = np.arange(len(balanced_files))

    # --- 4. Iterar sobre arquitecturas ---
    for i, architecture_encoding in enumerate(architectures):
        if i < checkpoint["last_completed"] + 1: # Corrección: <= a <
            print(f"⏭️ Saltando arquitectura {i+1}/{len(architectures)} (ya procesada según checkpoint)...")
            continue

        print(f"\n🚀 Evaluando arquitectura {i + 1}/{len(architectures)} con 10-Fold Cross-Validation...")
        
        # Validar la arquitectura antes de construir el modelo (opcional pero recomendado)
        # decoded_arch = decode_model_architecture(architecture_encoding) # Asumiendo que existe decode
        # print(f"   Arquitectura decodificada: {decoded_arch}") # Para depuración

        fold_results = []  # Almacenar métricas [loss, acc, prec, rec, f1, spec] por fold
        
        # --- 5. K-Fold Cross-Validation Loop ---
        # Usar `indices` y `y_labels` para `kfold.split`
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, y_labels)):
            print(f"\n Fold {fold+1}/10 - Preparando datos y modelo...")

            # Crear Subsets para este fold específico
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            # Crear DataLoaders para los subsets del fold con collate_fn personalizado
            num_workers = 2
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=True,  # Usar pin_memory si se usa GPU
                collate_fn=collate_fn_skip_none # Usar la función de collate personalizada
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers, 
                pin_memory=True,
                collate_fn=collate_fn_skip_none # Usar la función de collate personalizada
            )

            # Construir y mover el modelo al dispositivo para este fold
            try:
                # BuildPyTorchModel debe aceptar la codificación directamente
                fold_model = BuildPyTorchModel(architecture_encoding, input_shape=(1, 128, 128), verbose=verbose).to(device) # Asumiendo input_shape
            except NameError:
                 print("❌ Error: La clase BuildPyTorchModel no está definida o importada.")
                 break # Salir del loop de folds para esta arquitectura
            except Exception as e:
                 print(f"❌ Error construyendo el modelo para el fold {fold+1}: {e}")
                 break # Salir del loop de folds

            optimizer = optim.Adam(fold_model.parameters(), lr=0.001) # Considerar ajustar lr
            criterion = nn.BCEWithLogitsLoss() # Adecuado para clasificación binaria

            # --- 6. Training Loop por Epoch ---
            for epoch in range(config.epochs):
                fold_model.train()
                running_loss = 0.0
                num_batches = 0
                for batch_idx, (inputs, batch_labels) in enumerate(train_loader):
                    # Verificar si las entradas son válidas (pueden ser None si AudioDataset falla)
                    if inputs is None or batch_labels is None:
                         if verbose: print(f"⚠️ Saltando batch inválido en fold {fold+1}, epoch {epoch+1}, batch {batch_idx}")
                         continue
                         
                    inputs, batch_labels = inputs.to(device), batch_labels.float().to(device)
                    
                    # Asegurar que batch_labels tenga la forma [batch_size, 1] para BCEWithLogitsLoss
                    batch_labels = batch_labels.view(-1, 1) 
                    
                    optimizer.zero_grad()
                    
                    try:
                        outputs = fold_model(inputs)
                        # Asegurar que outputs tenga la forma [batch_size, 1]
                        if outputs.shape != batch_labels.shape:
                             print(f"⚠️ Ajustando forma de salida: {outputs.shape} -> {batch_labels.shape}")
                             outputs = outputs.view(-1, 1) # Intentar ajuste simple
                             if outputs.shape != batch_labels.shape:
                                  raise ValueError(f"Forma de salida incompatible: {outputs.shape} vs {batch_labels.shape}")

                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        num_batches += 1
                    except Exception as e:
                        print(f"❌ Error durante el entrenamiento (Fold {fold+1}, Epoch {epoch+1}, Batch {batch_idx}): {e}")
                        # Considerar saltar el resto de la época o fold si el error es grave
                        # O intentar continuar con el siguiente batch
                        continue # Saltar al siguiente batch


                if num_batches > 0: # Evitar división por cero si todos los batches fallaron
                   avg_epoch_loss = running_loss / num_batches
                   if verbose or (epoch + 1) % 5 == 0: # Imprimir cada 5 épocas o si verbose
                        print(f"  Epoch {epoch+1}/{config.epochs} - Avg Loss: {avg_epoch_loss:.4f}")
                else:
                   print(f"  Epoch {epoch+1}/{config.epochs} - No se completaron batches.")
                   avg_epoch_loss = float('nan') # Indicar que no hubo loss válida


            # --- 7. Validation Loop ---
            fold_model.eval()
            y_true_fold, y_pred_fold = [], []
            final_fold_loss = avg_epoch_loss # Usar la loss promedio de la última época como representativa
            
            with torch.no_grad():
                for inputs, batch_labels in val_loader:
                    # Skip empty batches potentially created by collate_fn
                    if inputs.numel() == 0:
                        print("⏩ Skipping empty validation batch.")
                        continue
                            
                    inputs, batch_labels = inputs.to(device), batch_labels.float().to(device)
                    
                    try:
                        outputs = fold_model(inputs)
                         # Asegurar forma consistente para la predicción
                        if outputs.dim() > 1 and outputs.shape[1] == 1:
                              outputs = outputs.squeeze(1) # Convertir [N, 1] a [N] si es necesario
                        
                        # Aplicar sigmoide y umbral para obtener predicciones binarias
                        predictions = (torch.sigmoid(outputs) > 0.5).int() 
                        
                        y_true_fold.extend(batch_labels.cpu().numpy().flatten().tolist()) # Usar flatten()
                        y_pred_fold.extend(predictions.cpu().numpy().flatten().tolist()) # Usar flatten()
                    except Exception as e:
                         print(f"❌ Error durante la validación (Fold {fold+1}): {e}")
                         # Marcar el fold como inválido o manejar según sea necesario
                         y_true_fold, y_pred_fold = [], [] # Vaciar para indicar fallo
                         break # Salir del loop de validación


            # --- 8. Calcular Métricas del Fold ---
            if len(y_true_fold) > 0 and len(y_pred_fold) > 0: # Solo si la validación tuvo éxito
                accuracy = (np.array(y_true_fold) == np.array(y_pred_fold)).mean()
                precision, recall, f1, specificity = calculate_metrics(y_true_fold, y_pred_fold)
                fold_result = [final_fold_loss, accuracy, precision, recall, f1, specificity]
                print(f"  Fold {fold+1} Resultados - Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, Spec: {specificity:.4f}")
            else:
                print(f"  Fold {fold+1} - Validación fallida o sin datos.")
                # Marcar resultados como NaN o un valor indicador
                fold_result = [float('nan')] * 6 
                
            fold_results.append(fold_result)
            
            # Liberar memoria GPU del modelo del fold (opcional, pero puede ayudar)
            del fold_model, optimizer, train_loader, val_loader, train_subset, val_subset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 9. Promediar resultados de los Folds ---
        if fold_results: # Si hubo al menos un fold
             # Convertir a numpy array, ignorando NaNs para el cálculo de la media
             fold_results_np = np.array(fold_results)
             avg_results = np.nanmean(fold_results_np, axis=0).tolist() 
             
             print("\n📌 Resultados individuales por fold:")
             for idx, result in enumerate(fold_results):
                 print(f"  Fold {idx+1}: Loss={result[0]:.4f}, Acc={result[1]:.4f}, Prec={result[2]:.4f}, Rec={result[3]:.4f}, F1={result[4]:.4f}, Spec={result[5]:.4f}")

             print(f"\n📊 Resultados Promediados (ignora NaNs) - Loss: {avg_results[0]:.4f}, Accuracy: {avg_results[1]:.4f}, F1: {avg_results[4]:.4f}")
             
             # Guardar resultados promediados para la arquitectura actual
             save_results_to_csv(save_file, architecture_encoding, avg_results)
        else:
            print("❌ No se completó ningún fold para esta arquitectura.")
            # Opcional: Guardar un indicador de fallo para esta arquitectura
            # save_results_to_csv(save_file, architecture_encoding, [float('nan')] * 6)


        print(f"✅ Arquitectura {i+1} evaluada. Guardando checkpoint...")
        save_checkpoint(config.checkpoint_file, i) # Guardar el índice de la última arquitectura *completada*

    print("🏁 Entrenamiento completado para todas las arquitecturas.")

# Custom collate function to handle None values from the dataset
def collate_fn_skip_none(batch):
    """
    Collate function that filters out None samples returned by the Dataset.
    """
    # Filter out samples where the data (first element) is None
    batch = [item for item in batch if item[0] is not None]
    # If the batch becomes empty after filtering, return empty tensors
    if not batch:
        # Return structure expected by the training loop, but empty
        return torch.empty(0), torch.empty(0)
    # Use the default collate function on the filtered batch
    return default_collate(batch)
