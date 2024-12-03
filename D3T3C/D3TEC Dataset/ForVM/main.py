from data.data_processing import load_audio_data, preprocess_audio, train_test_split_audio
from model.model_generator import generate_random_architecture, build_tf_model_from_dict
from model.architecture_corrections import fixArch
from model.encoding import encode_layer_params, decode_layer_params
from config.config import layer_type_options
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from model.model_utils import encode_model_architecture, decode_model_architecture
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
# Función de especificidad
class Config:
    def __init__(self, architecture='random', epochs=50, sample_rate=None, time=5, n_splits=5, window_size=5):
        self.architecture = architecture
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.time = time
        self.n_splits = n_splits
        self.window_size = window_size
        
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Entrenamiento y evaluación de cada arquitectura decodificada
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config):
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy", 'Precision', 'Recall'])
    model.fit(X_train, Y_train, epochs=config.epochs, validation_data=(X_val, Y_val), verbose=0)
    results = model.evaluate(X_test, Y_test, verbose=0)

    # Obtener predicciones para métricas adicionales
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = results[1]
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    specificity = specificity_score(Y_test, Y_pred)

    return [results[0], accuracy, precision, recall, f1, specificity]

def generate_and_train_models(predefined_architectures, num_random_models=250, directory='./SM-27', target_shape=(128, 128, 1), use_kfold=True):
    results_data = []
    config = Config(epochs=50)

    # Cargar y preprocesar datos de audio
    print("Cargando y preprocesando datos de audio...")
    audio_dict, sample_rate = load_audio_data(directory, config.window_size, config.sample_rate)
    audio_dict = preprocess_audio(audio_dict, sample_rate)
    X, Y = train_test_split_audio(audio_dict)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    if use_kfold:
        kfold = KFold(n_splits=config.n_splits, shuffle=True)

    # Entrenar y evaluar arquitecturas predefinidas
    for i, architecture in enumerate(predefined_architectures):
        print(f"\nEvaluando arquitectura predefinida {i + 1}/{len(predefined_architectures)}...")
        evaluate_and_store_model(architecture, X_train_val, X_test, Y_train_val, Y_test, config, use_kfold, kfold, target_shape, results_data)

    # Generar, entrenar y evaluar arquitecturas aleatorias
    for i in range(num_random_models):
        print(f"\nGenerando y evaluando modelo aleatorio {i + 1}/{num_random_models}...")
        random_architecture = generate_random_architecture()
        encoded_architecture = encode_model_architecture(random_architecture, max_alleles=48)
        repaired_architecture = fixArch(encoded_architecture)
        evaluate_and_store_model(repaired_architecture, X_train_val, X_test, Y_train_val, Y_test, config, use_kfold, kfold, target_shape, results_data)

    # Guardar resultados en CSV
    columns = ["Encoded Architecture", "Loss", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
    results_df = pd.DataFrame(results_data, columns=columns)
    results_df.to_csv("./model_results_combined_50_epochs.csv", index=False)
    print("Resultados guardados en 'model_results_combined.csv'")

# Función para evaluar y almacenar los resultados de un modelo
def evaluate_and_store_model(architecture, X_train_val, X_test, Y_train_val, Y_test, config, use_kfold, kfold, target_shape, results_data):
    repaired_architecture = fixArch(architecture)
    decoded_model_dict = decode_model_architecture(repaired_architecture)
    model_results = [repaired_architecture]

    if use_kfold:
        fold_results = []
        for fold, (train_index, val_index) in enumerate(kfold.split(X_train_val)):
            print(f"Entrenando fold {fold + 1}/{config.n_splits}...")
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
            tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=(target_shape[0], target_shape[1], 1))
            fold_results.append(train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config))
            print(f"Fold {fold + 1} completado.")

        avg_results = np.mean(fold_results, axis=0)
        model_results.extend(avg_results)

    else:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)
        tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=(target_shape[0], target_shape[1], 1))
        single_run_results = train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, config)
        model_results.extend(single_run_results)
        print("Modelo evaluado sin K-Fold Cross Validation.")

    results_data.append(model_results)
 



def main():
    audio_dir = './SM-27'
    sample_rate = 16000
    window_size = 5

      
    # Definición de arquitecturas predefinidas
    predefined_architectures = [
        [0, 30, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0, 4, 1, 2, 0],
        [1, 0, 0, 0, 0, 16, 0, 1, 1, 0, 0, 0, 0, 8, 0, 1, 1, 0, 0, 0, 5, 0, 0, 0, 4, 32, 1, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0],
        [0, 32, 0, 1, 1, 0, 0, 0, 0, 32, 0, 1, 2, 1, 0, 0, 0, 32, 0, 1, 1, 0, 0, 0, 0, 32, 0, 1, 0, 32, 0, 1, 1, 0, 0, 0, 0, 32, 0, 1, 0, 32, 0, 1, 1, 0, 0, 0]
    ]


    # Ejecutar la generación y entrenamiento de modelos, incluyendo arquitecturas predefinidas y modelos aleatorios
    generate_and_train_models(predefined_architectures, num_random_models=270, target_shape=(128, 128, 1), use_kfold=True)



if __name__ == '__main__':
    main()
