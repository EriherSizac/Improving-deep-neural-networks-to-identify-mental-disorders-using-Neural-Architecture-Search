import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from tensorflow.keras.utils import to_categorical
import pandas as pd
from utils import encode_model_architecture, fixArch, decode_model_architecture, build_tf_model_from_dict

# Función para cargar y preparar los datos
def prepare_data(dataset_path, target_shape=(128, 128, 1)):
    X = np.load(os.path.join(dataset_path, "X.npy"))  # Cargar espectrogramas
    Y = np.load(os.path.join(dataset_path, "Y.npy"))  # Cargar etiquetas

    # Normalización y reshape
    X = X / np.max(X)
    X = X.reshape((-1, *target_shape))

    # Dividir en conjunto de entrenamiento/validación y prueba
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train_val, X_test, Y_train_val, Y_test

# Métricas adicionales
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Entrenamiento y evaluación
def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=50):
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy", "Precision", "Recall"])
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), verbose=0)
    results = model.evaluate(X_test, Y_test, verbose=0)

    # Predicciones para métricas adicionales
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = results[1]
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    specificity = specificity_score(Y_test, Y_pred)

    return [results[0], accuracy, precision, recall, f1, specificity]

# Generar y entrenar modelos
def generate_and_train_models(predefined_architectures, num_random_models, X_train_val, X_test, Y_train_val, Y_test, target_shape=(128, 128, 1), epochs=50, k_folds=5):
    results_data = []
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Entrenar arquitecturas predefinidas
    for i, architecture in enumerate(predefined_architectures):
        print(f"Evaluando arquitectura predefinida {i + 1}/{len(predefined_architectures)}...")
        repaired_architecture = fixArch(architecture)
        decoded_model_dict = decode_model_architecture(repaired_architecture)

        fold_results = []
        for fold, (train_index, val_index) in enumerate(kfold.split(X_train_val)):
            print(f"Fold {fold + 1}/{k_folds}...")
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
            tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=target_shape)
            fold_results.append(train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs))
        
        avg_results = np.mean(fold_results, axis=0)
        results_data.append([repaired_architecture] + list(avg_results))
    
    # Generar y entrenar modelos aleatorios
    for i in range(num_random_models):
        print(f"Generando modelo aleatorio {i + 1}/{num_random_models}...")
        random_architecture = generate_random_architecture()
        repaired_architecture = fixArch(random_architecture)
        decoded_model_dict = decode_model_architecture(repaired_architecture)

        fold_results = []
        for fold, (train_index, val_index) in enumerate(kfold.split(X_train_val)):
            print(f"Fold {fold + 1}/{k_folds}...")
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
            tf_model = build_tf_model_from_dict(decoded_model_dict, input_shape=target_shape)
            fold_results.append(train_and_evaluate_model(tf_model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs))
        
        avg_results = np.mean(fold_results, axis=0)
        results_data.append([repaired_architecture] + list(avg_results))
    
    # Guardar resultados en CSV
    columns = ["Encoded Architecture", "Loss", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
    results_df = pd.DataFrame(results_data, columns=columns)
    results_df.to_csv("training_results.csv", index=False)
    print("Resultados guardados en 'training_results.csv'")

# Generar arquitectura aleatoria
def generate_random_architecture():
    num_layers = np.random.randint(1, 13)  # 1 a 12 capas
    genes = []
    for _ in range(num_layers):
        layer_type = np.random.randint(0, 4)  # Tipos de capas: 0, 1, 2, 3
        param1 = np.random.randint(0, 32)    # Parámetro 1
        param2 = np.random.randint(0, 32)    # Parámetro 2
        param3 = np.random.randint(0, 32)    # Parámetro 3
        genes.extend([layer_type, param1, param2, param3])
    while len(genes) < 48:  # Rellenar hasta 48 alelos
        genes.append(0)
    return genes

# Uso del script
if __name__ == "__main__":
    predefined_architectures = [
        [0, 30, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0, 4, 1, 2, 0],
        # Agregar más arquitecturas predefinidas...
    ]
    dataset_path = "./SM-27"
    X_train_val, X_test, Y_train_val, Y_test = prepare_data(dataset_path, target_shape=(128, 128, 1))
    generate_and_train_models(predefined_architectures, num_random_models=250, X_train_val=X_train_val, X_test=X_test, Y_train_val=Y_train_val, Y_test=Y_test, target_shape=(128, 128, 1), epochs=50, k_folds=5)
