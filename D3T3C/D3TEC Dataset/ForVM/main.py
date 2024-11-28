
from data_processing import load_audio_data, preprocess_audio, train_test_split_audio
from model_utils import encode_layer_params, decode_layer_params, fixArch
from keras.models import Sequential
from data_processing import load_audio_data, preprocess_audio, train_test_split_audio, generate_and_train_models

if __name__ == '__main__':
    audio_dir = './audio_data'
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



