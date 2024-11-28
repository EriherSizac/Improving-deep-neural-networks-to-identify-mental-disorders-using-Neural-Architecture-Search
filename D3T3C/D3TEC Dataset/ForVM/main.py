from data.data_processing import load_audio_data, preprocess_audio, train_test_split_audio
from model.model_generator import generate_random_architecture, build_tf_model_from_dict
from model.architecture_corrections import fixArch
from model.encoding import encode_layer_params, decode_layer_params
from config.config import layer_type_options

def main():
    audio_dir = './audio_data'
    sample_rate = 16000
    window_size = 5

    # Arquitecturas predefinidas
    predefined_architectures = [
        [0, 30, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 16, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 5, 0, 0, 0, 4, 256, 0, 0, 3, 1, 0, 0, 4, 1, 2, 0],
        [1, 0, 0, 0, 0, 16, 0, 1, 1, 0, 0, 0, 0, 8, 0, 1, 1, 0, 0, 0, 5, 0, 0, 0, 4, 32, 1, 0, 4, 1, 2, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0],
        [0, 32, 0, 1, 1, 0, 0, 0, 0, 32, 0, 1, 2, 1, 0, 0, 0, 32, 0, 1, 1, 0, 0, 0, 0, 32, 0, 1, 0, 32, 0, 1, 1, 0, 0, 0, 0, 32, 0, 1, 0, 32, 0, 1, 1, 0, 0, 0]
    ]

    # Carga y preprocesamiento de datos
    audio_dict, sample_rate = load_audio_data(audio_dir, window_size, sample_rate)
    audio_dict = preprocess_audio(audio_dict, sample_rate)
    X, Y = train_test_split_audio(audio_dict)

    # Generar arquitectura aleatoria como ejemplo
    random_arch = generate_random_architecture()
    print("Generated Architecture:", random_arch)

    # Construir y mostrar modelo
    model = build_tf_model_from_dict(random_arch)
    model.summary()

if __name__ == '__main__':
    main()
