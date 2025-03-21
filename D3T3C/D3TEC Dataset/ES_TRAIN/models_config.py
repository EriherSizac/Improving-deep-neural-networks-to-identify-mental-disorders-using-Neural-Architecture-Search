# Definición de modelos a entrenar
# Cada modelo está representado como una lista de parámetros codificados

models_to_train = [


    [
      7,
      0,
      0,
      0,
      6,
      36,
      4,
      0,
      1,
      36,
      0,
      0,
      7,
      0,
      0,
      0,
      2,
      0,
      0,
      0,
      0,
      19,
      1,
      0,
      0,
      7,
      0,
      0,
      7,
      0,
      0,
      0,
      4,
      481,
      0,
      0,
      4,
      268,
      0,
      0,
      4,
      512,
      0,
      0,
      1,
      4,
      0,
      0
    ]
  
]

# Puedes agregar más modelos a la lista models_to_train si deseas entrenar múltiples arquitecturas

"""   
    [
      6,
      26,
      1,
      0,
      7,
      0,
      0,
      0,
      0,
      32,
      0,
      0,
      8,
      1,
      1,
      0,
      7,
      0,
      0,
      0,
      0,
      19,
      1,
      3,
      0,
      4,
      0,
      0,
      7,
      0,
      0,
      0,
      4,
      476,
      0,
      0,
      4,
      111,
      0,
      0,
      4,
      512,
      3,
      0,
      1,
      4,
      0,
      0
    ],,

 [
      4,  # Conv2D
      512,  # Filtros
      0,  # Stride
      0,  # Activación (ReLU)
      4,  # Conv2D
      512,  # Filtros
      0,  # Stride
      0,  # Activación (ReLU)
      7,  # BatchNorm
      0,
      0,
      0,
      4,  # Conv2D
      1,  # Filtros
      0,  # Stride
      0,  # Activación (ReLU)
      4,  # Conv2D
      1,  # Filtros
      0,  # Stride
      0,  # Activación (ReLU)
      4,  # Conv2D
      512,  # Filtros
      0,  # Stride
      0,  # Activación (ReLU)
      1,  # MaxPooling
      4,  # Stride
      0,
      0,
      7,  # BatchNorm
      0,
      0,
      0,
      4,  # Conv2D
      512,  # Filtros
      2,  # Stride
      0,  # Activación (ReLU)
      4,  # Conv2D
      512,  # Filtros
      0,  # Stride
      0,  # Activación (ReLU)
      4,  # Conv2D
      512,  # Filtros
      3,  # Stride
      0,  # Activación (ReLU)
      7,  # BatchNorm
      0,
      0,
      0
    ] """