import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Caminho para o diretório que contém suas pastas de treinamento
diretorio_base = 'training'

data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Parâmetros
img_size = (224, 224)
batch_size = 32

# Criar conjuntos de treinamento e validação
train_generator = data_generator.flow_from_directory(
    diretorio_base,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Utilizando 'categorical' para rótulos de várias classes
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    diretorio_base,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Construir o modelo CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(224, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # Alteração aqui

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Mudança aqui
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator, validation_data=validation_generator, epochs=15)

# Salvar o modelo treinado no formato Keras
model.save('meu_modelo.keras')
