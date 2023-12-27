from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo salvo
modelo = load_model("meu_modelo.keras")

def load_image(caminhoImagem):
    # Carregar a imagem e redimensionar para o tamanho esperado pelo modelo
    img = image.load_img(caminhoImagem, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Pré-processar a imagem
    img_array = img_array / 255.0

    return img_array

# Substituir 'caminho/para/sua/imagem.jpg' pelo caminho da imagem que você deseja testar
caminho_imagem_teste = 'training/cachorro/cachorro.jpg'

# Carregar e preparar imagem
nova_img = load_image(caminho_imagem_teste)

# Fazer previsão usando modelo
predicoes = modelo.predict(nova_img)

# Onde as 'predicoes' é um vetor de probabilidade para cada classe
print("Previsões (probabilidades por classe): ", predicoes)

# Caso a classe seja categórica, você pode querer o índice da classe com maior probabilidade
classe_predita = np.argmax(predicoes[0])

# Se você quiser o rótulo da classe predita, você precisa do mapeamento de índice para nome da classe
# Esse mapeamento é o 'class_indices' que você obteve enquanto carregava os dados do treinamento
# Dicionário que mapeia índices de previsão para rótulos de classe
rotulo_classe_invertido = {0: "Cachorro", 1: "Gato"}

nome_classe_predita = rotulo_classe_invertido[classe_predita]
print("Nome da classe predita : ", nome_classe_predita)
