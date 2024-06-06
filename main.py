import streamlit as st
import pandas as pd
import Levenshtein
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity

file_path = 'Libros.parquet'
libros = pd.read_parquet(file_path)

# Cargar el tokenizador y el modelo BERT multilingüe
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
# Definir la función para obtener los embeddings de un texto
def get_embedding(text):
    # Tokenizar el texto, convirtiéndolo en tensores y asegurando la longitud máxima y el padding
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # Desactivar el cálculo del gradiente ya que estamos en modo de inferencia
    with torch.no_grad():
        # Pasar los tokens a través del modelo BERT para obtener los embeddings
        outputs = model(**tokens)
    # Promediar los embeddings de los tokens para obtener un solo vector de representación
    return outputs.last_hidden_state.mean(dim=1)

# Obtener los embeddings de las descripciones de los libros
libros['Embedding'] = libros['Descripcion'].apply(get_embedding)

# Definir la función para recomendar libros basados en la entrada del usuario
def recomendar_libros(user_input):
    # Obtener el embedding de la entrada del usuario
    user_embedding = get_embedding(user_input)
    # Calcular la similitud del coseno entre el embedding del usuario y los embeddings de los libros
    libros['Similitud'] = libros['Embedding'].apply(lambda x: cosine_similarity(user_embedding, x).item())
    # Ordenar los libros por la similitud en orden descendente
    libros_ordenados = libros.sort_values(by='Similitud', ascending=False)
    # Retornar los primeros 'top_n' libros más similares
    return libros_ordenados.head(3)

#--------RECOMENDACION DIRECTA----------#
# Barra de búsqueda
input_user = st.text_input('', value='¿Qué tienes ganas de leer hoy?')

if input_user != '¿Qué tienes ganas de leer hoy?':
    # Obtener las recomendaciones
    recomendaciones = recomendar_libros(input_user)
    #Mostrar las recomendaciones
    st.write("Recomendaciones:")
    for index, row in recomendaciones.iterrows():
        st.write(f"**Título:** {row['Titulo']}")
        st.write(f"**Autor:** {row['Autor']}")
        st.write(f"**Género:** {', '.join(row['Genero'])}")
        st.write(f"**Descripción:** {row['Descripcion']}")
        st.write(f"**Similitud:** {row['Similitud']:.4f}")
        st.write("---")

          
    
    
#--------GENERO----------#
# Contar la cantidad de autores de cada autor
gender_counts = libros['Genero'].explode().value_counts()

# Barra de búsqueda
input_user_g = st.text_input('', value='¿Qué genero tienes ganas de leer hoy?')
# Lista de géneros únicos para la coincidencia difusa
generos = set([g for sublist in libros['Genero'] for g in sublist])

if input_user_g != '¿Qué genero tienes ganas de leer hoy?':
    # Inicializar variables para almacenar el autor con la menor distancia y la distancia mínima
    min_distance_g = float('inf')
    closest_gender = None
    
    # Calcular la distancia de Levenshtein para cada autor
    for genero in generos:
        distance_g = Levenshtein.distance(genero, input_user_g)
        if distance_g < min_distance_g:
            min_distance_g = distance_g
            closest_gender = genero
    
    # Filtrar el DataFrame para mostrar solo los libros del autor más cercano
    filtered_df_g = libros[libros['Genero'].apply(lambda x: closest_gender in x)]
    
    # Crear columnas para mostrar lado a lado
    col3, col4 = st.columns(2)
    
    # Mostrar el DataFrame filtrado en la primera columna
    with col3:
        st.write(f"Libros recomendados del genero {closest_gender}:")
        st.dataframe(filtered_df_g)
    
    # Mostrar la cantidad de autores de cada autor en la segunda columna
    with col4:
        st.write("Cantidad de libros por genero:")
        st.dataframe(gender_counts)
        
        
        
        
        
#--------AUTOR----------#
# Contar la cantidad de autores de cada autor
author_counts = libros['Autor'].explode().value_counts()

# Barra de búsqueda
input_user_a = st.text_input('', value='¿Qué autor tienes ganas de leer hoy?')
# Lista de géneros únicos para la coincidencia difusa
autores = set([g for sublist in libros['Autor'] for g in sublist])

if input_user_a != '¿Qué autor tienes ganas de leer hoy?':
    # Inicializar variables para almacenar el autor con la menor distancia y la distancia mínima
    min_distance_a = float('inf')
    closest_author = None
    
    # Calcular la distancia de Levenshtein para cada autor
    for autor in autores:
        distance_a = Levenshtein.distance(autor, input_user_a)
        if distance_a < min_distance_a:
            min_distance_a = distance_a
            closest_author = autor
    
    # Filtrar el DataFrame para mostrar solo los libros del autor más cercano
    filtered_df_a = libros[libros['Autor'].apply(lambda x: closest_author in x)]
    
    # Crear columnas para mostrar lado a lado
    col1, col2 = st.columns(2)
    
    # Mostrar el DataFrame filtrado en la primera columna
    with col1:
        st.write(f"Libros recomendados del autor {closest_author}:")
        st.dataframe(filtered_df_a)
    
    # Mostrar la cantidad de autores de cada autor en la segunda columna
    with col2:
        st.write("Cantidad de libros por autor:")
        st.dataframe(author_counts)