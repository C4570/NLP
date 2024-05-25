# Libreries

# !pip install streamlit
# !pip install requests
# !pip install beautifulsoup4

import requests
from bs4 import BeautifulSoup
import streamlit as st

def scrape_lectulandia():
    url = "https://ww3.lectulandia.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encuentra los elementos de la página web que contienen la información que quieres extraer.
    # Por ejemplo, si los títulos de los libros están en etiquetas <h2>, podrías hacer:
    book_titles = soup.find_all('h2')

    # Luego puedes extraer el texto de estos elementos y agregarlo a tu dataset.
    for title in book_titles:
        print(title.text)
        
        
    
def main(book_titles):
    st.title("Mi Aplicación de Lectura")
    respuesta = st.text_input("¿Qué tienes ganas de leer hoy?")
    if respuesta:
        st.write(f"¡Genial! Buscaremos libros relacionados con {respuesta}.")
        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(book_titles)
        st.write("Aquí tienes algunas recomendaciones:")
        
        


if __name__ == "__main__":
    scrape_lectulandia()
    main(book_titles)



