import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Librosa la sabrosa')

'''
Contenido obtenido de: www.kaggle.com/datasets/rounakbanik/pokemon
'''

url = 'https://ww3.lectulandia.com/'
PokemonDB = pd.read_csv(url)

st.subheader('Raw data')
st.write(PokemonDB.head())
st.subheader('data describe')
st.write(PokemonDB.describe())
st.subheader('data map')

fig, axs = plt.subplots(figsize=(8, 5))

# Histograma
sns.histplot(PokemonDB['Type'])
axs.set_title('Histograma de el tipo de Pokémon')
axs.set_xlabel('Sucursales')
axs.set_ylabel('Frecuencia')

plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig)


st.write(PokemonDB['Type'].describe())
