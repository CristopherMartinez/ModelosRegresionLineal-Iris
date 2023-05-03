import pandas as pd
import streamlit as st
import numpy as np
import Pages


def home():
    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    dfIris = pd.read_csv(url)
    print(dfIris.head(10))
    st.title("Analisis estadistico Iris Dataset")
    st.dataframe(dfIris.head(10))
    st.write("Estadisticas")
    st.write("Filas, columnas:")
    st.write(dfIris.shape)
    st.write("Describe:")
    st.dataframe(dfIris.describe())
    st.write("Clases:")
    st.write(dfIris["variety"].value_counts())

def about():
    st.title("Acerca de")
    st.write("Esta es una aplicación de ejemplo para Streamlit.")
    # resto del código para la página "Acerca de"

def contact():
    st.title("Contacto")
    st.write("Puedes contactarnos en contact@example.com.")
    # resto del código para la página "Contacto"


# definir un diccionario para asociar cada página con su función correspondiente
pages = {
    "Inicio": home,
    "Acerca de": about,
    "Contacto": contact,
}


# mostrar la barra lateral con las opciones de página
selection = st.sidebar.radio("Páginas:", list(pages.keys()))


# llamar a la función correspondiente para mostrar la página seleccionada
pages[selection]()
