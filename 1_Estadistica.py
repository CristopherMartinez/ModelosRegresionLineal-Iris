import pandas as pd
import streamlit as st
import numpy as np
import Pages
import Pages.graficas
import Pages.analisisEstadistico
import Pages.modelo
#!pip install scikit-learn

#Se agrego un diccionario para relacionar las paginas
pages = {
    "Analisis Estadistico": Pages.analisisEstadistico.show,
    "Graficas": Pages.graficas.show,
    "Modelo": Pages.modelo.show,
}

selection = st.sidebar.radio("Páginas:", list(pages.keys()))
pages[selection]()
