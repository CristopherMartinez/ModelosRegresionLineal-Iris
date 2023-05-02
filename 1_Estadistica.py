import pandas as pd
import streamlit as st

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