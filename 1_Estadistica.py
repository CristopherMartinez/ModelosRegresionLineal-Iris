import pandas as pd
import streamlit as st
import numpy as np
import Pages


def analisisEstadistico():
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

def graficas():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    import Pages 
    
    st.title("Graficas")
    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    dfIris = pd.read_csv(url)

    st.subheader("Histogramas")
    for a in range(0, len(dfIris.columns)):
        #fig = px.histogram(go.Box(y=dfIris[a].values, name=dfIris[a]))
        fig = px.histogram(dfIris, x=dfIris.columns[a], nbins=20, title=dfIris.columns[a])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Grafica de correlacion")
    fig2 = px.scatter_matrix(dfIris, dimensions=dfIris.columns[0:4], color="variety")
    st.plotly_chart(fig2,use_container_width=True)
    st.subheader("Grafica de correlacion - Mapa de calor")
    dfIris = dfIris.drop('variety', axis=1)
    df_corr = dfIris.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=df_corr.columns,
            y=df_corr.index,
            z=np.array(df_corr)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def modelo():
    st.title("Modelo")
    #st.write("Puedes contactarnos en contact@example.com.")
    # resto del código para la página "Contacto"


# definir un diccionario para asociar cada página con su función correspondiente
pages = {
    "Analisis Estadistico": analisisEstadistico,
    "Graficas": graficas,
    "Modelo": modelo,
}


# mostrar la barra lateral con las opciones de página
selection = st.sidebar.radio("Páginas:", list(pages.keys()))


# llamar a la función correspondiente para mostrar la página seleccionada
pages[selection]()
