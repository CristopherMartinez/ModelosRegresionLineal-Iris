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
    import pandas as pd
    import streamlit as st
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    st.title("Modelo")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    df = pd.read_csv(url, names=names)
    x_train, x_test, y_train, y_test = train_test_split(df[df.columns[0:4]], df[df.columns[-1]], test_size=0.2)
    modelos = []
    modelo = LogisticRegression(random_state=0).fit(x_train, y_train)
    #st.write(modelo.score(x_test, y_test))
    #st.write(modelo.predict(x_test))

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    resultados = cross_val_score(modelo, x_train, y_train,cv=kfold, scoring="accuracy")
    #print(resultados)

    modeloKN = KNeighborsClassifier(n_neighbors=3)
    modeloKN.fit(x_train, y_train)
    #st.write(modeloKN.score(x_test, y_test))
    #st.write(modeloKN.predict(x_test))
    st.header('Modelo puntaje de Precisión:')
    st.write(modelo.score(x_test, y_test))
    st.header('Modelo Predicción:')
    st.write(modelo.predict(x_test))

#Se agrego un diccionario para relacionar las paginas
pages = {
    "Analisis Estadistico": analisisEstadistico,
    "Graficas": graficas,
    "Modelo": modelo,
}

selection = st.sidebar.radio("Páginas:", list(pages.keys()))
pages[selection]()
