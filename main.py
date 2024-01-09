import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()

# Carga el archivo Parquet 'Clean_australian_user_reviews_FE.parquet' en DataFrame 'df1'
df1 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_user_reviews_FE.parquet',)

# Carga el archivo Parquet 'Clean_output_steam_games.parquet' en DataFrame 'df2'
df2 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_output_steam_games.parquet')

# Carga el archivo Parquet 'Clean_australian_users_items.parquet' en DataFrame 'df3'
df3 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_users_items.parquet')

########################################################################################################
# Carga el modelo de recomendación de juegos que será consumido por la funcion recomendacion_juego(id)
########################################################################################################
# 1. Preparación de datos:
# Combina columnas relevantes en una columna 'combined_features'
df2['combined_features'] = df2['genres'] + ' ' + df2['specs'] + ' ' + df2['developer'] + ' ' + df2['id']

# 2. Vectorización TF-IDF:
# Inicializa el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()
# Transforma la columna 'combined_features' en una matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df2['combined_features'])

# 3. Cálculo de similitud del coseno:
# Calcula la similitud del coseno entre juegos
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
########################################################################################################


# End-point 1
@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str):
    # Devuelve el año con mas horas jugadas para el género.
    # Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
    
    # Elimina registros donde 'release_date' sea nulo
    df_aux = df2.dropna(subset=['release_date'])
    
    # Convierte 'release_date' a tipo datetime
    df_aux['release_date'] = pd.to_datetime(df_aux['release_date'], errors='coerce')
    
    # Elimina registros donde la conversión a datetime falló
    df_aux = df_aux.dropna(subset=['release_date'])
    
    # Extrae el año de 'release_date' y lo pone en columna 'release_year'
    df_aux['release_year'] = df_aux['release_date'].dt.year
    
    # Filtra por el género recibido como argumento 
    df_aux = df_aux[df_aux['genres'].apply(lambda x: genero in x)]
    
    # Une df_aux con df3
    df_aux = df_aux.merge(df3, left_on='id', right_on='item_id')
    
    # Agrupa por año y suma las horas jugadas
    df_aux = df_aux.groupby(df_aux['release_year'])['playtime_forever'].sum().reset_index()
    
    # Encuentra el año con más horas jugadas
    max_playtime_year = df_aux.loc[df_aux['playtime_forever'].idxmax()]['release_year']
    
    return {f"Año de lanzamiento con más horas jugadas para Género {genero}" : str(max_playtime_year)}
    

# End-point 2
@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
    # Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año. 
    # Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

    # Elimina registros donde 'release_date' sea nulo
    df_aux = df2.dropna(subset=['release_date'])

    # Convierte 'release_date' a tipo datetime
    df_aux['release_date'] = pd.to_datetime(df_aux['release_date'], errors='coerce')

    # Elimina registros donde la conversión a datetime falló
    df_aux = df_aux.dropna(subset=['release_date'])

    # Extrae el año de 'release_date' y lo pone en columna 'release_year'
    df_aux['release_year'] = df_aux['release_date'].dt.year

    # Filtra por el género recibido como argumento
    df_aux = df_aux[df_aux['genres'].apply(lambda x: genero in x)]

    # Une df_aux con df3
    df_aux = df_aux.merge(df3, left_on='id', right_on='item_id')

    # Encuentra el usuario con más horas jugadas para el género dado. Para ello...
    # Agrupa por usuario el df_aux, y suma las horas jugadas, ordenando por estas en forma descendente, para tomar de la 1ra fila el max_playtime_user
    max_playtime_user = df_aux.groupby(['user_id'])['playtime_forever'].sum().reset_index().sort_values(by='playtime_forever', ascending=False).iloc[0]['user_id']

    # Filtra por el usuario con más horas jugadas para el género recibido como argumento
    df_aux = df_aux[df_aux['user_id'] == max_playtime_user]

    # Genera la lista de acumulación de horas jugadas por año para el usuario y género 
    # Toca pasar a horas la estadística en 'playtime_forever', pues está en minutos. https://steamcommunity.com/discussions/forum/1/3814039097896647393/
    playtime_by_year = df_aux.groupby(['release_year'])['playtime_forever'].sum().reset_index().sort_values(by='release_year', ascending=False)
    playtime_by_year['playtime_forever_Hrs'] = round(playtime_by_year['playtime_forever'] / 60, 0)
    playtimeHrs_by_year_list = [{"Año": int(row['release_year']), "Horas": int(row['playtime_forever_Hrs'])} for index, row in playtime_by_year.iterrows()]

    return {f"Usuario con más horas jugadas para Género {genero}": max_playtime_user, "Horas jugadas": playtimeHrs_by_year_list}
    

# End-point 3
@app.get("/UsersRecommend/{annio}")
def UsersRecommend(annio: int): 
    # Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios 
    # positivos/neutrales)
    # Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

    # Filtra las recomendaciones por el año proporcionado
    df_aux = df1[df1['posted'].str.contains(str(annio))].copy()

    # Agrega columna 'good_review'. Valores True si 'recommend' es True y 'sentiment_analysis' es 1 o 2
    df_aux['good_review'] = np.where((df_aux['recommend'] == True) & (df_aux['sentiment_analysis'].isin([1, 2])), True, False)

    
    # Agrupa por juego (item_id) y suma las recomendaciones positivas, luego ordena dejando arriba los más recomendados
    # NOTA: si sumas una columna con True/False, el resultado obtenido es similar a contar los True, puesto que cada True se suma como 1 y cada False como 0
    df_aux = df_aux.groupby(['item_id'])['good_review'].sum().reset_index().sort_values(by='good_review', ascending=False)

    # Une df_aux con df2 a través del id del juego en uno y otro DataFrame. Es necesario para obtener el nombre del juego
    df_aux = df_aux.merge(df2, left_on='item_id', right_on='id')
    
    # Arma la lista con el top 3 de los juegos mas recomendados
    top3_most_recommended = [{"Puesto 1" : df_aux.iloc[0]['app_name']}, 
                             {"Puesto 2" : df_aux.iloc[1]['app_name']}, 
                             {"Puesto 3" : df_aux.iloc[2]['app_name']}]

    return top3_most_recommended


# End-point 4
@app.get("/UsersWorstDeveloper/{annio}")
def UsersWorstDeveloper(annio: int): 
    # Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = 
    # False y comentarios negativos)
    # Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

    # Filtra las recomendaciones por el año proporcionado
    df_aux = df1[df1['posted'].str.contains(str(annio))].copy()
    
    # Agrega columna 'bad_review'. Valores True si 'recommend' es False y 'sentiment_analysis' es 0
    df_aux['bad_review'] = np.where((df_aux['recommend'] == False) & (df_aux['sentiment_analysis'] == 0), True, False)

    # Agrupa por juego (item_id) y suma las recomendaciones negativas, luego ordena dejando arriba los peores recomendados
    # NOTA: si sumas una columna con True/False, el resultado obtenido es similar a contar los True, puesto que cada True se suma como 1 y cada False como 0
    df_aux = df_aux.groupby(['item_id'])['bad_review'].sum().reset_index().sort_values(by='bad_review', ascending=False)

    # Une df_aux con df2 a través del id del juego en uno y otro DataFrame. Es necesario para obtener el nombre del desarrollador
    df_aux = df_aux.merge(df2, left_on='item_id', right_on='id')
    
    # Arma la lista con el top 3 de las desarrolladoras con juegos menos recomendados
    top3_least_recommended_dev = [{"Puesto 1" : df_aux.iloc[0]['developer']}, 
                                  {"Puesto 2" : df_aux.iloc[1]['developer']}, 
                                  {"Puesto 3" : df_aux.iloc[2]['developer']}]

    return top3_least_recommended_dev


# End-point 5
@app.get("/sentiment_analysis/{empresa_desarrolladora}")
def sentiment_analysis(empresa_desarrolladora: str): 
    # Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista 
    # con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento 
    # como valor.
    # Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

    # Une df1 con df2 a través del id del juego en uno y otro DataFrame.
    df_aux = df1.merge(df2, left_on='item_id', right_on='id')

    # Filtra los registros en df_aux donde la columna 'developer' coincida con la 'empresa_desarrolladora' pasada como argumento
    df_aux = df_aux[df_aux['developer'] == empresa_desarrolladora]
    
    # Cuenta los valores únicos en la columna 'sentiment_analysis' y guarda los recuentos en variables separadas
    Negative = "Negative = " + str(df_aux['sentiment_analysis'].value_counts().get(0, 0))
    Neutral = "Neutral = " + str(df_aux['sentiment_analysis'].value_counts().get(1, 0))
    Positive = "Positive = " + str(df_aux['sentiment_analysis'].value_counts().get(2, 0))

    return {empresa_desarrolladora: [Negative, Neutral, Positive]} 
    

# End-point para obtener recomendaciones
@app.get("/recomendacion_juego/{id}")
def recomendacion_juego(id: str):
    # Obtiene el índice del id de juego proporcionado
    idx = df2[df2['id'] == id].index[0]
    
    # Obtiene las puntuaciones de similitud del juego en cuestión con otros juegos
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordena los juegos según las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtiene los índices de los juegos más similares (excluyendo el propio juego)
    similar_games_indices = [i[0] for i in sim_scores[1:6]]  # Recomendar los 5 juegos más similares
    
    # Devuelve los nombres de los juegos recomendados
    #return df['app_name'].iloc[similar_games_indices]

    # Crea una lista de diccionarios con 'id' como clave y 'app_name' como valor
    recommended_games = [{df2['id'].iloc[ind] : df2['app_name'].iloc[ind]} for ind in similar_games_indices]

    return recommended_games

