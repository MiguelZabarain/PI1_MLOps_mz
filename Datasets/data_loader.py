import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():

    # Carga el archivo Parquet 'Clean_australian_user_reviews_FE.parquet' en DataFrame 'df1'
    df1 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_user_reviews_FE.parquet')

    # Carga el archivo Parquet 'Clean_output_steam_games.parquet' en DataFrame 'df2'
    df2 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_output_steam_games.parquet')


    # Carga el archivo Parquet 'Clean_australian_users_items.parquet' en DataFrame 'df3'
    #df3 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_users_items.parquet')

    # Define filtro para cargar solo los registros con 'playtime_forever' mayor que 1000
    filters = [('playtime_forever', '>', 0)]

    # Especifica las columnas a cargar en el DataFrame
    columns_to_keep = ['item_id', 'item_name', 'playtime_forever', 'user_id']

    # Carga registros desde el archivo Parquet aplicando el filtro y especificando columnas
    df3 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_users_items.parquet',
                        columns=columns_to_keep, filters=filters)


    ########################################################################################################
    # Carga el modelo de recomendación de juegos que será consumido por la funcion recomendacion_juego(id)
    ########################################################################################################
    # 1. Preparación de datos:
    # Función que combina columnas relevantes de df2 en una columna 'combined_features' y devuelve la 5ta parte.
    # Esto fue necesario para optimizar el uso de memoria. Con los ajustes al código original se bajó de 4 GB
    # a menos de 400 MB
    def CombFeatures():
        total_rows = len(df2)
        start_row = 0 
        end_row = total_rows - 1 #// 10
        df = pd.DataFrame()
        df['combined_features'] = df2['genres'] + ' ' + df2['specs'] + ' ' + df2['developer'] + ' ' + df2['id']
        return df.loc[start_row:end_row, 'combined_features']

    # 2. Vectorización TF-IDF:
    # Inicializa el vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    # Transforma la columna 'combined_features' en una matriz TF-IDF
    #tfidf_matrix = tfidf_vectorizer.fit_transform(df2['combined_features'])
    tfidf_matrix = tfidf_vectorizer.fit_transform(CombFeatures())

    # 3. Cálculo de similitud del coseno:
    # Calcula la similitud del coseno entre juegos
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    ########################################################################################################

    return df1, df2, df3, cosine_sim

# Export the loaded data
df1, df2, df3, cosine_sim = load_data()