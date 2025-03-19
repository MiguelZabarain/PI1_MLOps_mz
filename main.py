import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import sys
from datasets.data_loader import df1, df2, df3, cosine_sim
from utils.helpers import trace, logger
import uvicorn

sys.stdout.reconfigure(encoding='utf-8') 
app = FastAPI()

# End-point 1
@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str):
    """
    Returns the year with the most hours played for a specific game genre.
    
    Parameters:
    -----------
    genero : str
        The game genre to analyze (e.g., "Action", "Adventure", "Strategy")
    
    Returns:
    --------
    dict
        A dictionary containing the year with the most hours played for the specified genre
        Format: {"Year of release with most hours played for Genre X" : "YYYY"}
    
    Raises:
    -------
    Exception
        If the genre doesn't exist or if there's an error processing the data
    """
    try:
        # Remove records where 'release_date' is null
        df_aux = df2.dropna(subset=['release_date'])
        
        # Convert 'release_date' to datetime type
        df_aux['release_date'] = pd.to_datetime(df_aux['release_date'], errors='coerce')
        
        # Remove records where datetime conversion failed
        df_aux = df_aux.dropna(subset=['release_date'])
        
        # Extract the year from 'release_date' and put it in 'release_year' column
        df_aux['release_year'] = df_aux['release_date'].dt.year
        
        # Filter by the genre received as argument
        df_aux = df_aux[df_aux['genres'].apply(lambda x: genero in x)]
        
        # Merge df_aux with df3
        df_aux = df_aux.merge(df3, left_on='id', right_on='item_id')
        
        # Group by year and sum the hours played
        df_aux = df_aux.groupby(df_aux['release_year'])['playtime_forever'].sum().reset_index()
        
        # Find the year with most hours played
        max_playtime_year = df_aux.loc[df_aux['playtime_forever'].idxmax()]['release_year']
        
        return {f"Year of release with most hours played for Genre {genero}" : str(max_playtime_year)}
    except Exception as e:
        logger.exception(f"Argument: genero = {genero}") #Joyuela: Log the exception
        raise e
    

# End-point 2
@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
        """
        Returns the user who has accumulated the most hours played for a specific genre,
        along with a breakdown of hours played by year.
    
        Parameters:
        -----------
        genero : str
            The game genre to analyze (e.g., "Action", "Adventure", "Strategy")
    
        Returns:
        --------
        dict
            A dictionary containing:
            - The user ID with the most hours played for the specified genre
            - A list of dictionaries with hours played per year
            Format: {
                "User with most hours played for Genre X": "user_id",
                "Hours played": [
                    {"Year": 2013, "Hours": 203},
                    {"Year": 2012, "Hours": 100},
                    ...
                ]
            }
    
        Raises:
        -------
        Exception
            If the genre doesn't exist or if there's an error processing the data
        """
        try:
            # Remove records where 'release_date' is null
            df_aux = df2.dropna(subset=['release_date'])

            # Convert 'release_date' to datetime type
            df_aux['release_date'] = pd.to_datetime(df_aux['release_date'], errors='coerce')

            # Remove records where datetime conversion failed
            df_aux = df_aux.dropna(subset=['release_date'])

            # Extract the year from 'release_date' and put it in 'release_year' column
            df_aux['release_year'] = df_aux['release_date'].dt.year

            # Filter by the genre received as argument
            df_aux = df_aux[df_aux['genres'].apply(lambda x: genero in x)]

            # Merge df_aux with df3
            df_aux = df_aux.merge(df3, left_on='id', right_on='item_id')

            # Find the user with most hours played for the given genre. To do this...
            # Group df_aux by user, sum the hours played, sort by these in descending order, to take from the 1st row the max_playtime_user
            max_playtime_user = df_aux.groupby(['user_id'])['playtime_forever'].sum().reset_index().sort_values(by='playtime_forever', ascending=False).iloc[0]['user_id']

            # Filter by the user with most hours played for the genre received as argument
            df_aux = df_aux[df_aux['user_id'] == max_playtime_user]

            # Generate the list of hours played accumulated by year for the user and genre
            # Need to convert the statistic in 'playtime_forever' to hours, as it's in minutes. https://steamcommunity.com/discussions/forum/1/3814039097896647393/
            playtime_by_year = df_aux.groupby(['release_year'])['playtime_forever'].sum().reset_index().sort_values(by='release_year', ascending=False)
            playtime_by_year['playtime_forever_Hrs'] = round(playtime_by_year['playtime_forever'] / 60, 0)
            playtimeHrs_by_year_list = [{"Year": int(row['release_year']), "Hours": int(row['playtime_forever_Hrs'])} for index, row in playtime_by_year.iterrows()]
    
            return {f"User with most hours played for Genre {genero}": max_playtime_user, "Hours played": playtimeHrs_by_year_list}
        except Exception as e:
            logger.exception(f"Argument: genero = {genero}") #Joyuela: Log the exception
            raise e

# End-point 3
@app.get("/UsersRecommend/{annio}")
def UsersRecommend(annio: int):
        """
        Returns the top 3 most recommended games by users for a specific year.
        Only considers positive or neutral reviews where users explicitly recommended the game.
    
        Parameters:
        -----------
        annio : int
            The year to analyze (e.g., 2015)
    
        Returns:
        --------
        list
            A list of dictionaries containing the top 3 most recommended games
            Format: [
                {"Position 1": "Game Name 1"},
                {"Position 2": "Game Name 2"},
                {"Position 3": "Game Name 3"}
            ]
    
        Raises:
        -------
        Exception
            If the year doesn't exist in the dataset or if there's an error processing the data
        """
        try:
            # Returns the top 3 MOST recommended games by users for the given year. (reviews.recommend = True and positive/neutral comments)
            # Return example: [{"Position 1" : X}, {"Position 2" : Y},{"Position 3" : Z}]

            # Filter recommendations by the provided year
            df_aux = df1[df1['posted'].str.contains(str(annio))].copy()

            # Add 'good_review' column. Values True if 'recommend' is True and 'sentiment_analysis' is 1 or 2
            df_aux['good_review'] = np.where((df_aux['recommend'] == True) & (df_aux['sentiment_analysis'].isin([1, 2])), True, False)

        
            # Group by game (item_id) and sum the positive recommendations, then sort leaving the most recommended at the top
            # NOTE: if you sum a column with True/False, the result is similar to counting the True values, since each True is added as 1 and each False as 0
            df_aux = df_aux.groupby(['item_id'])['good_review'].sum().reset_index().sort_values(by='good_review', ascending=False)

            # Merge df_aux with df2 through the game id in both DataFrames. It's necessary to get the game name
            df_aux = df_aux.merge(df2, left_on='item_id', right_on='id')
        
            # Create the list with the top 3 most recommended games
            top3_most_recommended = [{"Position 1" : df_aux.iloc[0]['app_name']}, 
                                    {"Position 2" : df_aux.iloc[1]['app_name']}, 
                                    {"Position 3" : df_aux.iloc[2]['app_name']}]
    
            return top3_most_recommended
        except Exception as e:
            logger.exception(f"Argument: annio = {annio}") #Joyuela: Log the exception
            raise e

# End-point 4
@app.get("/UsersWorstDeveloper/{annio}")
def UsersWorstDeveloper(annio: int):
        """
        Returns the top 3 developers with the least recommended games by users for a specific year.
        Only considers negative reviews where users explicitly did not recommend the game.
    
        Parameters:
        -----------
        annio : int
            The year to analyze (e.g., 2015)
    
        Returns:
        --------
        list
            A list of dictionaries containing the top 3 developers with least recommended games
            Format: [
                {"Position 1": "Developer Name 1"},
                {"Position 2": "Developer Name 2"},
                {"Position 3": "Developer Name 3"}
            ]
    
        Raises:
        -------
        Exception
            If the year doesn't exist in the dataset or if there's an error processing the data
        """
        try:
            # Returns the top 3 developers with LEAST recommended games by users for the given year. (reviews.recommend = False and negative comments)
            # Return example: [{"Position 1" : X}, {"Position 2" : Y},{"Position 3" : Z}]

            # Filter recommendations by the provided year
            df_aux = df1[df1['posted'].str.contains(str(annio))].copy()
        
            # Add 'bad_review' column. Values True if 'recommend' is False and 'sentiment_analysis' is 0
            df_aux['bad_review'] = np.where((df_aux['recommend'] == False) & (df_aux['sentiment_analysis'] == 0), True, False)

            # Group by game (item_id) and sum the negative recommendations, then sort leaving the worst recommended at the top
            # NOTE: if you sum a column with True/False, the result is similar to counting the True values, since each True is added as 1 and each False as 0
            df_aux = df_aux.groupby(['item_id'])['bad_review'].sum().reset_index().sort_values(by='bad_review', ascending=False)

            # Merge df_aux with df2 through the game id in both DataFrames. It's necessary to get the developer name
            df_aux = df_aux.merge(df2, left_on='item_id', right_on='id')
        
            # Create the list with the top 3 developers with least recommended games
            top3_least_recommended_dev = [{"Position 1" : df_aux.iloc[0]['developer']}, 
                                        {"Position 2" : df_aux.iloc[1]['developer']}, 
                                        {"Position 3" : df_aux.iloc[2]['developer']}]
        
            return top3_least_recommended_dev
        except Exception as e:
            logger.exception(f"Argument: annio = {annio}") #Joyuela: Log the exception
            raise e

# End-point 5
@app.get("/sentiment_analysis/{empresa_desarrolladora}")
def sentiment_analysis(empresa_desarrolladora: str):
        """
        Returns sentiment analysis statistics for all games from a specific developer.
    
        Parameters:
        -----------
        empresa_desarrolladora : str
            The name of the developer company (e.g., "Valve", "Ubisoft")
    
        Returns:
        --------
        dict
            A dictionary containing the developer name as key and a list of sentiment counts as value
            Format: {
                "Developer Name": [
                    "Negative = X",
                    "Neutral = Y",
                    "Positive = Z"
                ]
            }
    
        Raises:
        -------
        ValueError
            If the developer company doesn't exist in the dataset
        Exception
            If there's an error processing the data
        """
        try:
            # According to the developer company, returns a dictionary with the developer name as key and a list
            # with the total number of user review records that are categorized with a sentiment analysis as value.
            # Return example: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

            # Merge df1 with df2 through the game id in both DataFrames.
            df_aux = df1.merge(df2, left_on='item_id', right_on='id')

            # Filter records in df_aux where the 'developer' column matches the 'empresa_desarrolladora' passed as argument
            df_aux = df_aux[df_aux['developer'] == empresa_desarrolladora]

            # Check if the developer company exists in the dataset
            if df_aux.empty:
                raise ValueError(f"The developer company '{empresa_desarrolladora}' is not found in the dataset")

            # Count the unique values in the 'sentiment_analysis' column and store the counts in separate variables
            Negative = "Negative = " + str(df_aux['sentiment_analysis'].value_counts().get(0, 0))
            Neutral = "Neutral = " + str(df_aux['sentiment_analysis'].value_counts().get(1, 0))
            Positive = "Positive = " + str(df_aux['sentiment_analysis'].value_counts().get(2, 0))
        
            return {empresa_desarrolladora: [Negative, Neutral, Positive]} 
        except Exception as e:
            logger.exception(f"Argument: empresa_desarrolladora = {empresa_desarrolladora}") #Joyuela: Log the exception
            raise e
    
# End-point 6
@app.get("/recomendacion_juego/{id}")
def recomendacion_juego(id: str):
    """
    Returns a list of 5 game recommendations based on similarity to the provided game ID.
    Uses cosine similarity on game features to determine the most similar games.
    
    Parameters:
    -----------
    id : str
        The ID of the game to use as a basis for recommendations
    
    Returns:
    --------
    list
        A list of dictionaries containing the 5 recommended games
        Format: [
            {"game_id_1": "Game Name 1"},
            {"game_id_2": "Game Name 2"},
            ...
        ]
    
    Raises:
    -------
    Exception
        If the game ID doesn't exist or if there's an error processing the data
    """
    try:
        # Get the index of the provided game id
        idx = df2[df2['id'] == id].index[0]
        
        # Get the similarity scores of the game in question with other games
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the games according to similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices of the most similar games (excluding the game itself)
        similar_games_indices = [i[0] for i in sim_scores[1:6]]  # Recommend the 5 most similar games
        
        # Return the names of the recommended games
        #return df['app_name'].iloc[similar_games_indices]

        # Create a list of dictionaries with 'id' as key and 'app_name' as value
        recommended_games = [{df2['id'].iloc[ind] : df2['app_name'].iloc[ind]} for ind in similar_games_indices]
    
        return recommended_games
    except Exception as e:
        logger.exception(f"Argument: id = {id}") #Joyuela: Log the exception
        raise e

if __name__ == "__main__":
    if "--trace" in sys.argv:  #Only execute trace() when script is run with "trace" argument (e.g., python main.py trace)
        trace()  # Call the trace function to log memory usage and DataFrame contents
    uvicorn.run("main:app", reload=True) #Run the FastAPI application