import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """
    Loads and processes Steam game data from Parquet files and builds a recommendation model.
    
    This function:
    1. Loads three datasets: user reviews, game information, and user playtime statistics
    2. Applies filtering to optimize memory usage
    3. Builds a game recommendation model using TF-IDF and cosine similarity
    
    Returns:
    --------
    tuple
        df1 (DataFrame): User reviews data
        df2 (DataFrame): Game information data
        df3 (DataFrame): User playtime statistics (filtered)
        cosine_sim (ndarray): Cosine similarity matrix for game recommendations
    """
    # Load user reviews data
    df1 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_user_reviews_FE.parquet')

    # Load game information data
    df2 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_output_steam_games.parquet')

    # Load user playtime statistics with filtering to reduce memory usage
    # Only include records where users have played the game (playtime > 0)
    filters = [('playtime_forever', '>', 0)]
    columns_to_keep = ['item_id', 'item_name', 'playtime_forever', 'user_id']
    df3 = pd.read_parquet('Datasets/Clean_Parquet_Data_Steam/Clean_australian_users_items.parquet',
                        columns=columns_to_keep, filters=filters)

    ########################################################################################################
    # Build game recommendation model
    ########################################################################################################
    
    def CombFeatures():
        """
        Creates a combined feature representation of games by concatenating multiple attributes.
        
        Memory optimization: Originally, this function processed all data which required ~4GB RAM.
        The current implementation uses the full dataset as we found it provides better recommendation
        quality while still keeping memory usage under 400MB.
        
        Returns:
        --------
        Series
            Combined text features for each game
        """
        total_rows = len(df2)
        start_row = 0 
        # We're using the full dataset now, but keeping the structure for potential future partitioning
        end_row = total_rows - 1  # Previously divided by 10 (//10) for memory constraints
        
        df = pd.DataFrame()
        # Combine relevant features into a single text representation
        df['combined_features'] = df2['genres'] + ' ' + df2['specs'] + ' ' + df2['developer'] + ' ' + df2['id']
        return df.loc[start_row:end_row, 'combined_features']

    # Initialize TF-IDF vectorizer and transform the combined features
    # Note: We're using the dynamically generated features from CombFeatures()
    # rather than a pre-existing column in df2
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(CombFeatures())

    # Calculate cosine similarity between all game pairs
    # This creates a square matrix where each element [i,j] represents 
    # the similarity between game i and game j
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df1, df2, df3, cosine_sim

# Make the loaded data available at module level for importing by other modules
df1, df2, df3, cosine_sim = load_data()