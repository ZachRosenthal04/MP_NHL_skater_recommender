import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

# Load the datasets
pd.set_option('display.max_columns', None) # Display Preference

def calculate_playing_age(df, dob_col_name, age_col_name, season_year):
    """
    Updates the age of players in the DataFrame based on their date of birth.

    Parameters:
    df (pd.DataFrame): The DataFrame containing player data.
    dob_col_name (str): The name of the column with date of birth information.
    age_col_name (str): The name of the column where the age should be updated.
    current_year (int): The year to calculate current age from.

    Returns:
    pd.DataFrame: The DataFrame with updated ages.
    """
    # Extract the year and convert it to an integer
    df['Birth Year'] = df[dob_col_name].str[:4].astype(int)
    
    # Calculate the new age and replace the 'Age' column
    df[age_col_name] = season_year - df['Birth Year']
    
    # Drop the helper column
    df.drop(columns='Birth Year', inplace=True)
    
    return df



def MP_calculate_playing_age(df, dob_col_name, season_col_name, age_col_name):
    """
    Updates the age of players in the DataFrame based on their date of birth.

    Parameters:
    df (pd.DataFrame): The DataFrame containing player data.
    dob_col_name (str): The name of the column with date of birth information.
    age_col_name (str): The name of the column where the age should be updated.
    current_year (int): The year to calculate current age from.

    Returns:
    pd.DataFrame: The DataFrame with updated ages.
    """
    # Convert the 'birthDate' column to datetime format
    df[dob_col_name] = pd.to_datetime(df[dob_col_name], errors='coerce')  # Handle potential errors during conversion

    # Extract the year
    df['birth_year'] = df[dob_col_name].dt.year

    # Calculate the new age and replace the 'Age' column
    df[age_col_name] = df[season_col_name] - df['birth_year']

    # Drop the helper column
    df.drop(columns='birth_year', inplace=True)

    return df

def index_sorter(df):
    df.sort_index()
    return df  # Return the modified dataframe


def MP_create_player_index_dict(df):
      """
    Create a nested dictionary from a DataFrame that maps player names to their indices for each season.

    This function resets the index of the DataFrame to ensure that the index column 
    holds the original row indices. It then groups the DataFrame by 'name' and 'season' 
    and aggregates the indices into a list for each group. After grouping, it pivots the DataFrame 
    so each players' 'name' is a row with each 'season' as columns, containing lists of indices 
    as values. Finally, it converts the pivoted DataFrame into a nested dictionary where each player's 
    name is a key to a dictionary mapping each season to the player's indices.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process, which must contain 'Player' and 'Season' columns 
                           and has a unique index.

    Returns:
    dict: A nested dictionary where the first level keys are player names, and second level keys are 
          seasons, each mapping to a list of index positions for that player in that season.
    """

    # Reset the index 
      df = df.reset_index()

    # Group by 'Player' and 'Season', then aggregate the original index values into a list.
      grouped = df.groupby(['name', 'season'])['index'].agg(lambda x: list(x)).reset_index()

    # Pivot the DataFrame to have 'Player' as rows and 'Season' as columns with list of indices as values.
      pivot_df = grouped.pivot(index='name', columns='season', values='index')

    # Convert the pivoted DataFrame into a nested dictionary.
      MP_player_index_dict = pivot_df.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()

      return MP_player_index_dict

def MP_get_index_all_gamestates(player_name, MP_AS_dict= MP_AS_player_dict, MP_5on5_dict= MP_5on5_player_dict, 
                                MP_4on5_dict= MP_4on5_player_dict, MP_5on4_dict= MP_5on4_player_dict,
                                MP_OS_dict= MP_OS_player_dict):
    """
    Returns a string with all the indices for each game state (All Strengths, Even Strength,
    Power Play, and Penalty Kill) for a given player.
    
    Parameters:
    - player_name (str): The name of the player to lookup.
    - player_index_dict_AS (dict): The dictionary with indices for All Strengths.
    - player_index_dict_ES (dict): The dictionary with indices for Even Strength.
    - player_index_dict_PP (dict): The dictionary with indices for Power Play.
    - player_index_dict_PK (dict): The dictionary with indices for Penalty Kill.

    Returns:
    - str: A formatted string containing the indices for each game state for the player.
    """
    result_string= (
        f"{player_name}'s ALL SITUATIONS indices are: {MP_AS_dict.get(player_name)}\n"
        f"{player_name}'s 5-ON-5 indices are: {MP_5on5_dict.get(player_name)}\n"
        f"{player_name}'s 4-ON-5 indices are: {MP_4on5_dict.get(player_name)}\n"
        f"{player_name}'s 5-ON-4 indices are: {MP_5on4_dict.get(player_name)}\n"
        f"{player_name}'s OTHER SITUATIONS indices are: {MP_OS_dict.get(player_name)}\n"
    )

    return print(result_string)

def MP_get_players_baseline_gamestate_stats(original_gamestate_df, player_name):
    """
    Returns the baseline performance metrics of the player you are finding comparable players of 
    so you can see how their stats are over the course of the seasons in the engine.
    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
    - player_name: must be a string of the full name of the player you want to look up, 
    If player name is misspelled or there is no data for that player, 
    the function returns an empty dataframe 

    """
    baseline_gamestate_stats = original_gamestate_df.loc[original_gamestate_df['name'] == player_name]
    return baseline_gamestate_stats

# This is the processing pipeline:
# Columns not to included in the processing:
col_not_processed = ['playerId', 'season' , 'name', 'team', 'situation', 'iceTimeRank', 'I_F_shifts',
                      'nationality' ,'birthDate', 'weight','height', 'shoots', 'age']


# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
        ('age_group', Pipeline([
            ('ordinal', OrdinalEncoder(categories=[['New Pro', 'Young Pro', 'Prime Age', 'Vet', 'Old Vet']])),
            ('scaler', StandardScaler())  # Scale the ordinal-encoded age_group
        ]), ['age_group']),
        ('position', Pipeline([
            ('onehot', OneHotEncoder()),  # Apply OneHotEncoder to 'position'
            ('scaler', StandardScaler(with_mean=False))  # Apply StandardScaler after OneHotEncoder
        ]), ['position'])
    ])

# My current Pipeline
MP_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA())
])

# Columns not to included in the processing:
col_not_processed = ['playerId', 'season' , 'name', 'team', 'situation', 'iceTimeRank', 'I_F_shifts',
                      'nationality' ,'birthDate', 'weight','height', 'shoots', 'age', 'gameScore']

def MP_recommend_skaters(original_gamestate_df, processed_gamestate_df, season, player_index, top_n=6):
    """
    Recommends skaters based on their stats using a preprocessed PCA features.

    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
        Acceptable inputs for original_gamestate_df are: [MP_AS_stats, MP_5on5_stats, MP_4on5_stats, MP_5on4_stats, MP_OS_stats]
    - processed_gamestate_df (pd.DataFrame): PCA-transformed and scaled features of the skaters.
        Acceptable inputs for processed_gamestate_df are: 
        [MP_AS_processed_data, MP_5on5_processed_data, MP_4on5_processed_data, MP_5on4_processed_data, MP_OS_processed_data]
    - season (int): The target season for comparison.
        Acceptable inputs for season are: 2021, 2022, 2023 
    - player_index (int): Index of the player in the DataFrame to get recommendations for.
        player_index as accessed through the function: MP_get_index_all_gamestates() 
    - top_n (int): Number of top recommendations to return.

    Returns:
    - pd.DataFrame: DataFrame containing the top_n recommended skaters for the given player in the specified season.
    """

    # Filter DataFrame for the target season
    target_season_data = processed_gamestate_df[original_gamestate_df['season'] == season]

    # Compute pairwise distances between all skaters and those from the target season
    distances = pairwise_distances(processed_gamestate_df, target_season_data)

    # Find the indices of the closest skaters
    indices = np.argsort(distances, axis=1)[:, :top_n]

    # Retrieve the recommendations from the original stats DataFrame
    MP_recommended_skaters = original_gamestate_df[original_gamestate_df['season'] == season].iloc[indices[player_index], :]

    return MP_recommended_skaters


# Get feature names after the transformation
def get_feature_names(column_transformer):
    output_features = []
    for name, transformer, features in column_transformer.transformers_:
        if transformer == 'drop' or transformer is None:
            continue
        if isinstance(transformer, Pipeline):
            transformer = transformer.named_steps['onehot'] if 'onehot' in transformer.named_steps else transformer
        try:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names = transformer.get_feature_names_out(features)
                output_features.extend(feature_names)
            else:
                output_features.extend(features)
        except NotFittedError:
            output_features.extend(features)
    return output_features

def calculate_ZR_gameScore(df):
    """
    Calculates the ZR_gameScore for a given DataFrame.

    Args:
        df: The DataFrame containing player statistics.

    Returns:
        The DataFrame with the 'ZR_gameScore' column added.
    """

    df['ZR_gameScore'] = (
        (df['I_F_goals'] * 0.75) 
        + (df['I_F_primaryAssists'] * 0.7) 
        + (df['I_F_secondaryAssists'] * 0.55)
        + (df['I_F_shotsOnGoal'] * 0.075) 
        + (df['shotsBlockedByPlayer'] * 0.05) 
        + (df['penaltiesDrawn'] * 0.15) 
        - (df['penalties'] * 0.15)
        + (df['I_F_hits'] * 0.01) 
        - (df['I_F_dZoneGiveaways'] * 0.03) 
        + (df['I_F_takeaways'] * 0.015) 
        - (df['I_F_giveaways'] * 0.015)
        + (df['onIce_corsiPercentage']) 
        + (df['faceoffsWon'] * 0.01) 
        - (df['faceoffsLost'] * 0.01)
        + (df['OnIce_F_goals'] * 0.15) 
        - (df['OnIce_A_goals'] * 0.15)
    )

    return df

from sklearn.preprocessing import MinMaxScaler

# Group by 'season' and apply MinMaxScaler within each group
def scale_by_season(group):
    if len(group) == 1:  # Handle the case of only one value in a season
        return 100  # Assign the maximum rating if there's only one player
    else:
        scaler = MinMaxScaler(feature_range=(0, 100))
        return scaler.fit_transform(group.values.reshape(-1, 1)).ravel()