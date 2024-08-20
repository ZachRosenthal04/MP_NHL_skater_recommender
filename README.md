# MP_NHL_skater_recommender
This is an improved NHL skater recommender engine that uses data from MoneyPuck.com for the 2021-22, 2022-23, and 2023-24 regular seasons.

# NHL_Skater_Recommender_System
## Project Title: Finding Comparable NHL Skaters
This is a content-based recommender system. It takes NHL player data from individual performance totals and rates as well as on-ice performance totals and rates from 3 NHL regular seasons: 2021-22, 2022-23, 2023-24 that has been split into 5 different game states: all situations (AS), 5on5, 4on5, 5on4, and other situations (OS) and based on the features in the selected game state outputs the "n" most similar players. The recommender uses a player’s index as its reference point and uses the pairwise distances of indices when recommending.

The unique features of this recommender are currently twofold. First, since it is divided into different game states, this recommender offers a more nuanced perspective of a player’s performance than those traditionally considered in contract arbitration such as regular season and playoffs. Secondly, it can perform comparisons beyond the season of the original player index. For example, this recommender engine, using a player’s index from the 2021-2022 season can find similar indices of players in the 2023-24 season and so on. Something to note is that there are no records for goalies in this engine. A goalie-specific recommender system is something I hope to build in the future.


## Project Motivation:
My motivation for this project began with the simple fact that I'm a big hockey fan and being from Montreal, hockey is always big news in the city.
Hockey isn't just big news, its big money. The NHL is currently valued at USD $42 billion! Not only this but the NHL has one of the strictest salary caps of the major North American professional sports leagues and it is currently valued at USD $83.5 million per team. Teams spend millions every year in player development, scouting, and contract negotiation to mitigate the salary cap and I believe this recommender can reduce the time and resources spent on these tasks, enabling a better and more effective use of teams' budgets and resources. This engine not only offers a tool that can recommend skaters teams may be interested in acquiring but also is a great tool for teams to better understand the value of the players they already have.

Beyond just being a hockey fan, I have always been into sports and have always been really into the amazing stats that are shown on the screen during games that offer such interesting nuances into the game within the game. I’ve always enjoyed the strategy behind building a roster, scouting, and trades. I aimed to build something that could track player development using comparable players in the league. 
I wanted to build something that was constantly improvable and scalable. This project is endlessly scalable because it can continue to include upcoming seasons as well as past seasons and with time can be adapted to other divisions or sports. 

## What problem or problems does this engine solve?
There are two problems I’m trying to solve with this recommender engine. The first problem is player performance tracking. For young players, it is extremely difficult to know how a player’s development is progressing because of how difficult it is to compare and analyze metrics beyond simple scoring metrics while also taking into account positional metrics’ nuances and different game states. This recommender offers a means of tracking player progression, decline, or consistency based on references of similar players. This leads me to the second problem I aim to solve – contract arbitration and navigating the NHL’s salary cap.
Salary arbitration is when a third party is needed to determine a fair salary for a player when a player (and their agent) cannot reach an agreement with the team’s offer. Most players opt for arbitration but few end up progressing to an arbitration hearing. That being said, the process is often financially and mentally taxing on the player and team and can sour or destroy the relationship between the players and their team. A key factor in contract negotiations and arbitration hearings is player comparisons which are used in salary benchmarking. This is similar to how house prices are determined by comparisons. This recommender offers a nuanced, data-driven comparison, available to both players and teams which can help teams manage the cap, plan for the future, or avoid lengthy arbitration hearings that can ruin important inter- and intra-team relationships.
## Installation:
The notebooks and recommender us the following libraries and packages:
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

# Load the datasets
pd.set_option('display.max_columns', None) # Display Preference
```
### EDA:
The file “MP_EDA_and_recommender_functions.py” is a Python script that contains all the functions used to clean the season data, combine the seasons into their respective game-state data frames and to build and access the players’ indices as well as the function to run the recommender engine itself.
## How to use the recommender:
The recommender uses the pairwise distances of a player’s index with reference to the game state of a given season and returns the data for the 5 most similar players. The actual recommender is in the Jupyter Notebook called “MoneyPuckSKaterRecommender.ipynb” which was also used to combine the game-states of each season into the data frames used in the recommender engine.

### Step 1: Get the player’s index for the desired game state and season.
To do this, run the function: MP_get_index_all_gamestates(‘player’s full name you want recommended’) which looks like this: Remember to adjust the parameters for the desired game state.
```python
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
```
The use will look like this:
```python
MP_get_index_all_gamestates(player_name='Nick Suzuki')
```
and the result will look like:
```
Nick Suzuki's ALL SITUATIONS indices are: {2022: [827], 2023: [1909], 2024: [2496]}
Nick Suzuki's 5-ON-5 indices are: {2022: [827], 2023: [1909], 2024: [2496]}
Nick Suzuki's 4-ON-5 indices are: {2022: [827], 2023: [1909], 2024: [2496]}
Nick Suzuki's 5-ON-4 indices are: {2022: [827], 2023: [1909], 2024: [2496]}
Nick Suzuki's OTHER SITUATIONS indices are: {2022: [827], 2023: [1909], 2024: [2496]}
```


### Step 2 - Getting player's baseline stats *Optional*
```python
def MP_get_players_baseline_gamestate_stats(original_gamestate_df, player_name):
    """
    Returns the baseline performance metrics of the player you are finding comparable players of 
    so you can see how their stats are over the course of the seasons in the engine.
    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
    - player_name: must be a string of the full name of the player you want to look up, 
    If player name is misspelled or there is no data for that player, 
    the function returns an empty dataframe.
    -Small adustment from the other function. The MP function uses 'name' instead of 'Player' 

    """
    baseline_gamestate_stats = original_gamestate_df.loc[original_gamestate_df['name'] == player_name]
    return baseline_gamestate_stats
```
The use will look like:
```python
MP_get_players_baseline_gamestate_stats(MP_AS_stats, 'Nick Suzuki')
```
The output will be a dataframe of the players stats for the seasons available in the given gamestate

### Step 3- MoneyPuck: Finding Similar NHL Players

The function is:
```python
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
```
Using the function for each gamestate should look like this:
```python
# AS
nick_suzuki_AS_sim_skaters = MP_recommend_skaters(original_gamestate_df=MP_AS_stats,
                                                  processed_gamestate_df=MP_AS_stats_transformed,
                                                  season=2024,
                                                  player_index=2496,
                                                  top_n=6)

#5on5
nick_suzuki_5on5_sim_skaters = MP_recommend_skaters(original_gamestate_df=MP_5on5_stats,
                                                  processed_gamestate_df=MP_5on5_stats_transformed,
                                                  season=2024,
                                                  player_index=2496,
                                                  top_n=6)

#4on5
nick_suzuki_4on5_sim_skaters = MP_recommend_skaters(original_gamestate_df=MP_4on5_stats,
                                                  processed_gamestate_df=MP_4on5_stats_transformed,
                                                  season=2024,
                                                  player_index=2496,
                                                  top_n=6)

#5on4
nick_suzuki_5on4_sim_skaters = MP_recommend_skaters(original_gamestate_df=MP_5on4_stats,
                                                  processed_gamestate_df=MP_5on4_stats_transformed,
                                                  season=2024,
                                                  player_index=2496,
                                                  top_n=6)

#OS
nick_suzuki_OS_sim_skaters = MP_recommend_skaters(original_gamestate_df=MP_OS_stats,
                                                  processed_gamestate_df=MP_OS_stats_transformed,
                                                  season=2024,
                                                  player_index=2496,
                                                  top_n=6)
```

Without additional formating, the output will be a dataframe of the top_n most comparable skaters. The reason as a baseline I have used 6 is that since it uses the distances of the indeces, the most similar skater will always be the player himself and so if you want to find the top 5 you need to use the number of comparisons you desire + 1. 

## Contributing:
I encourage others to build on this project, as I will, by continuing to do feature engineering and using additional game state data as well as other seasons. Additionally, this project doesn’t take into account goaltender data and so this is one very clear space for additional contributions.
## Credits:
Though not technically involved in the direct creating of the project, I do want to shout out to Natural Stattrick.com which was my source for all the data used in this recommender.
## Future Considerations
### Short and medium term: 
I’m looking to do additional feature engineering including some feature engineering tailored to the specifics of each game-state and position. I would also like to build a complimentary recommender system using goalie data. Additionally, I’m hoping to have more features that indicate the players’ importance to their team to compliment and hopefully improve the quality of the recommendations.
### Long term: 
I’m hoping to be able to incorporate contract data and ML and DL algorithms to enable the recommender engine to be able to recommend similar players that also take into account contract and projected contract restrictions due to salary cap implications.  
## Contact Details:
Email: zach.rosenthal04@gmail.com
LinkedIn: zachary-rosenthal04/
GitHub: ZachRosenthal04

