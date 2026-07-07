

# International Football Scores Prediction - Gammafest Data Science Competition 2026
This competition aims to predict multiple target variables, which are team_goals and opp_goals based on elo, rank, goals previous match, geographical data, and more. Data set includes football history match from 1872 to 2026'

---
## Datasets
Datasets is obtained from this kaggle competition https://www.kaggle.com/t/73a673b3dee6c9358e37e183bb826b75, which consist of train.csv, test.csv, and sample submission.csv.

---
## Folder Structure

```
Gammafest-2026/
├── Notebooks/
   ├── early_preprocessing.ipynb            ← Cleaning data mostly for hanldling missing values
   ├── feature_engineering.ipynb            ← Feature Engineering section
   ├── fill_submission_gb.ipynb             ← Predict and fill the submission with Gradient Boosting
   ├── fill_submission_lgbm.ipynb           ← Predict and fill the submission with LightGBM
   ├── final_modeling.ipynb                  ← Training selected model
   ├── modeling_for_preprocessing.ipynb      ← Evaluate baseline model
```

---
## Notebook Works Order

```
early_preprocessing.ipynb
            ↓
feature_engineering.ipynb
            ↓
modeling_for_preprocessing.ipynb
            ↓
final_modeling.ipynb
            ↓
fill_submission_gb.ipynb
            ↓
fill_submission_lgbm.ipynb
```

---
## Pipeline Preprocessing

### Missing Values

![missing values](https://github.com/n44th44nn/Gammafest-2026/blob/9dc8d765e626b54020ed2e08d670d831626a19bf/Visualizations/early_preprocessing_notebook/missing%20value%20by%20year.png)

As we can see in visualization above, we have a lots of missing values from rank_team, rank_opponent, rank_diff, geographical features, and wheater in test.csv and some time-searies based features.

these are the methods for handle missing values:

- We fill days_since_last_match* columns with 0, cause we consider them as first match
- For *_last\* columns, we fill 0 if first match and fill the values according the match they had been played
- For geographical, wheather, and sosio-economic columns, we fill them with median, based on match venue, etc.
- for rank column, we keep them null but create a new column to inform about missing value

### Feature Engineering

- **Tournament Tier** : Created tournament tier based on tournament column  
```python
tier1 = [
    'FIFA World Cup', 'UEFA Euro', 'Copa América',
    'African Cup of Nations', 'AFC Asian Cup',
    'Gold Cup', 'OFC Nations Cup', 'CONCACAF Championship', 'Olympic Games'
]
tier2 = [
    'FIFA World Cup qualification', 'UEFA Euro qualification',
    'African Cup of Nations qualification', 'AFC Asian Cup qualification',
    'Copa América qualification', 'Gold Cup qualification',
    'CONCACAF Nations League', 'UEFA Nations League', 'AFF Championship'
]
tier3 = [
    "King's Cup", 'Algarve Cup', 'Merdeka Tournament', 'Gulf Cup',
    'CECAFA Cup', 'Nordic Championship', 'Island Games',
    'CFU Caribbean Cup', 'CFU Caribbean Cup qualification',
    'British Home Championship', 'Southeast Asian Games',
    'South Pacific Games', 'Asian Games', 'Muratti Vase',
    'Amilcar Cabral Cup', 'AFC Championship'
]
```
- **Temporal Features** : Extracted month, year, etc based on date column
- **ELO Features** : Create ELO difference and ELO ration between ELO team and ELO opponent
- **Form and Momentum Features**
```python
df['attack_vs_def_team']      = df['team_avg_goals_last5'] - df['opp_avg_conceded_last5']
df['attack_vs_def_opp']       = df['opp_avg_goals_last5']  - df['team_avg_conceded_last5']
df['xg_advantage']            = df['attack_vs_def_team'] - df['attack_vs_def_opp']
df['momentum_diff']           = df['team_win_rate_last10'] - df['opp_win_rate_last10']
df['momentum_ratio']          = df['team_win_rate_last10'] / (df['opp_win_rate_last10'] + 1e-5)
df['pts_diff_last10']         = df['team_points_last10'] - df['opp_points_last10']
df['team_attack_consistency'] = df['team_avg_goals_last5'] / (df['team_avg_conceded_last5'] + 1e-5)
df['opp_attack_consistency']  = df['opp_avg_goals_last5']  / (df['opp_avg_conceded_last5']  + 1e-5)
df['attack_consistency_diff'] = df['team_attack_consistency'] - df['opp_attack_consistency']
```
- **H2H Features** : Create head-to-head dominance, goal difference per match, and missing
- **Home Advantage Features**
- **Environmental Features**
```python
df['travel_fatigue_diff']   = df['distance_travel_team'] - df['distance_travel_opp']
df['total_travel']          = df['distance_travel_team'] + df['distance_travel_opp']
df['altitude_away_penalty'] = df['altitude_venue'] * (1 - df['is_home'])
df['gdp_diff']              = df['gdp_per_capita_team'] - df['gdp_per_capita_opp']
df['gdp_ratio']             = df['gdp_per_capita_team'] / (df['gdp_per_capita_opp'] + 1e-5)
df['population_diff']       = df['population_team'] - df['population_opp']
df['population_ratio']      = np.log1p(df['population_team']) - np.log1p(df['population_opp'])
df['temp_extreme']          = (df['temperature_venue'] - 20).abs()
```
- **Rest/Fatigue Features**
```python
df['rest_diff'] = df['days_since_last_match_team'] - df['days_since_last_match_opp']

def rest_cat(d):
    if pd.isna(d): return 1
    if d <= 4:  return 0
    if d <= 10: return 1
    return 2

df['rest_cat_team']  = df['days_since_last_match_team'].apply(rest_cat)
df['rest_cat_opp']   = df['days_since_last_match_opp'].apply(rest_cat)
df['rest_advantage'] = df['rest_cat_team'] - df['rest_cat_opp']
```

### Encoding
- **Binary Encoding** : gender
- **One-Hot Encoding** : confederation
- **Target Encoding with Median** : venue country, tournament
## Modeling

### Baseline Model

Since the objective is predict multiple-target, we use Regressor Chain from Scikit-Learn to train these baseline model to decide top 2 best model with baseline hyperparameter:

- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Adaptive Boosting

Here are the training results:

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| LightGBM | 1.520174 | 1.029386 | 0.295202 |
| Gradient Boosting | 1.539480 | 1.037510 | 0.277162 |
| Random Forest | 1.543043 | 1.054582 | 0.273813 |
| XGBoost | 1.550429 | 1.045792 | 0.266855 |
| Ada Boost | 2.763086 | 2.409590 | -1.344012 |

According that results, we bring LightGBM and Gradient Boosting to Hyperparameter Tuning Section.

### Hyperparameter Tuning

After using Grid Search Cross-Validate, here are the results:

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Gradient Boosting | 1.505475 | 1.017073 | 0.308752 |
| LightGBM | 1.507733 | 1.021674 | 0.306691 |

We got Gradient Boosting is the best model for this data, with Root Mean Squared Error approximately 1.5 goals.
