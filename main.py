# General
import pandas as pd
import numpy as np
import json
import sklearn
from xgboost.sklearn import XGBRegressor
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style

# Modeling
import xgboost as xgb
import lightgbm as lgb
import torch
from IPython.display import display
# Geolocation
from geopy.geocoders import Nominatim
import joblib

from processing import FeatureProcessorClass, create_revealed_targets_train

# Options
pd.set_option('display.max_columns', 100)

DEBUG = False  # False/True

# GPU or CPU use for model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# Helper functions
def display_df(df, name):
    '''Display df shape and first row '''
    PrintColor(text=f'{name} data has {df.shape[0]} rows and {df.shape[1]} columns. \n ===> First row:')
    display(df.head(1))


# Color printing
def PrintColor(text: str, color=Fore.BLUE, style=Style.BRIGHT):
    '''Prints color outputs using colorama of a text string'''
    print(style + color + text + Style.RESET_ALL)


DATA_DIR = "./input/"

# Read CSVs and parse relevant date columns
train = pd.read_csv(DATA_DIR + "train.csv")
client = pd.read_csv(DATA_DIR + "client.csv")
historical_weather = pd.read_csv(DATA_DIR + "historical_weather.csv")
forecast_weather = pd.read_csv(DATA_DIR + "forecast_weather.csv")
electricity = pd.read_csv(DATA_DIR + "electricity_prices.csv")
gas = pd.read_csv(DATA_DIR + "gas_prices.csv")

location = (pd.read_csv("./input/county_lon_lats.csv")
            .drop(columns=["Unnamed: 0"])
            )

display_df(train, 'train')
display_df(client, 'client')
display_df(historical_weather, 'historical weather')
display_df(forecast_weather, 'forecast weather')
display_df(electricity, 'electricity prices')
display_df(gas, 'gas prices')
display_df(location, 'location data')

# See county codes
with open(DATA_DIR + 'county_id_to_name_map.json') as f:
    county_codes = json.load(f)
pd.DataFrame(county_codes, index=[0])

# pd.DataFrame(train[train['is_consumption'] == 0].target.describe(
#    percentiles=[0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])).round(2).T()
# pd.DataFrame(train[train['is_consumption'] == 1].target.describe(
#    percentiles=[0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])).round(2).T()


# Create all features
N_day_lags = 15  # Specify how many days we want to go back (at least 2)

FeatureProcessor = FeatureProcessorClass()

data = FeatureProcessor(data=train.copy(),
                        client=client.copy(),
                        historical_weather=historical_weather.copy(),
                        forecast_weather=forecast_weather.copy(),
                        electricity=electricity.copy(),
                        gas=gas.copy(),
                        )

df = create_revealed_targets_train(data.copy(),
                                   N_day_lags=N_day_lags)
#print(df)

#### Create single fold split ######
# Remove empty target row
target = 'target'
df = df[df[target].notnull()].reset_index(drop=True)

train_block_id = list(range(0, 600))

tr = df[df['data_block_id'].isin(train_block_id)] # first 600 data_block_ids used for training
val = df[~df['data_block_id'].isin(train_block_id)] # rest data_block_ids used for validation

# Remove columns for features
no_features = ['date',
                'latitude',
                'longitude',
                'data_block_id',
                'row_id',
                'hours_ahead',
                'hour_h',
               ]

remove_columns = [col for col in df.columns for no_feature in no_features if no_feature in col]
remove_columns.append(target)
features = [col for col in df.columns if col not in remove_columns]
PrintColor(f'There are {len(features)} features: {features}')
PrintColor(f'There are {len(remove_columns)} removed columns: {remove_columns}')
clf = xgb.XGBRegressor(
                        device = device,
                        enable_categorical=True,
                        objective = 'reg:absoluteerror',
                        n_estimators = 2 if DEBUG else 1500,
                        early_stopping_rounds=100
                       )

clf.fit(X = tr[features],
        y = tr[target],
        eval_set = [(tr[features], tr[target]), (val[features], val[target])],
        xgb_model = 'model.pkl',
        verbose=True #False #True
       )
joblib.dump(clf, 'model.pkl')
PrintColor(f'Early stopping on best iteration #{clf.best_iteration} with MAE error on validation set of {clf.best_score:.2f}')
results = clf.evals_result()
train_mae, val_mae = results["validation_0"]["mae"], results["validation_1"]["mae"]
x_values = range(0, len(train_mae))
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x_values, train_mae, label="Train MAE")
ax.plot(x_values, val_mae, label="Validation MAE")
ax.legend()
plt.ylabel("MAE Loss")
plt.title("XGBoost MAE Loss")
plt.show()

TOP = 20
importance_data = pd.DataFrame({'name': clf.feature_names_in_, 'importance': clf.feature_importances_})
importance_data = importance_data.sort_values(by='importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=importance_data[:TOP],
            x='importance',
            y='name'
            )
patches = ax.patches
count = 0
for patch in patches:
    height = patch.get_height()
    width = patch.get_width()
    perc = 100 * importance_data['importance'].iloc[count]  # 100*width/len(importance_data)
    ax.text(width, patch.get_y() + height / 2, f'{perc:.1f}%')
    count += 1

plt.title(f'The top {TOP} features sorted by importance')
plt.show()
print(importance_data[importance_data['importance']<0.0005].name.values)