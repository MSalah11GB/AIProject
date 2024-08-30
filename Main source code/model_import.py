import pandas as pd
import numpy as np
import matplotlib as pyplot
import pickle

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import xgboost as xgb

# Loads the data
data = pd.read_csv("cleaned_data.csv")
data = data.drop('price_per_sqft', axis = 'columns')

data = data.dropna(how = 'any')

X = data.drop(['price'],axis='columns')
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 69)

location_list = X.columns[4:].tolist()

print(location_list)

# Sidebar

def user_input_features(): 
    noOfRooms = 3
    totalSquareFoot = 1000/1.0

    # extract location column
   
    location = '1st Phase JP Nagar'
    bath = 4
    balcony = 5
    data = {
        'No. of Rooms': noOfRooms,
        'Area (ft^2)' : totalSquareFoot,
        'Location' : location,
        'No. of Bathrooms': bath,
        'No. of Balcony': balcony
    }
    features = pd.DataFrame(data, index = [0])
    return features

default_input = user_input_features()

print(default_input)

# Main panel

# Prepare data
def encode_location(input_location, all_locations):
    encoded_location = [False] * len(all_locations)
    try:
        index = all_locations.index(input_location)
        encoded_location[index] = True
    except ValueError:
        print(f"Location '{input_location}' not found in the list of all locations.")
    return encoded_location

def prepare_input_for_prediction(user_input, location_list):
    encoded_location = encode_location(user_input.at[0, 'Location'], location_list)

    user_input = user_input.drop('Location', axis=1)
    input_values = user_input.iloc[0].values.tolist()
    input_values.extend(encoded_location)
    return [input_values]

list_input = prepare_input_for_prediction(default_input, location_list)
print(list_input)

print(len(list_input[0]))
print(X_train.iloc[0])


# 1. Random Forest 
rf_param = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

rf_model = RandomForestRegressor(**rf_param)
rf_model.fit(X_train, y_train)

# print(X_train.iloc[0].to_numpy())
# price = rf_model.predict(np.array(X_test.iloc[1]).reshape(1, -1))

# print(price)
# print(y_test.iloc[1])

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 2. Decision tree
dt_param = {
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

dt_model = DecisionTreeRegressor(**dt_param)
dt_model.fit(X_train, y_train)

with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# 3. ElasticNet Regression
er_param = {
    'alpha': 0.1,
    'l1_ratio': 0.5
}
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

er_model = ElasticNet(**er_param)
er_model.fit(X_train_scaled, y_train)

with open('elasticNet_model.pkl', 'wb') as f:
    pickle.dump(er_model, f)

# 4. Gradient boosting

gb_param = {
    'n_estimators': 200, 
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_split': 10, 
    'min_samples_leaf': 2,    
    'max_features': 'sqrt'
}

gradient_boosting_model = GradientBoostingRegressor(**gb_param)
gradient_boosting_model.fit(X_train, y_train)

with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gradient_boosting_model, f)

# 5. XGBoost

df_list_input = pd.DataFrame(list_input)
df_list_input.columns = X_test.columns
df_types = df_list_input.dtypes
df_type_count = df_types.value_counts()
print(df_type_count)

xgb_param = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 5
}

xgb_model = xgb.XGBRegressor(**xgb_param)
xgb_model.fit(X_train, y_train)

# xgb_model.predict(df_list_input)

print(X_test.iloc[[0]])
print(df_list_input)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)