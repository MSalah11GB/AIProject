import streamlit as st
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

# Define the user guide content
def show_user_guide():
    st.sidebar.title("User Guide")
    st.sidebar.markdown("""
    This app helps you predict house prices in Bengaluru, India based on various features.
    
    **Instructions:**
    1. Adjust the sliders or input fields in the sidebar to specify:
       - Number of rooms
       - Area in square feet
       - Location
       - Number of bathrooms
       - Number of balconies
    2. Ensure all inputs are valid and within the specified ranges.
    3. After selecting your inputs, click on the 'Predict' button to see the predicted house price.
    
    **Notes:**
    - The predictions are based on machine learning models trained on historical data.
    - Explore different models (Random Forest, Decision Tree, Gradient Boosting, XGBoost) to compare predictions.
    
    **Additional Options:**
    - You can select different models from the sidebar to see predictions from specific models.
    
    Enjoy predicting house prices in Bengaluru!
    """)

st.write("""
# Bengaluru House Price Prediction App

This app predicts the house price in **Bengaluru, India** 

""")

# Loads the data
data = pd.read_csv("cleaned_data.csv")

data = data.dropna(how='any')
data = data.drop('price_per_sqft', axis='columns')

X = data.drop(['price'], axis='columns')
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

location_list = X.columns[4:].tolist()

show_user_guide()

# Sidebar
def validate_float_input(input_str, min_val, max_val):
    try:
        input_float = float(input_str)
        if min_val <= input_float <= max_val:
            return input_float
        else:
            st.error(f"Please enter a number between {min_val} and {max_val}.")
            return None
    except ValueError:
        st.error("Please enter a valid number.")
        return None

min_value = X['total_sqft'].min()
max_value = 10000.0

def user_input_features():
    noOfRooms = st.sidebar.slider('No. of Rooms', X['size'].min(), X['size'].max(), 1)
    
    totalSquareFoot_str = st.sidebar.text_input('Area (ft^2)', min_value)
    totalSquareFoot = validate_float_input(totalSquareFoot_str, min_value, max_value)
    if totalSquareFoot is not None:
        totalSquareFoot = float(totalSquareFoot)
    
    location = st.sidebar.selectbox('Location', location_list)
    bath = st.sidebar.slider('No. of Bathrooms', int(X['bath'].min()), int(X['bath'].max()), 1)
    balcony = st.sidebar.slider('No. of Balcony', int(X['balcony'].min()), int(X['balcony'].max()), 1)
    
    data = {
        'No. of Rooms': noOfRooms,
        'Area (ft^2)': totalSquareFoot,
        'Location': location,
        'No. of Bathrooms': bath,
        'No. of Balcony': balcony
    }

    if totalSquareFoot is not None:
        st.success(f"Valid input: {totalSquareFoot}")

    features = pd.DataFrame(data, index=[0])

    return features

default_input = user_input_features()

# Main panel

st.header('Specified Input parameters')
st.write(default_input)
st.write('---')

# Encode location value
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

# Building Regression Model
# Random forest

rf_model = pickle.load(open('rf_model.pkl', 'rb'))
rf_prediction = rf_model.predict(list_input)

# Decision tree

dt_model = pickle.load(open('dt_model.pkl', 'rb'))
dt_prediction = dt_model.predict(list_input)

# Gradient Boosting

gb_model = pickle.load(open('gb_model.pkl', 'rb'))
gb_prediction = gb_model.predict(list_input)

print(list_input)

# XGBoost

xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
df_list_input = pd.DataFrame(list_input)
df_list_input.columns = X_test.columns


xgb_prediction = xgb_model.predict(df_list_input)

# Create a DataFrame to display predictions
predictions_df = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'XG Boost'],
    'Prediction': [rf_prediction[0], dt_prediction[0], gb_prediction[0], xgb_prediction[0]]
})

st.header('Predictions')
st.write('The following predictions are measured in 100 000 rupees')
st.dataframe(predictions_df)
st.write('---')

#define confidence score for models

rf_conf = 0.75
dt_conf = 1.31
gb_conf = 2.30
xgb_conf = 2.08

conf_score = [rf_conf, dt_conf, gb_conf, xgb_conf]

transformed_score = [np.exp(-x) for x in conf_score]

sum_values = sum(transformed_score)
normalized_values = [x / sum_values for x in transformed_score]
normalized_array = np.array(normalized_values)

original_score = np.array([rf_prediction[0], dt_prediction[0], gb_prediction[0], xgb_prediction[0]])
final_price = original_score.dot(normalized_array)

RUPEE_TO_USD = 0.012

# Calculate formatted prices
formatted_price_in_rupees = "{:,.0f}".format(final_price * 100000)
formatted_price_in_rupees = formatted_price_in_rupees.replace(",", " ")

formatted_price_in_usd = "{:,.2f}".format(final_price * 100000 * RUPEE_TO_USD)
formatted_price_in_usd = formatted_price_in_usd.replace(",", " ")

# Display prices
st.write('The price you should pay for this real estate: **' + formatted_price_in_rupees + '** Rupees, or **US$** ' + formatted_price_in_usd)