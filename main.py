# Importing important libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Reading the data

data_1 = pd.read_csv(r'C:\Users\SAMEER YADAV\Downloads\Surabhi\singapore_resale_flat_prices_prediction\RFP19901999.csv')
data_2 = pd.read_csv(r'C:\Users\SAMEER YADAV\Downloads\Surabhi\singapore_resale_flat_prices_prediction\RFP20002012.csv')
data_3 = pd.read_csv(r'C:\Users\SAMEER YADAV\Downloads\Surabhi\singapore_resale_flat_prices_prediction\RFP20122014.csv')
data_4 = pd.read_csv(r'C:\Users\SAMEER YADAV\Downloads\Surabhi\singapore_resale_flat_prices_prediction\RFP20152016.csv')
data_5 = pd.read_csv(r'C:\Users\SAMEER YADAV\Downloads\Surabhi\singapore_resale_flat_prices_prediction\RFP20172024.csv')

# Displaying basic info of the dataset

#print("Displaying data from 1990 to 1999", '\n', data_1.info())
#print("Displaying data from 1990 to 1999", '\n' ,data_1.isnull().sum(), '\n\n')
#print("Displaying data from 2000 to 2012", '\n' ,data_2.isnull().sum(), '\n\n')
#print("Displaying data from 2012 to 2014", '\n' ,data_3.isnull().sum(), '\n\n')
#print("Displaying data from 2015 to 2016", '\n' ,data_4.isnull().sum(), '\n\n')
#print("Displaying data from 2017 to 2024", '\n' ,data_5.isnull().sum(), '\n\n')

# Removing a column from some datasets

data_4 = data_4.drop(columns=['remaining_lease'])
data_5 = data_5.drop(columns=['remaining_lease'])
    
#print("Displaying data from 2015 to 2016", '\n' ,data_4.isnull().sum(), '\n\n')
#print("Displaying data from 2017 to 2024", '\n' ,data_5.isnull().sum(), '\n\n')

# Feature Engineering

dataframes = [data_1, data_2, data_3, data_4, data_5]

for df in dataframes:
    df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], format='%Y')
    df['flat_age'] = pd.to_datetime('today').year - df['lease_commence_date'].dt.year

data_1.to_csv('updated_data_1.csv', index=False)
data_2.to_csv('updated_data_2.csv', index=False)
data_3.to_csv('updated_data_3.csv', index=False)
data_4.to_csv('updated_data_4.csv', index=False)
data_5.to_csv('updated_data_5.csv', index=False)

#print("The 'lease_commence_date' has been converted and 'flat_age' calculated for all datasets.")

# Merge all the .csv files

csv_files = ['updated_data_1.csv', 'updated_data_2.csv', 'updated_data_3.csv', 'updated_data_4.csv', 'updated_data_5.csv']
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
merged_df = merged_df.drop(columns=['lease_commence_date'])

merged_df.to_csv('merged_data.csv', index=False)
#print("All CSV files have been merged into 'merged_data.csv'.")

data = pd.read_csv(r'C:\Users\SAMEER YADAV\Downloads\Surabhi\singapore_resale_flat_prices_prediction\merged_data.csv')
#print("Displaying all the data", '\n', data.info())

# Selecting the relevant features and target variable

features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'flat_age']
target = 'resale_price'

# Label encoding categorical features

categorical_features = ['town', 'flat_type', 'storey_range', 'flat_model']
numerical_features = ['floor_area_sqm', 'flat_age']

label_encoders = {}
for col in categorical_features:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col].astype(str))

encoded_categorical_features = data[categorical_features]

# Standardizing numerical features

scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(data[numerical_features])

# Combining encoded categorical and scaled numerical features

processed_features = np.hstack([encoded_categorical_features, scaled_numerical_features])
encoded_feature_names = categorical_features  # Using original column names for label encoding
processed_feature_names = encoded_feature_names + numerical_features

# Prepare the final dataset

X = pd.DataFrame(processed_features, columns=processed_feature_names)
y = data[target]

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train)

# Define and train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    # "XGBoost": XGBRegressor(),
    "KNeighbors": KNeighborsRegressor()
}

# Train and evaluate each model
r2_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Model: {name}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2_score(y_test, y_pred)}\n")
    
    r2_scores[name] = r2
    
# Compare R² scores
best_model = max(r2_scores, key=r2_scores.get)
print(f"Best model based on R² score: {best_model}")
print(f"R² scores: {r2_scores}")

# Load the best model (e.g., Random Forest)
best_model = models[best_model]

# Function to preprocess user input
def preprocess_input(town, flat_type, storey_range, floor_area_sqm, flat_model, flat_age):
    #flat_age = pd.to_datetime('today').year - lease_commence_date
    input_data = pd.DataFrame([[town, flat_type, storey_range, floor_area_sqm, flat_model, flat_age]],
                              columns=['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'flat_age'])
    #encoded_input = encoder.transform(input_data[categorical_features])
    for col in categorical_features:
        input_data[col] = label_encoders[col].fit_transform(input_data[col].astype(str))
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    return input_data

# Streamlit app layout
st.title('Singapore Resale Flat Price Predictor')

town = st.selectbox('Town', data['town'].unique())
flat_type = st.selectbox('Flat Type', data['flat_type'].unique())
storey_range = st.selectbox('Storey Range', data['storey_range'].unique())
floor_area_sqm = st.number_input('Floor Area (sqm)')
flat_model = st.selectbox('Flat Model', data['flat_model'].unique())
#lease_commence_date = st.number_input('Lease Commence Year', min_value=1990, max_value=pd.to_datetime('today').year)
flat_age = st.number_input('Flat Age', min_value = 0, max_value=100)

if st.button('Predict'):
    user_input = preprocess_input(town, flat_type, storey_range, floor_area_sqm, flat_model, flat_age)
    prediction = best_model.predict(user_input)
    st.write(f'Predicted Resale Price: ${prediction[0]:,.2f}')