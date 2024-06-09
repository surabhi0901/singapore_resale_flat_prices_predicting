# Importing important libraries

import pandas as pd
import streamlit as st
import pickle

# Streamlit app layout

st.title('Singapore Resale Flat Price Predictor')
st.write("")
st.write("")

# Loading the scaler

with open('C:/Users/sy090/Downloads/PROJECTS/singapore_resale_flat_prices_prediction/scaler_r.pkl', 'rb') as f:
    scaler_r = pickle.load(f)

# Loading the regression model 

regression_models = {}
for model_name in ["linear_regression", "decision_tree", "random_forest", 
                    "gradient_boosting", "xgboost", "kneighbors"]:
    with open(f'C:/Users/sy090/Downloads/PROJECTS/singapore_resale_flat_prices_prediction/{model_name}_regressor.pkl', 'rb') as f:
            regression_models[model_name] = pickle.load(f)

# Running the pre-trained regression model

reg_model_name = st.selectbox('Select Regression Model', list(regression_models.keys()))
regressor = regression_models[reg_model_name]

#features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'flat_age']
data = pd.read_csv(r'C:\Users\sy090\Downloads\PROJECTS\singapore_resale_flat_prices_prediction\merged_data.csv')
X_reg = data[['town', 'flat_type', 'storey_range', 'flat_model', 'floor_area_sqm', 'flat_age']]
y_reg = data['resale_price']
inputs_r = {}
st.write("Town")
st.write('''['ANG MO KIO':0, 'BEDOK':1, 'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4, 'BUKIT TIMAH':5,
 'CENTRAL AREA':6, 'CHOA CHU KANG':7, 'CLEMENTI':8, 'GEYLANG':9, 'HOUGANG':10,
 'JURONG EAST':11, 'JURONG WEST':12, 'KALLANG/WHAMPOA':13, 'MARINE PARADE':14,
 'QUEENSTOWN':15, 'SENGKANG':16, 'SERANGOON':17, 'TAMPINES':18, 'TOA PAYOH':19, 'WOODLANDS':20,
 'YISHUN':21, 'LIM CHU KANG':22, 'SEMBAWANG':23, 'BUKIT PANJANG':24, 'PASIR RIS':25, 'PUNGGOL':26]''')
st.write("")
st.write("Flat Type")
st.write('''['1 ROOM':0, '3 ROOM':1, '4 ROOM':2, '5 ROOM':3, '2 ROOM':4, 'EXECUTIVE':5,
 'MULTI GENERATION':6, 'MULTI-GENERATION':7]''')
st.write("")
st.write("Storey Range")
st.write('''['10 TO 12':0, '04 TO 06':1, '07 TO 09':2, '01 TO 03':3, '13 TO 15':4, '19 TO 21':5,
 '16 TO 18':6, '25 TO 27':7, '22 TO 24':8, '28 TO 30':9, '31 TO 33':10, '40 TO 42':11,
 '37 TO 39':12, '34 TO 36':13, '06 TO 10':14, '01 TO 05':15, '11 TO 15':16, '16 TO 20':17,
 '21 TO 25':18, '26 TO 30':19, '36 TO 40':20, '31 TO 35':21, '46 TO 48':22, '43 TO 45':23,
 '49 TO 51':24]''')
st.write("")
st.write("Flat Model")
st.write('''['IMPROVED':0, 'NEW GENERATION':1, 'MODEL A':2, 'STANDARD':3, 'SIMPLIFIED':4,
 'MODEL A-MAISONETTE':5, 'APARTMENT':6, 'MAISONETTE':7, 'TERRACE':8, '2-ROOM':9,
 'IMPROVED-MAISONETTE':10, 'MULTI GENERATION':11, 'PREMIUM APARTMENT':12, 'Improved':13,
 'New Generation':14, 'Model A':15 'Standard':16, 'Apartment':17, 'Simplified':18,
 'Model A-Maisonette':19, 'Maisonette':20, 'Multi Generation':21, 'Adjoined flat':22,
 'Premium Apartment':23, 'Terrace':24, 'Improved-Maisonette':25, 'Premium Maisonette':26,
 '2-room':27, 'Model A2':28, 'DBSS':29, 'Type S1':30 'Type S2':31, 'Premium Apartment Loft':32,
 '3Gen':33]''')
st.write("")
st.write("")
for col in X_reg.columns:
    inputs_r[col] = st.text_input(f'Enter value for {col}')

# Predicting Resale price
if st.button('Predict Resale Price'):
    input_df_r = pd.DataFrame([inputs_r])
    input_df_r = input_df_r.apply(pd.to_numeric, errors='ignore')
    input_df_r = scaler_r.transform(input_df_r)
    prediction = regressor.predict(input_df_r)
    st.write(f'Predicted Resale Price: {prediction[0]}')