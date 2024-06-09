# Singapore Resale Flat Price Predictor

## Overview
The Singapore Resale Flat Price Predictor is a machine learning web application designed to estimate the resale prices of flats in Singapore. This tool helps potential buyers and sellers make informed decisions based on various factors such as location, flat type, floor area, and lease duration.

## Motivation
The resale flat market in Singapore is highly competitive, and accurately estimating the resale value of a flat can be challenging. This project aims to provide a solution by developing a predictive model that considers multiple factors affecting resale prices. The resulting web application is intended to assist both buyers and sellers in the Singapore housing market.

## Features
- Predict resale prices of flats based on historical transaction data.
- Utilizes various machine learning models to provide accurate predictions.
- User-friendly web interface built with Streamlit for easy interaction.
- Supports multiple regression models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and K-Neighbors.

## Data
The historical resale flat transaction data used in this project is sourced from multiple CSV files covering different time periods:
- `RFP19901999.csv`
- `RFP20002012.csv`
- `RFP20122014.csv`
- `RFP20152016.csv`
- `RFP20172024.csv`

The datasets include information on:
- Town
- Flat type
- Storey range
- Floor area in square meters
- Flat model
- Lease commence date
- Resale price

## Preprocessing
- The datasets are combined into a single merged dataset.
- Feature engineering is performed to calculate the age of the flats.
- Categorical features are label-encoded, and numerical features are standardized.

## Models
Several machine learning models are trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- K-Neighbors Regressor

The model with the best R² score is selected for deployment.

## Usage
The application is built using Streamlit and can be run locally. Follow the instructions below to set up and run the application.

### Prerequisites
- Python 3.6 or higher
- Required Python libraries (see `requirements.txt`)

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/singapore-resale-flat-price-predictor.git
    cd singapore-resale-flat-price-predictor
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the CSV data files in the `data/` directory.

4. Run the preprocessing and training script:
    ```bash
    python train_model.py
    ```

### Running the Web Application
1. Launch the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to use the application.

### Predicting Resale Prices
1. Select the regression model from the dropdown menu.
2. Enter the required features:
   - Town
   - Flat Type
   - Storey Range
   - Floor Area (sqm)
   - Flat Model
   - Flat Age
3. Click the "Predict Resale Price" button to see the predicted resale price.

## Results
The project benefits both potential buyers and sellers in the Singapore housing market:
- **Buyers** can use the application to estimate resale prices and make informed decisions.
- **Sellers** can get an idea of their flat's potential market value.

Additionally, the project demonstrates the practical application of machine learning in real estate and web development.

## Conclusion
The Singapore Resale Flat Price Predictor is a valuable tool for anyone involved in the Singapore resale flat market. By leveraging historical data and machine learning, it provides accurate price predictions, helping users make better-informed decisions.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Thanks to the Housing & Development Board (HDB) of Singapore for providing the data.
- Gratitude to the open-source community for the tools and libraries used in this project.
