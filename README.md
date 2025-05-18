# Vistora Ai Project
"This project demonstrates a complete pipeline from data loading, feature engineering, model training, to prediction and deployment readiness."
a. Loading Data – load_airbnb_data.py
"This script likely uses Kaggle credentials to download Airbnb datasets or reads CSVs, then does basic cleaning."

b. Feature Engineering – feature_engineering.py
"It transforms columns – encoding categorical values, handling missing data, scaling features. This is crucial for making the data ML-friendly."

c. Model Training – train_model.py
"Here, we train a regression model – possibly using RandomForestRegressor or similar. It uses engineered features and saves:

The trained model (airbnb_price_model.joblib)

The scaler and column encoders (e.g., airbnb_feature_columns.joblib)"

d. Prediction – predict.py
"This script loads saved models and makes predictions on new Airbnb listings using the same feature pipeline."

The model gives a decent performance with an R² of 0.53 and an average error of around $362.

The most important features are:

Beds (72% importance),

Desert location, and

Rating.

For example:

A 2-bed house: $139,

A 5-bed luxe chalet in desert: $329.

So beds and desert location impact price the most.
