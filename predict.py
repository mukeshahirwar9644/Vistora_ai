import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engineering import create_snowflake_connection
import os
from dotenv import load_dotenv

def get_training_data(conn):
    """Get training data from Snowflake"""
    query = """
    SELECT 
        "REGULAR_PRICE",
        "NUMBER_OF_BEDS",
        "RATING",
        CASE WHEN "DESERT_PRICE" IS NOT NULL THEN 1 ELSE 0 END as IS_DESERT,
        CASE WHEN "LUXE_NAME" IS NOT NULL THEN 1 ELSE 0 END as IS_LUXE,
        COALESCE("DESERT_PRICE", "REGULAR_PRICE") as FINAL_PRICE,
        CASE 
            WHEN "TITLE" LIKE '%cabin%' THEN 'cabin'
            WHEN "TITLE" LIKE '%house%' THEN 'house'
            WHEN "TITLE" LIKE '%apartment%' THEN 'apartment'
            WHEN "TITLE" LIKE '%villa%' THEN 'villa'
            WHEN "TITLE" LIKE '%chalet%' THEN 'chalet'
            ELSE 'other'
        END as PROPERTY_TYPE
    FROM AIRBNB_LISTINGS
    WHERE COALESCE("DESERT_PRICE", "REGULAR_PRICE") > 0
      AND COALESCE("DESERT_PRICE", "REGULAR_PRICE") < 10000  -- Remove extreme outliers
    """
    
    return pd.read_sql(query, conn)

def train_price_prediction_model(conn):
    """Train a model to predict listing prices"""
    print("Training price prediction model...")
    
    # Get data from Snowflake
    df = get_training_data(conn)
    
    # Create property type dummy variables
    property_dummies = pd.get_dummies(df['PROPERTY_TYPE'], prefix='PROPERTY')
    
    # Prepare features
    numeric_features = df[['NUMBER_OF_BEDS', 'RATING', 'IS_DESERT', 'IS_LUXE']]
    X = pd.concat([numeric_features, property_dummies], axis=1)
    y = df['FINAL_PRICE']
    
    # Store feature names
    feature_names = list(X.columns)
    
    # Log transform the target variable
    y = np.log1p(y)  # log1p handles zero values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/airbnb_price_model.joblib')
    joblib.dump(scaler, 'models/airbnb_price_scaler.joblib')
    joblib.dump(feature_names, 'models/airbnb_feature_columns.joblib')
    
    # Get unique property types for later use
    property_types = df['PROPERTY_TYPE'].unique()
    joblib.dump(property_types, 'models/airbnb_property_types.joblib')
    
    # Evaluate model
    train_predictions = np.expm1(model.predict(X_train_scaled))  # Convert back from log scale
    test_predictions = np.expm1(model.predict(X_test_scaled))
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² Score: {train_score:.2f}")
    print(f"Testing R² Score: {test_score:.2f}")
    
    # Calculate error metrics
    train_mae = np.mean(np.abs(train_predictions - np.expm1(y_train)))
    test_mae = np.mean(np.abs(test_predictions - np.expm1(y_test)))
    
    print(f"Training MAE: ${train_mae:.2f}")
    print(f"Testing MAE: ${test_mae:.2f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        if row['importance'] >= 0.01:  # Only show features with at least 1% importance
            print(f"{row['feature']}: {row['importance']:.1%}")
    
    return model, scaler, feature_names

def predict_listing_price(beds, rating, property_type='house', is_desert=False, is_luxe=False):
    """Predict the price for a new listing"""
    try:
        # Load model and metadata
        model = joblib.load('models/airbnb_price_model.joblib')
        scaler = joblib.load('models/airbnb_price_scaler.joblib')
        feature_names = joblib.load('models/airbnb_feature_columns.joblib')
        property_types = joblib.load('models/airbnb_property_types.joblib')
        
        # Create input data with numeric features
        input_data = pd.DataFrame({
            'NUMBER_OF_BEDS': [beds],
            'RATING': [rating],
            'IS_DESERT': [int(is_desert)],
            'IS_LUXE': [int(is_luxe)]
        })
        
        # Add property type dummy variables
        for prop_type in property_types:
            col_name = f'PROPERTY_{prop_type}'
            if col_name in feature_names:
                input_data[col_name] = 1 if property_type == prop_type else 0
        
        # Ensure all features are present and in correct order
        missing_cols = set(feature_names) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[feature_names]
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction and convert back from log scale
        predicted_price = np.expm1(model.predict(input_scaled)[0])
        
        return round(predicted_price, 2)
        
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    # Load environment variables
    load_dotenv()
    
    # Create Snowflake connection
    print("Connecting to Snowflake...")
    conn = create_snowflake_connection()
    
    try:
        # Train or retrain the model
        print("\nTraining model...")
        train_price_prediction_model(conn)
        
        # Example predictions
        print("\nMaking sample predictions:")
        
        test_cases = [
            (2, 4.5, 'house', False, False, "Standard 2-bed house with good rating"),
            (4, 4.8, 'villa', False, True, "Luxe 4-bed villa with excellent rating"),
            (3, 4.0, 'cabin', True, False, "Desert 3-bed cabin with good rating"),
            (5, 4.9, 'chalet', True, True, "Luxe desert 5-bed chalet with top rating"),
            (1, 4.2, 'apartment', False, False, "Standard 1-bed apartment with good rating")
        ]
        
        for beds, rating, prop_type, is_desert, is_luxe, description in test_cases:
            price = predict_listing_price(beds, rating, prop_type, is_desert, is_luxe)
            print(f"\n{description}:")
            print(f"- Property Type: {prop_type}")
            print(f"- Beds: {beds}")
            print(f"- Rating: {rating}")
            print(f"- Desert Location: {'Yes' if is_desert else 'No'}")
            print(f"- Luxe Property: {'Yes' if is_luxe else 'No'}")
            print(f"Predicted Price: ${price:.2f}")
        
    finally:
        conn.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    main() 