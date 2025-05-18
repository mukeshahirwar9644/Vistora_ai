import snowflake.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import os
import joblib

# Load environment variables
load_dotenv()

# Snowflake connection parameters
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')

def create_snowflake_connection():
    """Create a connection to Snowflake"""
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    return conn

def get_features():
    """Retrieve all features from the feature store"""
    conn = create_snowflake_connection()
    try:
        query = """
        SELECT 
            CUSTOMER_ID,
            AVG_TRANSACTION_AMOUNT,
            STD_TRANSACTION_AMOUNT,
            MIN_TRANSACTION_AMOUNT,
            MAX_TRANSACTION_AMOUNT,
            TRANSACTION_COUNT,
            MOST_FREQUENT_CATEGORY,
            ROLLING_MEAN_AMOUNT,
            ROLLING_STD_AMOUNT
        FROM CUSTOMER_FEATURES
        ORDER BY CREATED_AT DESC
        """
        
        features_df = pd.read_sql(query, conn)
        return features_df
    
    finally:
        conn.close()

def prepare_features(df):
    """Prepare features for model training"""
    # Convert categorical variables to dummy variables
    features_df = pd.get_dummies(df, columns=['MOST_FREQUENT_CATEGORY'])
    
    # Separate features and target
    X = features_df.drop(['CUSTOMER_ID', 'AVG_TRANSACTION_AMOUNT'], axis=1)
    y = features_df['AVG_TRANSACTION_AMOUNT']
    
    return X, y

def train_model(X, y):
    """Train a Random Forest model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    return model, X_train.columns

def save_model(model, feature_columns):
    """Save the trained model and feature columns"""
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'models/customer_transaction_model.joblib')
    
    # Save feature columns
    pd.Series(feature_columns).to_pickle('models/feature_columns.pkl')
    
    print("\nModel and feature columns saved to 'models' directory")

def main():
    print("Retrieving features from Feature Store...")
    features_df = get_features()
    print(f"Retrieved {len(features_df)} customer features")
    
    print("\nPreparing features for training...")
    X, y = prepare_features(features_df)
    
    print("\nTraining model...")
    model, feature_columns = train_model(X, y)
    
    print("\nSaving model...")
    save_model(model, feature_columns)

if __name__ == "__main__":
    main() 