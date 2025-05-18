import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

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

def create_sample_data():
    """Create sample customer transaction data"""
    # Generate sample data
    np.random.seed(42)
    n_customers = 100
    n_transactions_per_customer = 50
    
    data = []
    for customer_id in range(n_customers):
        # Generate transactions for each customer
        for _ in range(n_transactions_per_customer):
            transaction_date = datetime.now() - timedelta(
                days=np.random.randint(0, 365)
            )
            amount = np.random.normal(100, 30)  # Transaction amount with mean 100 and std 30
            category = np.random.choice(['grocery', 'electronics', 'clothing', 'entertainment'])
            
            data.append({
                'CUSTOMER_ID': f'CUST_{customer_id:03d}',
                'TRANSACTION_DATE': transaction_date,
                'TRANSACTION_AMOUNT': abs(round(amount, 2)),
                'CATEGORY': category
            })
    
    return pd.DataFrame(data)

def create_transaction_table(conn, df):
    """Create and populate the customer transactions table"""
    # Create the table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS CUSTOMER_TRANSACTIONS (
        TRANSACTION_ID NUMBER AUTOINCREMENT,
        CUSTOMER_ID VARCHAR,
        TRANSACTION_DATE TIMESTAMP,
        TRANSACTION_AMOUNT FLOAT,
        CATEGORY VARCHAR,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    """
    
    conn.cursor().execute(create_table_query)
    
    # Write data to Snowflake
    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=df,
        table_name='CUSTOMER_TRANSACTIONS',
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    
    return success, nrows

def perform_feature_engineering(df):
    """Perform feature engineering on the dataset"""
    # Create time-based features
    df['DAY_OF_WEEK'] = df['TRANSACTION_DATE'].dt.dayofweek
    df['MONTH'] = df['TRANSACTION_DATE'].dt.month
    df['HOUR'] = df['TRANSACTION_DATE'].dt.hour
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)  # New feature
    df['IS_BUSINESS_HOUR'] = df['HOUR'].between(9, 17).astype(int)  # New feature
    
    # Calculate customer-level aggregations
    customer_features = df.groupby('CUSTOMER_ID').agg({
        'TRANSACTION_AMOUNT': ['mean', 'std', 'min', 'max', 'count', 'sum'],  # Added sum
        'CATEGORY': [
            lambda x: x.value_counts().index[0],  # Most frequent category
            'nunique'  # Number of unique categories
        ],
        'TRANSACTION_DATE': [
            lambda x: (x.max() - x.min()).days,  # Customer lifetime in days
            'count'  # Total number of transactions
        ],
        'IS_WEEKEND': 'mean',  # Proportion of weekend transactions
        'IS_BUSINESS_HOUR': 'mean'  # Proportion of business hour transactions
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = [
        'CUSTOMER_ID',
        'AVG_TRANSACTION_AMOUNT',
        'STD_TRANSACTION_AMOUNT',
        'MIN_TRANSACTION_AMOUNT',
        'MAX_TRANSACTION_AMOUNT',
        'TRANSACTION_COUNT',
        'TOTAL_AMOUNT',
        'MOST_FREQUENT_CATEGORY',
        'CATEGORY_DIVERSITY',
        'CUSTOMER_LIFETIME_DAYS',
        'TOTAL_TRANSACTIONS',
        'WEEKEND_TRANSACTION_RATIO',
        'BUSINESS_HOUR_RATIO'
    ]
    
    # Calculate category-specific metrics
    category_pivot = pd.pivot_table(
        df,
        index='CUSTOMER_ID',
        columns='CATEGORY',
        values='TRANSACTION_AMOUNT',
        aggfunc=['count', 'mean', 'sum'],
        fill_value=0
    ).reset_index()
    
    # Flatten category pivot columns
    category_pivot.columns = [
        'CUSTOMER_ID' if col[0] == '' else f'{col[1].upper()}_{col[0].upper()}_AMOUNT'
        for col in category_pivot.columns
    ]
    
    # Calculate rolling statistics (last 7 transactions)
    df = df.sort_values('TRANSACTION_DATE')
    
    # Rolling amount statistics
    for window in [7, 30]:  # Add 30-day window
        df[f'ROLLING_{window}D_MEAN_AMOUNT'] = df.groupby('CUSTOMER_ID')['TRANSACTION_AMOUNT'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'ROLLING_{window}D_STD_AMOUNT'] = df.groupby('CUSTOMER_ID')['TRANSACTION_AMOUNT'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f'ROLLING_{window}D_MAX_AMOUNT'] = df.groupby('CUSTOMER_ID')['TRANSACTION_AMOUNT'].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    
    # Get the most recent transaction features for each customer
    recent_features = df.sort_values('TRANSACTION_DATE').groupby('CUSTOMER_ID').last().reset_index()
    recent_cols = [col for col in df.columns if 'ROLLING' in col]
    recent_features = recent_features[['CUSTOMER_ID'] + recent_cols]
    
    # Calculate transaction frequency features
    frequency_features = df.groupby('CUSTOMER_ID').agg({
        'TRANSACTION_DATE': lambda x: np.mean(np.diff(sorted(x)).astype('timedelta64[D]').astype(float))
    }).reset_index()
    frequency_features.columns = ['CUSTOMER_ID', 'AVG_DAYS_BETWEEN_TRANSACTIONS']
    
    # Merge all features
    final_features = customer_features.merge(category_pivot, on='CUSTOMER_ID')
    final_features = final_features.merge(recent_features, on='CUSTOMER_ID')
    final_features = final_features.merge(frequency_features, on='CUSTOMER_ID')
    
    # Fill NaN values
    final_features = final_features.fillna(0)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = [
        col for col in final_features.columns 
        if col not in ['CUSTOMER_ID', 'MOST_FREQUENT_CATEGORY'] 
        and final_features[col].dtype in ['int64', 'float64']
    ]
    
    final_features[numerical_cols] = scaler.fit_transform(final_features[numerical_cols])
    
    return final_features

def store_features(conn, features_df):
    """Store features in Snowflake Feature Store"""
    # Create feature store table with all columns
    columns = [f"{col} FLOAT" if col not in ['CUSTOMER_ID', 'MOST_FREQUENT_CATEGORY'] else f"{col} VARCHAR"
              for col in features_df.columns if col != 'CREATED_AT']
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS CUSTOMER_FEATURES (
        FEATURE_ID NUMBER AUTOINCREMENT,
        {','.join(columns)},
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    """
    
    conn.cursor().execute(create_table_query)
    
    # Write features to Snowflake
    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=features_df,
        table_name='CUSTOMER_FEATURES',
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    
    return success, nrows

def retrieve_features(conn, customer_ids=None):
    """Retrieve features from Feature Store"""
    cursor = conn.cursor()
    
    if customer_ids:
        query = f"""
        SELECT *
        FROM CUSTOMER_FEATURES
        WHERE CUSTOMER_ID IN ({','.join([f"'{id}'" for id in customer_ids])})
        ORDER BY CREATED_AT DESC
        """
    else:
        query = """
        SELECT *
        FROM CUSTOMER_FEATURES
        ORDER BY CREATED_AT DESC
        """
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    features_df = pd.DataFrame(cursor.fetchall(), columns=columns)
    
    return features_df

def main():
    # Create Snowflake connection
    print("Connecting to Snowflake...")
    conn = create_snowflake_connection()
    
    try:
        # Generate and store sample data
        print("\nGenerating sample data...")
        raw_data = create_sample_data()
        print(f"Generated {len(raw_data)} sample transactions")
        
        print("\nStoring transactions in Snowflake...")
        success, nrows = create_transaction_table(conn, raw_data)
        print(f"Successfully stored {nrows} transactions")
        
        # Perform feature engineering
        print("\nPerforming feature engineering...")
        features_df = perform_feature_engineering(raw_data)
        print(f"Generated features for {len(features_df)} customers")
        
        # Store features in Feature Store
        print("\nStoring features in Feature Store...")
        success, nrows = store_features(conn, features_df)
        print(f"Successfully stored features for {nrows} customers")
        
        # Retrieve and display sample features
        print("\nRetrieving features for sample customers...")
        sample_customers = features_df['CUSTOMER_ID'].head(5).tolist()
        retrieved_features = retrieve_features(conn, sample_customers)
        print("\nSample features:")
        print(retrieved_features[['CUSTOMER_ID', 'AVG_TRANSACTION_AMOUNT', 'TRANSACTION_COUNT', 'MOST_FREQUENT_CATEGORY']])
        
    finally:
        conn.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    main() 