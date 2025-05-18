import os
import zipfile
import pandas as pd
from feature_engineering import create_snowflake_connection
from snowflake.connector.pandas_tools import write_pandas
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import shutil
import json

def setup_kaggle_credentials():
    """Setup Kaggle credentials from local kaggle.json"""
    try:
        # Read the kaggle.json file from current directory
        with open('kaggle.json', 'r') as f:
            credentials = json.load(f)
            
        # Set environment variables
        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']
        return True
    except Exception as e:
        print(f"Error setting up Kaggle credentials: {e}")
        return False

def download_kaggle_dataset():
    """Download the Airbnb dataset from Kaggle"""
    print("Downloading Airbnb dataset from Kaggle...")
    dataset_name = "joyshil0599/airbnb-listing-data-for-data-science"
    
    try:
        # Setup Kaggle credentials from local file
        if not setup_kaggle_credentials():
            return False
            
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        api.dataset_download_files(dataset_name)
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure kaggle.json is present in the current directory with valid credentials")
        return False

def extract_dataset():
    """Extract the downloaded zip file"""
    print("Extracting dataset...")
    zip_file = "airbnb-listing-data-for-data-science.zip"
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data')
        os.remove(zip_file)  # Remove zip file after extraction
        print("Dataset extracted successfully!")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def clean_and_standardize_dataframe(df):
    """Clean and standardize DataFrame columns and data"""
    # Rename columns to avoid duplicates and standardize names
    column_mapping = {
        'Title': 'TITLE',
        'Detail': 'DETAIL',
        'Details': 'ADDITIONAL_DETAILS',
        'Date': 'LISTING_DATE',
        'Price(in dollar)': 'REGULAR_PRICE',
        'Price(In dollar)': 'DESERT_PRICE',
        'Offer price(in dollar)': 'OFFER_PRICE',
        'Review and rating': 'REVIEW_RATING',
        'Number of bed': 'NUMBER_OF_BEDS',
        'Luxe name': 'LUXE_NAME',
        'Distance': 'DISTANCE',
        'Desert name': 'DESERT_NAME',
        'Rating': 'RATING'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Convert price columns to numeric
    price_columns = ['REGULAR_PRICE', 'DESERT_PRICE', 'OFFER_PRICE']
    for col in price_columns:
        if col in df.columns:
            # Remove '$' and ',' from price strings and convert to float
            df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Convert date column to datetime
    df['LISTING_DATE'] = pd.to_datetime(df['LISTING_DATE'], errors='coerce')
    
    # Convert number of beds to numeric
    df['NUMBER_OF_BEDS'] = pd.to_numeric(df['NUMBER_OF_BEDS'].str.extract('(\d+)', expand=False), errors='coerce')
    
    # Fill NaN values
    df = df.fillna({
        'RATING': 0,
        'NUMBER_OF_BEDS': 0,
        'REVIEW_RATING': 'No reviews',
        'DISTANCE': 'Unknown',
        'LISTING_DATE': pd.Timestamp('1900-01-01')
    })
    
    return df

def create_snowflake_table(conn, df):
    """Create Snowflake table for Airbnb data"""
    # Drop existing table if it exists
    conn.cursor().execute("DROP TABLE IF EXISTS AIRBNB_LISTINGS")
    
    # Map DataFrame dtypes to Snowflake types
    dtype_mapping = {
        'object': 'VARCHAR',
        'int64': 'NUMBER',
        'float64': 'FLOAT',
        'datetime64[ns]': 'TIMESTAMP',
        'bool': 'BOOLEAN'
    }
    
    # Create column definitions
    columns = []
    for col in df.columns:
        # Get DataFrame dtype and map to Snowflake type
        df_type = str(df[col].dtype)
        sf_type = dtype_mapping.get(df_type, 'VARCHAR')
        
        # Add column definition with quoted name
        columns.append(f'"{col}" {sf_type}')
    
    # Create the table
    create_table_query = f"""
    CREATE TABLE AIRBNB_LISTINGS (
        LISTING_ID NUMBER AUTOINCREMENT,
        {','.join(columns)}
    )
    """
    
    conn.cursor().execute(create_table_query)
    print("Snowflake table created successfully!")

def load_data_to_snowflake(conn, df):
    """Load the Airbnb data into Snowflake"""
    print("Loading data into Snowflake...")
    
    try:
        # Write data to Snowflake
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name='AIRBNB_LISTINGS',
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            quote_identifiers=True  # Quote column names
        )
        
        if success:
            print(f"Successfully loaded {nrows} rows into Snowflake!")
            
            # Verify the data
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM AIRBNB_LISTINGS")
            count = cursor.fetchone()[0]
            print(f"Verified {count} rows in Snowflake table")
            
            # Show sample data
            cursor.execute("""
                SELECT LISTING_ID, "TITLE", "REGULAR_PRICE", "NUMBER_OF_BEDS", "RATING"
                FROM AIRBNB_LISTINGS
                LIMIT 5
            """)
            print("\nSample data from Snowflake:")
            for row in cursor.fetchall():
                print(row)
        else:
            print("Error loading data into Snowflake")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

def read_csv_with_encoding(file_path):
    """Read CSV file with different encodings"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")

def main():
    # Load environment variables
    load_dotenv()
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download dataset
    if not download_kaggle_dataset():
        return
    
    # Extract dataset
    if not extract_dataset():
        return
    
    # Read the CSV files
    try:
        # Read and combine all CSV files with proper encoding
        print("Reading CSV files...")
        df_main = read_csv_with_encoding('data/airnb.csv')
        df_luxe = read_csv_with_encoding('data/airnb_luxe.csv')
        df_desert = read_csv_with_encoding('data/airnb_desert.csv')
        
        # Combine all dataframes
        df = pd.concat([df_main, df_luxe, df_desert], ignore_index=True)
        print(f"Loaded {len(df)} records from CSV files")
        
        # Clean and standardize the data
        print("\nCleaning and standardizing data...")
        df = clean_and_standardize_dataframe(df)
        
        # Print column information
        print("\nProcessed dataset columns:")
        for col in df.columns:
            print(f"- {col} ({df[col].dtype})")
        
        # Create Snowflake connection
        print("\nConnecting to Snowflake...")
        conn = create_snowflake_connection()
        
        try:
            # Create Snowflake table with actual schema
            create_snowflake_table(conn, df)
            
            # Load data into Snowflake
            load_data_to_snowflake(conn, df)
            
        finally:
            conn.close()
            print("\nSnowflake connection closed.")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Cleanup
    try:
        for file in ['airnb.csv', 'airnb_luxe.csv', 'airnb_desert.csv']:
            try:
                os.remove(os.path.join('data', file))
            except:
                print(f"Warning: Could not remove {file}")
        try:
            os.rmdir('data')
            print("\nCleanup completed successfully!")
        except:
            print("\nWarning: Could not remove data directory")
    except Exception as e:
        print(f"\nWarning: Could not clean up some temporary files: {e}")

if __name__ == "__main__":
    main() 