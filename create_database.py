import snowflake.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Snowflake connection parameters
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')

# Print configuration (without password)
print("Configuration:")
print(f"User: {SNOWFLAKE_USER}")
print(f"Account: {SNOWFLAKE_ACCOUNT}")

def create_snowflake_database():
    """Create a new database in Snowflake"""
    try:
        print("Attempting to connect to Snowflake...")
        
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT
        )
        
        print("Successfully connected to Snowflake!")
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Create database
        database_name = "VISTORA_DB"
        print(f"\nCreating database '{database_name}'...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"Database '{database_name}' created successfully!")
        
        # Create schema
        schema_name = "VISTORA_SCHEMA"
        print(f"\nCreating schema '{schema_name}'...")
        cursor.execute(f"USE DATABASE {database_name}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        print(f"Schema '{schema_name}' created successfully!")
        
        # Create a warehouse if it doesn't exist
        warehouse_name = "VISTORA_WH"
        print(f"\nCreating warehouse '{warehouse_name}'...")
        cursor.execute(f"""
        CREATE WAREHOUSE IF NOT EXISTS {warehouse_name}
        WITH WAREHOUSE_SIZE = 'X-SMALL'
        AUTO_SUSPEND = 300
        AUTO_RESUME = TRUE
        """)
        print(f"Warehouse '{warehouse_name}' created successfully!")
        
        # Show the created objects
        print("\nCreated Snowflake objects:")
        cursor.execute("SHOW DATABASES LIKE 'VISTORA_DB'")
        print("\nDatabase details:")
        for row in cursor.fetchall():
            print(row)
            
        cursor.execute("SHOW SCHEMAS IN DATABASE VISTORA_DB")
        print("\nSchema details:")
        for row in cursor.fetchall():
            print(row)
            
        cursor.execute("SHOW WAREHOUSES LIKE 'VISTORA_WH'")
        print("\nWarehouse details:")
        for row in cursor.fetchall():
            print(row)
            
    except Exception as e:
        print(f"\nError occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
            print("\nConnection closed.")

if __name__ == "__main__":
    create_snowflake_database() 