"""
Test script to verify the anomaly detection and fixing functionality.
"""

import os
import pandas as pd
from main import FlightDataCleaner
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Run a test of the anomaly detection and fixing functionality.
    """
    print("Flight Data Anomaly Detection Test")
    print("==================================")

    # Get API key for LLM-powered operations
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found. Please set it in the .env file.")
        print("API key is required for agent-based cleaning and analysis.")
        return

    # Set up the data cleaner
    print("\nSetting up data cleaner...")
    cleaner = FlightDataCleaner("data/Flight Bookings.csv", "data/Airline ID to Name.csv")

    # Load the data
    cleaner.load_data()
    print("Data loaded successfully.")

    # Set up the LLM
    print("Setting up LLM...")
    cleaner.setup_llm(api_key)
    print("LLM setup completed successfully.")

    # Clean column names
    print("\nCleaning column names...")
    cleaner.clean_column_names()
    print("Column names cleaned successfully.")

    # Handle missing values
    print("\nHandling missing values...")
    cleaner.handle_missing_values()
    print("Missing values handled successfully.")

    # Correct data types
    print("\nCorrecting data types...")
    cleaner.correct_data_types()
    print("Data types corrected successfully.")

    # Introduce artificial anomalies for testing
    print("\nIntroducing artificial anomalies for testing...")

    # Print column names to identify the correct columns
    print("Available columns in cleaned data:")
    print(cleaner.flight_data_cleaned.columns.tolist())

    # Make sure we have datetime columns
    if 'departure_time' in cleaner.flight_data_cleaned.columns and 'arrival_time' in cleaner.flight_data_cleaned.columns:
        # 1. Create some arrival < departure anomalies
        print("Creating arrival before departure anomalies...")
        # Get a sample of rows to modify
        sample_indices = cleaner.flight_data_cleaned.sample(n=50).index
        for idx in sample_indices:
            # Swap departure and arrival times to create anomalies
            temp = cleaner.flight_data_cleaned.at[idx, 'departure_time']
            cleaner.flight_data_cleaned.at[idx, 'departure_time'] = cleaner.flight_data_cleaned.at[idx, 'arrival_time']
            cleaner.flight_data_cleaned.at[idx, 'arrival_time'] = temp
        print(f"Created {len(sample_indices)} arrival before departure anomalies.")

        # 2. Create some unrealistic duration anomalies
        # Since we're working with time objects, not datetime, we can't directly add days
        # Instead, we'll set arrival times to be very early in the morning and departure times to be late at night
        print("Creating unrealistic duration anomalies...")
        # Get another sample of rows to modify
        long_duration_indices = cleaner.flight_data_cleaned.sample(n=30).index

        # Import datetime module for time manipulation
        from datetime import time

        for idx in long_duration_indices:
            # Set departure time to late evening (23:00)
            cleaner.flight_data_cleaned.at[idx, 'departure_time'] = time(23, 0, 0)
            # Set arrival time to early morning (1:00)
            cleaner.flight_data_cleaned.at[idx, 'arrival_time'] = time(1, 0, 0)
        print(f"Created {len(long_duration_indices)} unrealistic duration anomalies.")
    else:
        print("Could not find departure_time and arrival_time columns. Skipping datetime anomalies.")

    # 3. Create some negative fare anomalies
    if 'ticket_fare' in cleaner.flight_data_cleaned.columns:
        print("Creating negative fare anomalies...")
        # Get a sample of rows to modify
        negative_fare_indices = cleaner.flight_data_cleaned.sample(n=40).index
        for idx in negative_fare_indices:
            # Make fare negative
            cleaner.flight_data_cleaned.at[idx, 'ticket_fare'] = -1 * abs(cleaner.flight_data_cleaned.at[idx, 'ticket_fare'])
        print(f"Created {len(negative_fare_indices)} negative fare anomalies.")
    else:
        print("Could not find ticket_fare column. Skipping fare anomalies.")

    # 4. Create some unrealistically high fare anomalies
    if 'ticket_fare' in cleaner.flight_data_cleaned.columns:
        print("Creating unrealistically high fare anomalies...")
        # Get a sample of rows to modify
        high_fare_indices = cleaner.flight_data_cleaned.sample(n=20).index
        for idx in high_fare_indices:
            # Make fare unrealistically high
            cleaner.flight_data_cleaned.at[idx, 'ticket_fare'] = 20000 + abs(cleaner.flight_data_cleaned.at[idx, 'ticket_fare'])
        print(f"Created {len(high_fare_indices)} unrealistically high fare anomalies.")
    else:
        print("Could not find ticket_fare column. Skipping high fare anomalies.")

    print("Artificial anomalies introduced successfully.")

    # Detect and fix anomalies
    print("\nDetecting and fixing anomalies...")
    cleaner.detect_and_fix_anomalies()
    print("Anomaly detection and fixing completed.")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
