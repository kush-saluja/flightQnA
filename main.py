"""
Flight Data Analysis System - Main Entry Point

This script serves as the entry point for the Flight Data Analysis System.
It imports the necessary components from the modular structure and runs the interface.
"""

import os
from dotenv import load_dotenv
from interface import FlightDataInterface

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get file paths from environment variables or use defaults
    flight_data_path = os.environ.get("FLIGHT_DATA_PATH", "data/Flight Bookings.csv")
    airline_mapping_path = os.environ.get("AIRLINE_MAPPING_PATH", "data/Airline ID to Name.csv")

    # Create and run the interface with the specified file paths
    interface = FlightDataInterface()
    interface.run(flight_data_path, airline_mapping_path)