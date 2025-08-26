"""
Flight Data Analyzer Module

This module contains the FlightDataAnalyzer class, which is responsible for
analyzing flight booking data using LLM-powered methods.
"""

from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables from .env file
load_dotenv()


class FlightDataAnalyzer:
    """
    A class to analyze flight booking data using LLM-powered methods.
    """

    def __init__(self, flight_data):
        """
        Initialize the FlightDataAnalyzer with the cleaned flight data.

        Args:
            flight_data (pandas.DataFrame): The cleaned flight booking data
        """
        self.flight_data = flight_data
        self.llm = None
        self.agent = None

    def setup_llm(self, api_key=None):
        """
        Set up the LLM for data analysis.

        Args:
            api_key (str, optional): OpenAI API key. Defaults to None.
        """
        if api_key:
            import openai
            openai.api_key = api_key

        try:
            self.llm = OpenAI(temperature=0, max_tokens=5000, model='gpt-4o-mini')
            self.agent = create_pandas_dataframe_agent(self.llm, self.flight_data, verbose=True,
                                                       allow_dangerous_code=True, max_execution_time=100,
                                                       max_iterations=25)
            print("Flight data analysis agent set up successfully.")
        except Exception as e:
            print(f"Error setting up LLM for analysis: {e}")

    def run_analysis(self, question):
        """
        Run a natural language analysis on the flight data.

        Args:
            question (str): The natural language question to analyze

        Returns:
            str: The analysis result
        """
        if self.agent is None:
            print("Agent not set up. Please run setup_llm() first.")
            return "Error: Agent not set up. Please run setup_llm() first."

        try:
            result = self.agent.run(question)
            return result
        except Exception as e:
            print(f"Error running analysis: {e}")
            return f"Error running analysis: {e}"

    def most_flights_by_airline(self):
        """
        Find the airline with the most flights.

        Returns:
            tuple: (airline_name, flight_count)
        """
        try:
            # Group by airline and count flights
            airline_counts = self.flight_data['airline_name'].value_counts()

            # Get the airline with the most flights
            top_airline = airline_counts.index[0]
            top_count = airline_counts.iloc[0]

            return (top_airline, top_count)
        except Exception as e:
            print(f"Error finding most flights by airline: {e}")
            return ("Unknown", 0)

    def top_destinations(self, n=5):
        """
        Find the top N destinations by number of flights.

        Args:
            n (int, optional): Number of top destinations to return. Defaults to 5.

        Returns:
            list: List of tuples (destination, flight_count)
        """
        try:
            # Find the destination column
            destination_col = None
            for col in self.flight_data.columns:
                if 'destination' in col.lower() or 'arrival_city' in col.lower() or 'to_city' in col.lower():
                    destination_col = col
                    break

            if destination_col is None:
                print("Could not find destination column.")
                return [("Unknown", 0)] * n

            # Group by destination and count flights
            destination_counts = self.flight_data[destination_col].value_counts().head(n)

            # Convert to list of tuples
            return list(zip(destination_counts.index, destination_counts.values))
        except Exception as e:
            print(f"Error finding top destinations: {e}")
            return [("Unknown", 0)] * n

    def month_with_most_bookings(self):
        """
        Find the month with the most bookings.

        Returns:
            tuple: (month_name, booking_count)
        """
        try:
            # Find the date column
            date_col = None
            for col in self.flight_data.columns:
                if 'date' in col.lower() and ('book' in col.lower() or 'depart' in col.lower()):
                    date_col = col
                    break

            if date_col is None:
                print("Could not find booking date column.")
                return ("Unknown", 0)

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(self.flight_data[date_col]):
                self.flight_data[date_col] = pd.to_datetime(self.flight_data[date_col], errors='coerce')

            # Extract month and count bookings
            month_counts = self.flight_data[date_col].dt.month.value_counts()

            # Get the month with the most bookings
            top_month_num = month_counts.index[0]
            top_count = month_counts.iloc[0]

            # Convert month number to name
            month_name = datetime(2000, top_month_num, 1).strftime('%B')

            return (month_name, top_count)
        except Exception as e:
            print(f"Error finding month with most bookings: {e}")
            return ("Unknown", 0)

    def cancellation_patterns(self):
        """
        Analyze patterns in booking cancellations.

        Returns:
            dict: Dictionary with cancellation analysis
        """
        try:
            # Find the cancellation column
            cancellation_col = None
            for col in self.flight_data.columns:
                if 'cancel' in col.lower() or 'status' in col.lower():
                    cancellation_col = col
                    break

            if cancellation_col is None:
                print("Could not find cancellation column.")
                return {
                    "cancellations_by_day": {},
                    "cancellations_by_airline": {},
                    "cancellation_rates": {}
                }

            # Find the date column
            date_col = None
            for col in self.flight_data.columns:
                if 'date' in col.lower() and ('book' in col.lower() or 'depart' in col.lower()):
                    date_col = col
                    break

            # Prepare results
            results = {}

            # Analyze cancellations by day of week (if date column exists)
            if date_col is not None:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(self.flight_data[date_col]):
                    self.flight_data[date_col] = pd.to_datetime(self.flight_data[date_col], errors='coerce')

                # Get day of week and count cancellations
                self.flight_data['day_of_week'] = self.flight_data[date_col].dt.day_name()
                cancellations_by_day = self.flight_data[self.flight_data[cancellation_col] == True][
                    'day_of_week'].value_counts().to_dict()
                results["cancellations_by_day"] = cancellations_by_day
            else:
                results["cancellations_by_day"] = {}

            # Analyze cancellations by airline
            cancellations_by_airline = self.flight_data[self.flight_data[cancellation_col] == True][
                'airline_name'].value_counts().to_dict()
            results["cancellations_by_airline"] = cancellations_by_airline

            # Calculate cancellation rates by airline
            airline_totals = self.flight_data['airline_name'].value_counts().to_dict()
            cancellation_rates = {}
            for airline, total in airline_totals.items():
                cancellations = cancellations_by_airline.get(airline, 0)
                rate = cancellations / total if total > 0 else 0
                cancellation_rates[airline] = round(rate * 100, 2)  # as percentage

            results["cancellation_rates"] = cancellation_rates

            return results
        except Exception as e:
            print(f"Error analyzing cancellation patterns: {e}")
            return {
                "cancellations_by_day": {},
                "cancellations_by_airline": {},
                "cancellation_rates": {}
            }

    def seat_occupancy_analysis(self):
        """
        Analyze seat occupancy patterns.

        Returns:
            dict: Dictionary with seat occupancy analysis
        """
        try:
            # Group by flight number and count passengers
            flight_occupancy = self.flight_data.groupby(['flight_number', 'airline_name']).size().reset_index(
                name='passenger_count')

            # Find flights with highest and lowest occupancy
            most_popular = flight_occupancy.loc[flight_occupancy['passenger_count'].idxmax()]
            least_popular = flight_occupancy.loc[flight_occupancy['passenger_count'].idxmin()]

            return {
                "most_popular": {
                    "flight_number": int(most_popular['flight_number']),
                    "airline": most_popular['airline_name'],
                    "passenger_count": int(most_popular['passenger_count'])
                },
                "least_popular": {
                    "flight_number": int(least_popular['flight_number']),
                    "airline": least_popular['airline_name'],
                    "passenger_count": int(least_popular['passenger_count'])
                }
            }
        except Exception as e:
            print(f"Error analyzing seat occupancy: {e}")
            return {
                "most_popular": {
                    "flight_number": 0,
                    "airline": "Unknown",
                    "passenger_count": 0
                },
                "least_popular": {
                    "flight_number": 0,
                    "airline": "Unknown",
                    "passenger_count": 0
                }
            }

    def fare_analysis_by_class(self):
        """
        Analyze fare patterns by class.

        Returns:
            dict: Dictionary with fare analysis by class
        """
        try:
            # Find the class column
            class_col = None
            for col in self.flight_data.columns:
                if 'class' in col.lower() or 'cabin' in col.lower():
                    class_col = col
                    break

            # Find the fare column
            fare_col = None
            for col in self.flight_data.columns:
                if 'fare' in col.lower() or 'price' in col.lower() or 'cost' in col.lower():
                    fare_col = col
                    break

            if class_col is None or fare_col is None:
                print("Could not find class or fare column.")
                return {}

            # Group by class and calculate statistics
            fare_stats = self.flight_data.groupby(class_col)[fare_col].agg(['mean', 'median', 'min', 'max']).round(
                2).to_dict()

            # Convert to a more readable format
            result = {}
            for class_type in self.flight_data[class_col].unique():
                if pd.notna(class_type):
                    result[str(class_type)] = {
                        "mean_fare": fare_stats['mean'].get(class_type, 0),
                        "median_fare": fare_stats['median'].get(class_type, 0),
                        "min_fare": fare_stats['min'].get(class_type, 0),
                        "max_fare": fare_stats['max'].get(class_type, 0)
                    }

            return result
        except Exception as e:
            print(f"Error analyzing fares by class: {e}")
            return {}

    def day_of_week_analysis(self):
        """
        Analyze booking patterns by day of week.

        Returns:
            dict: Dictionary with booking counts by day of week
        """
        try:
            # Find the date column
            date_col = None
            for col in self.flight_data.columns:
                if 'date' in col.lower() and ('book' in col.lower() or 'depart' in col.lower()):
                    date_col = col
                    break

            if date_col is None:
                print("Could not find date column.")
                return {}

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(self.flight_data[date_col]):
                self.flight_data[date_col] = pd.to_datetime(self.flight_data[date_col], errors='coerce')

            # Get day of week and count bookings
            day_counts = self.flight_data[date_col].dt.day_name().value_counts().to_dict()

            # Sort by day of week
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            sorted_counts = {day: day_counts.get(day, 0) for day in days_order if day in day_counts}

            return sorted_counts
        except Exception as e:
            print(f"Error analyzing bookings by day of week: {e}")
            return {}

    def fare_duration_correlation(self):
        """
        Analyze correlation between fare price and flight duration.

        Returns:
            dict: Dictionary with correlation analysis
        """
        try:
            # Find the fare column
            fare_col = None
            for col in self.flight_data.columns:
                if 'fare' in col.lower() or 'price' in col.lower() or 'cost' in col.lower():
                    fare_col = col
                    break

            # Find the duration column
            duration_col = None
            for col in self.flight_data.columns:
                if 'duration' in col.lower() or 'flight_time' in col.lower() or 'hours' in col.lower():
                    duration_col = col
                    break

            if fare_col is None or duration_col is None:
                print("Could not find fare or duration column.")
                return {
                    "correlation": 0,
                    "interpretation": "Could not calculate correlation due to missing columns."
                }

            # Calculate correlation
            correlation = self.flight_data[[fare_col, duration_col]].corr().iloc[0, 1]

            # Interpret the correlation
            if abs(correlation) < 0.3:
                interpretation = "Weak correlation between fare and duration."
            elif abs(correlation) < 0.7:
                interpretation = "Moderate correlation between fare and duration."
            else:
                interpretation = "Strong correlation between fare and duration."

            if correlation > 0:
                interpretation += " As duration increases, fare tends to increase."
            else:
                interpretation += " As duration increases, fare tends to decrease."

            return {
                "correlation": round(correlation, 2),
                "interpretation": interpretation
            }
        except Exception as e:
            print(f"Error analyzing fare-duration correlation: {e}")
            return {
                "correlation": 0,
                "interpretation": f"Error calculating correlation: {e}"
            }
