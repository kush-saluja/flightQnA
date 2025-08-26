"""
Simple demo script for Flight Booking Data Analysis.

This script demonstrates how to use the FlightDataCleaner and FlightDataAnalyzer
classes programmatically in a purely agentic way, using OpenAI for all operations.
"""

import os

from dotenv import load_dotenv

from data_analyzer import FlightDataAnalyzer
from data_cleaner import FlightDataCleaner

load_dotenv()


def main():
    """
    Run a simple demonstration of the flight data analysis capabilities with OpenAI.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found. Please set it in the .env file.")
        print("API key is required for agent-based cleaning and analysis.")
        return

    print("\nStep 1: Setting up agent-based data cleaning...")
    cleaner = FlightDataCleaner("data/Flight Bookings.csv", "data/Airline ID to Name.csv")

    # Load the data first to avoid timeout
    cleaner.load_data()
    print("Data loaded successfully.")

    # Clean the data using the agent-based approach
    print("Cleaning data using agent-based methods...")
    try:
        cleaned_data = cleaner.clean_data(api_key)
        print("Agent-based data cleaning completed successfully.")
    except Exception as e:
        print(f"Error during agent-based data cleaning: {e}")
        print("Exiting the demo as agent-based cleaning is required.")
        return

    # Step 2: Create the analyzer and set up the LLM
    print("\nStep 2: Setting up the analyzer with LLM...")
    analyzer = FlightDataAnalyzer(cleaned_data)
    try:
        analyzer.setup_llm(api_key)
        print("LLM set up successfully for analysis.")
    except Exception as e:
        print(f"Error setting up LLM for analysis: {e}")
        print("Exiting the demo as LLM-based analysis is required.")
        return

    # Step 3: Run a simple natural language query
    print("\nStep 3: Running a natural language query...")
    question = "Which airline has the most flights listed?"
    print(f"Question: {question}")
    try:
        result = analyzer.run_analysis(question)
        print(f"Answer: {result}")
    except Exception as e:
        print(f"Error running analysis: {e}")
        return

    # Step 4: Demonstrate an interactive Q&A session
    print("\nStep 4: Demonstrating interactive Q&A session...")
    print("You can ask multiple questions in succession.")
    print("Type 'exit' to end the session.")

    interactive_mode = input("\nWould you like to try the interactive Q&A session? (y/n): ")
    if interactive_mode.lower() == 'y':
        print("\n===== Interactive Q&A Session =====")
        print("Ask questions about the flight data.")
        print("Type 'exit', 'quit', or 'back' to end the session.")
        print("=====================================")

        while True:
            user_question = input("\nYour question (or 'exit' to end): ")
            if user_question.lower() in ['exit', 'quit', 'back']:
                print("Ending interactive session.")
                break

            print("Processing your question...")
            try:
                answer = analyzer.run_analysis(user_question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error processing question: {e}")

    else:
        print("Skipping interactive Q&A session.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
