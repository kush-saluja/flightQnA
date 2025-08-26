"""
Flight Data Interface Module

This module contains the FlightDataInterface class, which provides a command-line
interface for interacting with the flight data analysis system.
"""

import os
from dotenv import load_dotenv
from data_cleaner import FlightDataCleaner
from data_analyzer import FlightDataAnalyzer

# Load environment variables from .env file
load_dotenv()


class FlightDataInterface:
    """
    A class to provide a command-line interface for the flight data analysis system.
    """

    def __init__(self):
        """
        Initialize the FlightDataInterface.
        """
        self.cleaner = None
        self.analyzer = None
        self.cleaned_data = None

    def setup(self, flight_data_path=None, airline_mapping_path=None):
        """
        Set up the data cleaner and analyzer.

        Args:
            flight_data_path (str, optional): Path to the flight bookings CSV file.
                Defaults to "data/Flight Bookings.csv".
            airline_mapping_path (str, optional): Path to the airline ID to name mapping CSV file.
                Defaults to "data/Airline ID to Name.csv".
        """
        # Use default paths if not provided
        if flight_data_path is None:
            flight_data_path = os.environ.get("FLIGHT_DATA_PATH", "data/Flight Bookings.csv")
        if airline_mapping_path is None:
            airline_mapping_path = os.environ.get("AIRLINE_MAPPING_PATH", "data/Airline ID to Name.csv")

        # Get API key for LLM-powered operations from environment variable
        print("Setting up LLM-powered data cleaning and analysis...")
        api_key = os.environ.get("OPENAI_API_KEY")

        # Fall back to manual input if environment variable is not set
        if not api_key:
            print("No OpenAI API key found in environment variables.")
            print("Please set the OPENAI_API_KEY environment variable or enter it below:")
            api_key = input("Enter your OpenAI API key: ")

        if not api_key.strip():
            print("API key is required for agent-based cleaning and analysis.")
            return False

        # Create and set up the data cleaner
        self.cleaner = FlightDataCleaner(flight_data_path, airline_mapping_path)

        try:
            # Clean the data using the agent-based approach
            self.cleaned_data = self.cleaner.clean_data(api_key)
            print("Agent-based data cleaning completed successfully.")
        except Exception as e:
            print(f"Error during agent-based data cleaning: {e}")
            print("Exiting setup as agent-based cleaning is required.")
            return False

        # Create and set up the analyzer
        self.analyzer = FlightDataAnalyzer(self.cleaned_data)
        try:
            self.analyzer.setup_llm(api_key)
            print("LLM set up successfully for analysis.")
        except Exception as e:
            print(f"Error setting up LLM for analysis: {e}")
            print("Exiting setup as LLM-based analysis is required.")
            return False

        print("Setup completed successfully.")
        return True

    def display_menu(self):
        """
        Display the main menu.
        """
        print("\n===== Flight Data Analysis Interface =====")
        print("1. Ask a natural language question")
        print("2. Interactive Q&A session")
        print("3. Example questions")
        print("4. Exit")
        print("=====================================")

    def _get_default_questions(self):
        """
        Return a dictionary of default example questions.

        Returns:
            dict: A dictionary of default example questions.
        """
        return {
            '1': "Which airline has the most flights listed?",
            '2': "What are the top three most frequented destinations?",
            '3': "How many bookings were there for American Airlines yesterday?",
            '4': "What is the average flight delay per airline?",
            '5': "Which month has the highest number of bookings?",
            '6': "What are the patterns in booking cancellations?",
            '7': "Which flights have the highest and lowest seat occupancy?",
            '8': "What is the average fare for business class flights?",
            '9': "Which day of the week has the most flight bookings?",
            '10': "Is there a correlation between fare price and flight duration?"
        }

    def display_example_questions(self, questions_dict=None):
        """
        Display example questions that can be asked.

        Args:
            questions_dict (dict, optional): Dictionary of questions to display.
                If None, uses the default questions.
        """
        if questions_dict is None:
            questions_dict = self._get_default_questions()

        print("\n===== Example Questions =====")
        for key in sorted(questions_dict.keys(), key=lambda x: int(x)):
            print(f"{key}. {questions_dict[key]}")
        print(f"{len(questions_dict) + 1}. Back to main menu")
        print("===============================")

    def run(self, flight_data_path=None, airline_mapping_path=None):
        """
        Run the interface.

        Args:
            flight_data_path (str, optional): Path to the flight bookings CSV file.
                Defaults to None, which will use the default path.
            airline_mapping_path (str, optional): Path to the airline ID to name mapping CSV file.
                Defaults to None, which will use the default path.
        """
        if not self.setup(flight_data_path, airline_mapping_path):
            print("\nSetup failed. Exiting the program.")
            return

        # Generate predefined questions dynamically using the LLM agent
        print("\nGenerating example questions based on the dataset...")
        try:
            example_questions_prompt = """
            Based on the flight booking dataset, generate 10 interesting analytical questions that could be answered using this data.
            Focus on questions that would provide valuable business insights.
            Return the questions as a Python dictionary with keys '1' through '10' and the questions as values.
            Example format: {'1': 'Which airline has the most flights?', '2': 'What is the average fare?', ...}
            """

            # Use the analyzer's agent to generate questions
            example_questions_response = self.analyzer.run_analysis(example_questions_prompt)

            # Extract the dictionary from the response
            if "```python" in example_questions_response:
                example_questions_str = example_questions_response.split("```python")[1].split("```")[0].strip()
            elif "```" in example_questions_response:
                example_questions_str = example_questions_response.split("```")[1].split("```")[0].strip()
            else:
                example_questions_str = example_questions_response.strip()

            # Try to evaluate the string as a dictionary
            try:
                predefined_questions = eval(example_questions_str)
                print("Successfully generated example questions based on the dataset.")
            except:
                # Fall back to default questions if evaluation fails
                print("Could not parse generated questions. Using default questions instead.")
                predefined_questions = self._get_default_questions()
        except Exception as e:
            print(f"Error generating example questions: {e}")
            print("Using default questions instead.")
            predefined_questions = self._get_default_questions()

        while True:
            self.display_menu()
            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                # Single question
                question = input("\nEnter your question: ")
                print("Processing your question...")
                result = self.analyzer.run_analysis(question)
                print(f"\nAnalysis Result:\n{result}")

            elif choice == '2':
                # Interactive Q&A session
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
                    result = self.analyzer.run_analysis(user_question)
                    print(f"\nAnswer: {result}")

            elif choice == '3':
                # Example questions
                while True:
                    self.display_example_questions(predefined_questions)
                    back_option = str(len(predefined_questions) + 1)
                    example_choice = input(f"Select an example question (1-{back_option}): ")

                    if example_choice == back_option:
                        # Back to main menu
                        break

                    if example_choice in predefined_questions:
                        question = predefined_questions[example_choice]
                        print(f"\nQuestion: {question}")
                        print("Processing your question...")
                        result = self.analyzer.run_analysis(question)
                        print(f"\nAnalysis Result:\n{result}")
                    else:
                        print(f"\nInvalid choice. Please enter a number between 1 and {back_option}.")

                    input("\nPress Enter to continue...")

            elif choice == '4':
                # Exit
                print("\nExiting the Flight Data Analysis Interface. Goodbye!")
                break

            else:
                print("\nInvalid choice. Please enter a number between 1 and 4.")

            input("\nPress Enter to continue...")
