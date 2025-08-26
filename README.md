# Flight Booking Data Analysis with LLM-Powered Agents

This project uses LLM-powered agents to automatically clean, analyze, and model flight booking data. It addresses challenges like inconsistent naming and missing values in the dataset.

## Overview

The system consists of three main components:

1. **FlightDataCleaner**: Cleans and preprocesses the flight booking data
   - Standardizes column names
   - Handles missing values
   - Corrects data types
   - Anomaly detection and correction
   - Merges airline data with flight bookings

2. **FlightDataAnalyzer**: Analyzes the cleaned data to answer business questions
   - Provides methods for common analysis tasks
   - Supports custom analysis through LLM-powered agents

3. **FlightDataInterface**: Provides a simple command-line interface for interacting with the system

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (required):
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your-api-key-here`

## Usage

Run the main script to start the interactive interface:

```bash
python main.py
```

The interface provides the following options:

1. **Ask a natural language question**: Ask a single question and get an answer
2. **Interactive Q&A session**: Enter a continuous dialogue mode where you can ask multiple questions in succession without returning to the main menu
3. **Example questions**: Choose from a list of predefined example questions
4. **Exit**: Close the application

## Data Cleaning Process

The system now uses LLM-powered agents to automatically clean and preprocess the data. This agentic approach eliminates the need for hardcoded cleaning rules and allows the system to adapt to different datasets.

### Agent-Based Cleaning

The data cleaning process is now fully automated using LLM-powered agents:

1. **Automated Column Name Standardization**:
   - The agent analyzes column names to identify typos, abbreviations, and inconsistent naming
   - It suggests standardized names based on the semantic meaning of each column
   - No hardcoded mappings are required

2. **Intelligent Missing Value Handling**:
   - The agent analyzes the data to understand the meaning and distribution of each column
   - It suggests appropriate strategies for handling missing values based on the column's characteristics
   - The agent implements the strategies and explains its reasoning

3. **Smart Data Type Correction**:
   - The agent determines the semantic meaning of each column
   - It suggests the most appropriate data type for each column
   - The agent provides and executes code to convert columns to the appropriate types

4. **Adaptive Data Merging**:
   - The agent analyzes both datasets to understand their structure
   - It identifies the best columns to use as keys for merging
   - It suggests the appropriate join type and handles potential issues

## Analysis Capabilities

The system can answer any question about the flight data using natural language processing. All analyses are performed by LLM-powered agents, which can interpret questions, generate appropriate pandas code, and return results in a user-friendly format.

### Example Questions
- Which airline has the most flights listed?
- What are the top three most frequented destinations?
- How many bookings were there for American Airlines yesterday?
- What is the average flight delay per airline?
- Which month has the highest number of bookings?
- What are the patterns in booking cancellations?
- Which flights have the highest and lowest seat occupancy?
- What is the average fare for business class flights compared to economy class?
- Which day of the week has the most flight bookings?
- Is there a correlation between fare price and flight duration?

The system is not limited to these predefined questions - it can answer any question that can be answered from the data.

## Natural Language Query Processing

The system now supports converting any natural language question into pandas DataFrame queries, similar to how text2sql works but for pandas operations. This allows users to ask questions in plain English without needing to know pandas syntax or the specific structure of the data.

This functionality is powered by LangChain's pandas dataframe agent, which uses OpenAI's language models to interpret natural language questions and convert them to pandas operations.

### How It Works

1. The user enters a natural language question
2. The LLM-powered agent interprets the question
3. The agent generates and executes the appropriate pandas operations
4. The results are returned to the user

### Interactive Q&A Mode

The system now features an interactive Q&A mode that allows users to have a continuous conversation with the data:

1. **Continuous Dialogue**: Ask multiple questions in succession without returning to the main menu
2. **Context Retention**: The system maintains the context of the conversation
3. **Real-time Responses**: Get immediate answers to your questions
4. **Natural Interaction**: Interact with the data as if you were talking to a data analyst

To use the interactive mode:
1. Select option 2 "Interactive Q&A session" from the main menu
2. Ask as many questions as you want about the flight data
3. Type 'exit', 'quit', or 'back' to return to the main menu

You can also try the interactive mode in the demo script by running `python demo.py` and selecting 'y' when prompted to try the interactive Q&A session.

### Example Questions

The system can answer a wide variety of questions, including but not limited to:

- "Which airline has the most flights listed?"
- "What are the top three most frequented destinations?"
- "How many bookings were there for American Airlines yesterday?"
- "What is the average flight delay per airline?"
- "Which month has the highest number of bookings?"
- "What are the patterns in booking cancellations?"
- "Which flights have the highest and lowest seat occupancy?"
- "What is the average fare for business class flights compared to economy class?"
- "Which day of the week has the most flight bookings?"
- "Is there a correlation between fare price and flight duration?"
- "What percentage of flights have wifi available?"
- "Show me the distribution of fares by airline and class"
- "What's the average loyalty points earned per booking by airline?"

### API Key Requirement

The system requires an OpenAI API key to function. This key is used for both data cleaning and analysis operations. Without a valid API key, the system will not be able to process natural language queries or perform any data operations.

## Project Structure

The project has been refactored into a modular structure for better maintainability and separation of concerns:

```
.
├── data/
│   ├── Airline ID to Name.csv    # Mapping between airline IDs and names
│   └── Flight Bookings.csv       # Main flight booking dataset
├── data_cleaner.py               # Flight data cleaning functionality
├── data_analyzer.py              # Flight data analysis functionality
├── interface.py                  # Command-line interface for the system
├── main.py                       # Main entry point
├── demo.py                       # Simple demonstration script
├── test_flight_data.py           # Unit tests for data cleaning and analysis
├── test_anomalies.py             # Tests for anomaly detection functionality
├── requirements.txt              # Project dependencies
├── .env.template                 # Template for environment variables
└── README.md                     # Project documentation
```

### Module Descriptions

- **data_cleaner.py**: Contains the `FlightDataCleaner` class that handles all data cleaning operations including loading data, standardizing column names, handling missing values, correcting data types, detecting anomalies, and merging airline data.

- **data_analyzer.py**: Contains the `FlightDataAnalyzer` class that provides methods for analyzing the cleaned flight data, including both predefined analyses and custom natural language queries.

- **interface.py**: Contains the `FlightDataInterface` class that provides a command-line interface for interacting with the system, including menus for asking questions and viewing example analyses.

- **main.py**: A simple entry point that creates and runs the interface with the appropriate configuration.

- **demo.py**: A demonstration script that shows how to use the system programmatically without the interactive interface.

## Future Improvements

- Add visualization capabilities for better data insights
- Implement more sophisticated data cleaning techniques
- Expand the analysis capabilities with more predefined questions
- Create a web-based interface for easier interaction
