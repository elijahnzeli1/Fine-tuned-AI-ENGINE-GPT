import argparse
import json
from sklearn.model_selection import train_test_split
from utils.data_loader import load_json_data, save_json_data
from typing import List, Dict
import nltk

# Download the punkt tokenizer
nltk.download('punkt', quiet=True)

def preprocess_project_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Preprocesses the project data by creating input-output pairs.
    
    Args:
        data (List[Dict[str, str]]): The raw project data, where each dictionary contains a 'description' key.
    
    Returns:
        List[Dict[str, str]]: The preprocessed project data, each item containing 'input' and 'output' keys.
    """
    processed_data = []
    for item in data:
        description = item['description']
        # Extract the first sentence as the input (question)
        first_sentence = description.split('.')[0]
        processed_item = {
            "input": f"What is {first_sentence}?",
            "output": description
        }
        processed_data.append(processed_item)
    return processed_data

def main(args):
    """
    Main function to preprocess data, split into train and test sets, and save the processed data.
    
    Args:
        args: Command line arguments including input_file and output_dir.
    """
    # Load raw data
    raw_data = load_json_data(args.input_file)
    
    # Check if raw_data is in expected format
    if not isinstance(raw_data, list) or not all(isinstance(item, dict) and 'description' in item for item in raw_data):
        raise ValueError("Input data must be a list of dictionaries with a 'description' field.")

    # Preprocess data
    processed_data = preprocess_project_data(raw_data)

    # Split into train and test sets
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

    # Save processed data
    save_json_data(train_data, f"{args.output_dir}/train_data.json")
    save_json_data(test_data, f"{args.output_dir}/test_data.json")

    print(f"Processed data saved to {args.output_dir}/train_data.json and {args.output_dir}/test_data.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess project data")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the processed data")
    args = parser.parse_args()

    main(args)