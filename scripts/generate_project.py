import os

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow after setting environment variables
import tensorflow as tf

import argparse
from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from models.nlp.intent_classifier import AdvancedIntentClassifier
from utils.data_loader import load_config

def main(args):
    # Load configurations
    arch_config = load_config('config/model_config.yaml')['architecture_model']
    code_config = load_config('config/model_config.yaml')['code_generator']

    # Initialize models
    architecture_model = ArchitectureModel(arch_config)
    code_generator = CodeGenerator(code_config)
    intent_classifier = AdvancedIntentClassifier()

    # Process user input
    project_description = args.description
    language = args.language

    # Classify intent
    intent = intent_classifier.predict([project_description])[0]

    # Generate architecture
    architecture = architecture_model.predict_architecture(project_description)

    # Generate code
    code = code_generator.generate_code(architecture, language)

    # Output results
    print(f"Project Type: {intent}")
    print(f"Generated Architecture: {architecture}")
    print(f"Generated Code:\n{code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a project based on description")
    parser.add_argument("--description", type=str, help="Project description")
    parser.add_argument("--language", type=str, default="python", help="Programming language")
    args = parser.parse_args()
    main(args)