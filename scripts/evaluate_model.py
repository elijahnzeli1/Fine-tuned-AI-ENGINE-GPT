import argparse
import json
from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from utils.data_loader import load_config, load_json_data

def evaluate_models(config_path, test_data_path, output_path):
    config = load_config(config_path)
    test_data = load_json_data(test_data_path)

    architecture_model = ArchitectureModel(config['architecture_model'])
    code_generator = CodeGenerator(config['code_generator'])

    arch_metrics = architecture_model.evaluate(test_data)
    code_metrics = code_generator.evaluate(test_data)

    evaluation_results = {
        'architecture_model': arch_metrics,
        'code_generator': code_metrics
    }

    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data file")
    parser.add_argument("--output", type=str, required=True, help="Path to save evaluation results")
    args = parser.parse_args()

    evaluate_models(args.config, args.test_data, args.output)