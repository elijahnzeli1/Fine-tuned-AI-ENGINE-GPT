import logging
import os
import json
from datetime import datetime
from typing import Dict, Any

from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from utils.data_loader import load_config, load_json_data
from utils.visualization import plot_training_history

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Experiment:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"experiments/{self.experiment_name}_{self.timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.config = self.load_configurations()
        self.data = self.load_data()
        self.models = self.initialize_models()

    def load_configurations(self) -> Dict[str, Any]:
        logger.info("Loading configurations")
        model_config = load_config('config/model_config.yaml')
        training_config = load_config('config/training_config.yaml')
        return {"model": model_config, "training": training_config}

    def load_data(self) -> Dict[str, Any]:
        logger.info("Loading data")
        train_data = load_json_data(self.config['training']['data']['train_data_path'])
        test_data = load_json_data(self.config['training']['data']['test_data_path'])
        return {"train": train_data, "test": test_data}

    def initialize_models(self) -> Dict[str, Any]:
        logger.info("Initializing models")
        architecture_model = ArchitectureModel(self.config['model']['architecture_model'])
        code_generator = CodeGenerator(self.config['model']['code_generator'])
        return {"architecture_model": architecture_model, "code_generator": code_generator}

    def run(self):
        logger.info(f"Starting experiment: {self.experiment_name}")
        
        results = {}
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}")
            history = model.train(self.data['train'], epochs=self.config['training']['training']['epochs'])
            evaluation = model.evaluate(self.data['test'])
            
            logger.info(f"{model_name} Evaluation: {evaluation}")
            plot_training_history(history, f"{self.experiment_dir}/{model_name}_training_history.png")
            
            results[model_name] = {
                'evaluation': evaluation,
                'history': history.history
            }
        
        self.save_experiment_results(results)

    def save_experiment_results(self, results: Dict[str, Any]):
        logger.info("Saving experiment results")
        results_file = f"{self.experiment_dir}/results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

def run_experiment():
    try:
        experiment = Experiment("001")
        experiment.run()
    except Exception as e:
        logger.exception(f"An error occurred during the experiment: {str(e)}")

if __name__ == "__main__":
    run_experiment()