import argparse
from models.fine_tuning.fine_tuning_pipeline import FineTuningPipeline
from utils.data_loader import save_json_data
import logging
import yaml

def load_yaml_data(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    try:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        logger.info("Loading configuration file.")
        config = load_yaml_data(args.config_file)
        
        logger.debug(f"Loaded configuration: {config}")
        
        logger.info("Initializing fine-tuning pipeline.")
        pipeline = FineTuningPipeline(config)  # Pass the entire config
        
        logger.info("Running fine-tuning pipeline.")
        performance = pipeline.run_pipeline(args.data_file)
        
        logger.info("Saving performance metrics.")
        save_json_data(performance, args.output_file)
        
        logger.info(f"Training completed. Performance metrics saved to {args.output_file}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.exception("Exception details:")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fine-tune models using a specified configuration")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file (YAML format)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data file (JSON format)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save performance metrics (JSON format)")
    args = parser.parse_args()
    main(args)