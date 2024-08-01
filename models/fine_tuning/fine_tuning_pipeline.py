import json
import logging

class FineTuningPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if 'fine_tuning' not in self.config:
            raise KeyError("Configuration is missing 'fine_tuning' section")

    def prepare_data(self, data_path):
        """
        Prepare data for training and testing.

        Args:
            data_path (str): Path to the data file.

        Returns:
            tuple: Prepared training and testing data.
        """
        try:
            self.logger.info(f"Loading data from {data_path}")
            with open(data_path, 'r') as file:
                data = json.load(file)

            # Debug: Print the first few elements of data to understand its structure
            self.logger.debug(f"First few elements of data: {data[:5]}")

            # Ensure data is a list of dictionaries
            if not all(isinstance(item, dict) for item in data):
                raise ValueError("Data should be a list of dictionaries")

            inputs = [item.get('input', None) for item in data]
            labels = [item.get('label', None) for item in data]

            # Split data into training and testing sets
            split_index = int(len(data) * (1 - self.config['fine_tuning']['validation_split']))
            train_data = (inputs[:split_index], labels[:split_index])
            test_data = (inputs[split_index:], labels[split_index:])

            return train_data, test_data
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise

    def run_pipeline(self, data_path):
        """
        Run the fine-tuning pipeline.

        Args:
            data_path (str): Path to the data file.

        Returns:
            dict: Performance metrics.
        """
        try:
            train_data, test_data = self.prepare_data(data_path)

            # Add your training logic here
            self.logger.info("Starting training process.")
            # Example: model.fit(train_data[0], train_data[1])

            # Add your evaluation logic here
            self.logger.info("Starting evaluation process.")
            # Example: performance_metrics = model.evaluate(test_data[0], test_data[1])

            # Placeholder for actual performance metrics
            performance_metrics = {"accuracy": 0.95}  # Example metric

            return performance_metrics
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {e}")
            raise

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Fine-tune models using a specified configuration")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file (YAML format)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data file (JSON format)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save performance metrics (JSON format)")
    args = parser.parse_args()

    try:
        logger.info("Loading configuration file.")
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)

        logger.info("Initializing fine-tuning pipeline.")
        pipeline = FineTuningPipeline(config)

        logger.info("Running fine-tuning pipeline.")
        performance = pipeline.run_pipeline(args.data_file)

        logger.info("Saving performance metrics.")
        with open(args.output_file, 'w') as file:
            json.dump(performance, file)

        logger.info(f"Training completed. Performance metrics saved to {args.output_file}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error("Exception details:", exc_info=True)