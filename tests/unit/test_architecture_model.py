import unittest
from models.project_generator.architecture_model import ArchitectureModel
from utils.data_loader import load_config

class TestArchitectureModel(unittest.TestCase):
    def setUp(self):
        config = load_config('config/model_config.yaml')
        self.model = ArchitectureModel(config['architecture_model'])

    def test_predict_architecture(self):
        project_description = "Create a mobile app for tracking daily expenses"
        architecture = self.model.predict_architecture(project_description)
        self.assertIsNotNone(architecture)
        self.assertTrue(isinstance(architecture, dict))
        self.assertIn('components', architecture)

    def test_preprocess_input(self):
        project_description = "Build a web application for online shopping"
        processed_input = self.model._preprocess_input(project_description)
        self.assertIsNotNone(processed_input)
        # Add more specific assertions based on your preprocessing logic

    def test_postprocess_output(self):
        mock_prediction = [0.1, 0.2, 0.7]  # Example prediction
        postprocessed_output = self.model._postprocess_output(mock_prediction)
        self.assertIsNotNone(postprocessed_output)
        # Add more specific assertions based on your postprocessing logic

if __name__ == '__main__':
    unittest.main()