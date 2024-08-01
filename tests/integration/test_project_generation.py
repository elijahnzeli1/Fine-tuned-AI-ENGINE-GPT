import unittest
from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from utils.data_loader import load_config

class TestProjectGeneration(unittest.TestCase):
    def setUp(self):
        config = load_config('config/model_config.yaml')
        self.architecture_model = ArchitectureModel(config['architecture_model'])
        self.code_generator = CodeGenerator(config['code_generator'])

    def test_end_to_end_generation(self):
        project_description = "Create a Flask API for a todo list application"
        language = "python"

        # Generate architecture
        architecture = self.architecture_model.predict_architecture(project_description)
        self.assertIsNotNone(architecture)

        # Generate code
        code = self.code_generator.generate_code(architecture, language)
        self.assertIsNotNone(code)
        self.assertIn('from flask import Flask', code)
        self.assertIn('def create_app()', code)
        self.assertIn('class TodoItem(db.Model):', code)

    def test_multi_language_support(self):
        project_description = "Build a simple calculator app"
        languages = ["python", "javascript", "java"]

        for language in languages:
            architecture = self.architecture_model.predict_architecture(project_description)
            code = self.code_generator.generate_code(architecture, language)
            self.assertIsNotNone(code)
            if language == "python":
                self.assertIn('def calculate(', code)
            elif language == "javascript":
                self.assertIn('function calculate(', code)
            elif language == "java":
                self.assertIn('public class Calculator', code)

if __name__ == '__main__':
    unittest.main()