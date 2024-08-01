import os
from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from models.nlp.intent_classifier import AdvancedIntentClassifier
from models.knowledge_base.technology_kb import TechnologyKnowledgeBase
from utils.file_templates import get_file_template, render_template

class AdvancedProjectGenerator:
    def __init__(self, config):
        self.config = config
        self.architecture_model = ArchitectureModel(config['architecture_model'])
        self.code_generator = CodeGenerator(config['code_generator'])
        self.intent_classifier = AdvancedIntentClassifier()
        self.kb = TechnologyKnowledgeBase(config['mongodb_uri'], config['db_name'])

    def generate_project(self, project_description, language, framework):
        # Classify intent
        intent, _ = self.intent_classifier.predict([project_description])
        intent = intent[0]

        # Generate architecture
        architecture = self.architecture_model.predict_architecture(project_description)

        # Get best practices and design patterns
        best_practices = self.kb.get_best_practices(language)
        design_patterns = self.kb.get_design_patterns(intent)

        # Generate code
        generated_code = self.code_generator.generate_code(architecture, language, framework, best_practices, design_patterns)

        # Assemble project
        project_structure = self._create_project_structure(architecture, language, framework)
        file_contents = self._distribute_code(generated_code, project_structure, language, framework)

        return self._write_project_files(file_contents)

    def _create_project_structure(self, architecture, language, framework):
        # Implementation details...
        pass

    def _distribute_code(self, generated_code, project_structure, language, framework):
        file_contents = {}
        for file_path, code in generated_code.items():
            template = get_file_template(language, file_path)
            rendered_content = render_template(template, code=code)
            file_contents[file_path] = rendered_content
        return file_contents

    def _write_project_files(self, file_contents):
        for file_path, content in file_contents.items():
            full_path = os.path.join(self.config['output_directory'], file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
        return f"Project files written successfully to {self.config['output_directory']}"