from typing import Dict, Any
from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from models.knowledge_base.technology_kb import TechnologyKnowledgeBase
from models.nlp.intent_classifier import AdvancedIntentClassifier
from models.nlp.entity_extractor import EntityExtractor

class ProjectGenerationPipeline:
    def __init__(
        self,
        architecture_model: ArchitectureModel,
        code_generator: CodeGenerator,
        knowledge_base: TechnologyKnowledgeBase,
        intent_classifier: AdvancedIntentClassifier,
        entity_extractor: EntityExtractor
    ):
        self.architecture_model = architecture_model
        self.code_generator = code_generator
        self.knowledge_base = knowledge_base
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor

    def generate(self, description: str, language: str, framework: str = None) -> Dict[str, Any]:
        """
        Generates a complete project based on the given description, language, and framework.

        Args:
            description (str): The project description provided by the user.
            language (str): The programming language to use for the project.
            framework (str, optional): The framework to use for the project, if any.

        Returns:
            Dict[str, Any]: A dictionary containing the generated project details.
        """
        # Step 1: Analyze the project description
        intent = self.intent_classifier.predict(description)
        entities = self.entity_extractor.extract_entities(description)

        # Step 2: Get technology recommendations
        tech_stack = self.knowledge_base.suggest_technologies(description, language, framework)

        # Step 3: Generate project architecture
        architecture = self.architecture_model.predict_architecture(
            description, intent, entities, tech_stack
        )

        # Step 4: Get best practices
        best_practices = self.knowledge_base.get_best_practices(language, framework)

        # Step 5: Generate code
        code = self.code_generator.generate_code(
            architecture, language, framework, best_practices
        )

        # Step 6: Prepare project structure
        project_structure = self.prepare_project_structure(architecture, code)

        # Step 7: Generate documentation
        documentation = self.generate_documentation(
            description, architecture, tech_stack, best_practices
        )

        return {
            "intent": intent,
            "entities": entities,
            "technology_stack": tech_stack,
            "architecture": architecture,
            "best_practices": best_practices,
            "code": code,
            "project_structure": project_structure,
            "documentation": documentation
        }

    def prepare_project_structure(self, architecture: Dict[str, Any], code: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepares the project structure based on the architecture and generated code.

        Args:
            architecture (Dict[str, Any]): The project architecture.
            code (Dict[str, str]): The generated code for each component.

        Returns:
            Dict[str, Any]: A dictionary representing the project structure.
        """
        project_structure = {}
        for component, details in architecture.items():
            if component in code:
                project_structure[component] = {
                    "path": details.get("path", f"{component.lower()}.py"),
                    "code": code[component]
                }
        return project_structure

    def generate_documentation(
        self,
        description: str,
        architecture: Dict[str, Any],
        tech_stack: Dict[str, Any],
        best_practices: Dict[str, Any]
    ) -> str:
        """
        Generates project documentation based on the project details.

        Args:
            description (str): The original project description.
            architecture (Dict[str, Any]): The project architecture.
            tech_stack (Dict[str, Any]): The technology stack used in the project.
            best_practices (Dict[str, Any]): The best practices applied to the project.

        Returns:
            str: The generated documentation as a string.
        """
        doc = f"# Project Documentation\n\n## Description\n{description}\n\n"
        doc += "## Architecture\n"
        for component, details in architecture.items():
            doc += f"- {component}: {details.get('description', 'No description available')}\n"
        
        doc += "\n## Technology Stack\n"
        for tech, details in tech_stack.items():
            doc += f"- {tech}: {details.get('version', 'Version not specified')}\n"
        
        doc += "\n## Best Practices\n"
        for practice, details in best_practices.items():
            doc += f"- {practice}: {details}\n"
        
        return doc