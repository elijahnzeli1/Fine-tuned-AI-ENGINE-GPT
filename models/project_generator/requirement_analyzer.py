from collections import Counter
from models.nlp.intent_classifier import AdvancedIntentClassifier
from models.nlp.entity_extractor import EntityExtractor
from models.nlp.technology_detector import TechnologyDetector  # Hypothetical new model

class RequirementAnalyzer:
    def __init__(self, config):
        self.intent_classifier = AdvancedIntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.technology_detector = TechnologyDetector()  # Initialize the new model
        self.config = config

    def analyze(self, requirements):
        analysis = {
            'intents': [],
            'entities': [],
            'technologies': [],
            'project_type': None,
            'dependencies': []  # New field to capture dependencies
        }

        for requirement in requirements:
            intent = self.intent_classifier.predict([requirement])[0]
            entities = self.entity_extractor.extract_entities(requirement)
            technologies = self.technology_detector.detect_technologies(requirement)  # Use the new model
            
            analysis['intents'].append(intent)
            analysis['entities'].extend(entities)
            analysis['technologies'].extend(technologies)  # Update with new model's output

        analysis['technologies'] = list(set(analysis['technologies']))  # Remove duplicates
        analysis['project_type'] = self._determine_project_type(analysis['intents'], analysis['technologies'])
        analysis['dependencies'] = self._analyze_dependencies(analysis)  # New method to analyze dependencies

        return analysis

    def _extract_technologies(self, entities):
        technologies = []
        for entity in entities:
            if entity['label'] in ['TECHNOLOGY', 'FRAMEWORK', 'LANGUAGE', 'DATABASE']:
                tech_info = self.tech_kb.get_technology_info(entity['text'].lower())
                if tech_info:
                    technologies.append({
                        'name': entity['text'],
                        'category': tech_info['category'],
                        'version': tech_info.get('latest_version', 'Unknown')
                    })
        
        # Remove duplicates while preserving order
        return list({tech['name']: tech for tech in technologies}.values())

    def _determine_project_type(self, intents, technologies):
        intent_counter = Counter(intents)
        most_common_intent = intent_counter.most_common(1)[0][0]

        tech_categories = [tech['category'] for tech in technologies]
        tech_counter = Counter(tech_categories)
        most_common_tech = tech_counter.most_common(1)[0][0]

        project_type_map = {
            'web_development': ['WEB_FRAMEWORK', 'FRONTEND', 'BACKEND'],
            'mobile_app': ['MOBILE_FRAMEWORK', 'IOS', 'ANDROID'],
            'data_science': ['DATA_PROCESSING', 'MACHINE_LEARNING', 'DATA_VISUALIZATION'],
            'desktop_app': ['DESKTOP_FRAMEWORK', 'GUI'],
            'api': ['API_FRAMEWORK', 'BACKEND'],
            'database': ['DATABASE', 'ORM'],
        }

        for project_type, related_techs in project_type_map.items():
            if most_common_tech in related_techs:
                return project_type

        # If no clear match, use the most common intent
        intent_to_project_type = {
            'CREATE_WEB_APP': 'web_development',
            'CREATE_MOBILE_APP': 'mobile_app',
            'ANALYZE_DATA': 'data_science',
            'CREATE_DESKTOP_APP': 'desktop_app',
            'CREATE_API': 'api',
            'MANAGE_DATABASE': 'database'
        }

        return intent_to_project_type.get(most_common_intent, 'general_software')

    def _analyze_dependencies(self, analysis):
        dependencies = []
        technologies = analysis['technologies']
        
        # Check for framework dependencies
        frameworks = [tech for tech in technologies if tech['category'] in ['WEB_FRAMEWORK', 'MOBILE_FRAMEWORK', 'DESKTOP_FRAMEWORK']]
        if frameworks:
            main_framework = frameworks[0]
            dependencies.append(f"Project depends on {main_framework['name']} framework")

        # Check for database dependencies
        databases = [tech for tech in technologies if tech['category'] == 'DATABASE']
        if databases:
            dependencies.append(f"Project requires {databases[0]['name']} database")

        # Check for language dependencies
        languages = [tech for tech in technologies if tech['category'] == 'LANGUAGE']
        if languages:
            dependencies.append(f"Project will be implemented in {languages[0]['name']}")

        # Check for API dependencies
        if 'API' in analysis['entities']:
            dependencies.append("Project involves API integration")

        # Check for authentication requirements
        if 'AUTHENTICATION' in analysis['entities']:
            dependencies.append("Project requires user authentication system")

        # Check for cloud deployment
        cloud_platforms = [tech for tech in technologies if tech['category'] == 'CLOUD_PLATFORM']
        if cloud_platforms:
            dependencies.append(f"Project will be deployed on {cloud_platforms[0]['name']}")

        # Analyze potential conflicts or compatibility issues
        for i, tech1 in enumerate(technologies):
            for tech2 in technologies[i+1:]:
                compatibility = self.tech_kb.check_compatibility(tech1['name'], tech2['name'])
                if compatibility == 'INCOMPATIBLE':
                    dependencies.append(f"Warning: {tech1['name']} may be incompatible with {tech2['name']}")
                elif compatibility == 'VERSION_CONFLICT':
                    dependencies.append(f"Note: Check version compatibility between {tech1['name']} and {tech2['name']}")

        return dependencies