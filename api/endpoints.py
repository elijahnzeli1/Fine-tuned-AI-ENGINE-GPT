from flask import Flask, request, jsonify
from models.project_generator.architecture_model import ArchitectureModel
from models.project_generator.code_generator import CodeGenerator
from models.nlp.intent_classifier import AdvancedIntentClassifier
from models.nlp.entity_extractor import EntityExtractor
from models.knowledge_base.technology_kb import TechnologyKnowledgeBase
from models.feedback.feedback_system import FeedbackSystem  # Updated import
from utils.data_loader import load_config
from utils.project_pipeline import ProjectGenerationPipeline

app = Flask(__name__)

# Load configurations
config = load_config('config/model_config.yaml')

# Initialize models and components
architecture_model = ArchitectureModel(config['architecture_model'])
code_generator = CodeGenerator(config['code_generator'])
intent_classifier = AdvancedIntentClassifier()
entity_extractor = EntityExtractor()
knowledge_base = TechnologyKnowledgeBase()
feedback_system = FeedbackSystem(config['mongodb_uri'], config['db_name'])  # Updated initialization

# Initialize project generation pipeline
project_pipeline = ProjectGenerationPipeline(
    architecture_model,
    code_generator,
    knowledge_base,
    intent_classifier,
    entity_extractor
)

@app.route('/generate_project', methods=['POST'])
def generate_project():
    data = request.json
    project_description = data['description']
    language = data.get('language', 'python')
    framework = data.get('framework')
    
    # Advanced NLP processing
    intent = intent_classifier.predict(project_description)
    entities = entity_extractor.extract_entities(project_description)
    
    # Generate project using the pipeline
    project = project_pipeline.generate(
        description=project_description,
        language=language,
        framework=framework,
        intent=intent,
        entities=entities
    )
    
    response = {
        'intent': intent,
        'entities': entities,
        'architecture': project['architecture'],
        'code': project['code'],
        'best_practices': project['best_practices'],
        'technology_stack': project['technology_stack']
    }
    
    return jsonify(response)

@app.route('/analyze_requirements', methods=['POST'])
def analyze_requirements():
    data = request.json
    requirements = data['requirements']
    
    analysis = {
        'intents': intent_classifier.predict_batch(requirements),
        'entities': entity_extractor.extract_entities_batch(requirements),
        'technology_suggestions': knowledge_base.suggest_technologies(requirements)
    }
    
    return jsonify(analysis)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    feedback = data['feedback']
    project_id = data['project_id']
    
    feedback_system.process(feedback, project_id)
    
    return jsonify({'status': 'Feedback received and processed'})

@app.route('/analyze_feedback', methods=['GET'])
def analyze_feedback():
    analysis = feedback_system.analyze_feedback()
    return jsonify(analysis)

@app.route('/get_best_practices', methods=['GET'])
def get_best_practices():
    language = request.args.get('language')
    framework = request.args.get('framework')
    
    best_practices = knowledge_base.get_best_practices(language, framework)
    
    return jsonify(best_practices)

if __name__ == '__main__':
    app.run(debug=True)