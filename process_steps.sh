set TF_ENABLE_ONEDNN_OPTS=0

python scripts/train_model.py --config_file config/training_config.yaml --data_file data/processed/train_data.json --output_file data/performance_metrics.json
python scripts/evaluate_model.py --config config/model_config.yaml --test_data data/processed/test_data.json --output data/output/output.json

python scripts/generate_project.py --description "Develop an AI-driven personal assistant that can handle scheduling, reminders, personal finance management, and even wellness tracking."
python generate_project.py "Develop an AI-driven personal assistant that can handle scheduling, reminders, personal finance management, and even wellness tracking." --language python

or

$env:PYTHONPATH = "C:\Users\KONZA-VDI\ai-engine"
python scripts/generate_project.py --description "Develop an AI-driven personal assistant that can handle scheduling, reminders, personal finance management, and even wellness tracking." --language python