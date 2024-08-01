import os
import logging
from typing import Dict, Any
from pyparsing import List
import tensorflow as tf
import torch
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from utils.model_utils import load_pretrained_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all messages except errors
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow Python API warnings

# Load local dataset
dataset = load_dataset('csv', data_files='data/embeddings/Data.csv')

print(dataset)

class ArchitectureModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model, self.tokenizer = self._build_model()

    def _build_model(self):
        try:
            base_model_name = self.config.get('base_model', 'default_model')
            self.logger.info(f"Building model with base model: {base_model_name}")
            
            if base_model_name == 'bert-base-uncased':
                model = BertModel.from_pretrained(base_model_name)
                tokenizer = BertTokenizer.from_pretrained(base_model_name)
                return model, tokenizer
            else:
                base_model = load_pretrained_model(base_model_name, trainable=False)
                
                if base_model_name == 'default_model':
                    input_shape = (224, 224, 3)  # Default input shape
                # Add more model-specific logic here if needed
                
                return base_model, None  # Assuming no tokenizer for non-BERT models
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise

    def fine_tune(self, dataset_name: str, output_dir: str, num_train_epochs: int = 3, batch_size: int = 8):
        if isinstance(self.model, BertModel):
            # Fine-tune BERT model
            if not self.tokenizer:
                self.logger.error("Tokenizer is not available for fine-tuning.")
                return
    
            # Load dataset
            dataset = load_dataset(dataset_name)
    
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(examples['text'], padding="max_length", truncation=True)
    
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
            # Add attention mask
            def add_attention_mask(examples):
                examples['attention_mask'] = [[1 if token != self.tokenizer.pad_token_id else 0 for token in seq] for seq in examples['input_ids']]
                return examples
    
            tokenized_datasets = tokenized_datasets.map(add_attention_mask, batched=True)
    
            # Set training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f'{output_dir}/logs',
            )
    
            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['test']
            )
    
            # Train model
            trainer.train()
        else:
            # Fine-tune TensorFlow model
            try:
                # Load and preprocess dataset
                dataset = load_dataset(dataset_name)
                train_data = dataset['train']
                test_data = dataset['test']
    
                # Preprocess data for TensorFlow model
                def preprocess_data(data):
                    texts = data['text']
                    labels = data['label']
                    return texts, labels
    
                train_texts, train_labels = preprocess_data(train_data)
                test_texts, test_labels = preprocess_data(test_data)
    
                # Tokenize data
                train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
                test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)
    
                # Convert to TensorFlow dataset
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(train_encodings),
                    train_labels
                )).batch(batch_size)
    
                test_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(test_encodings),
                    test_labels
                )).batch(batch_size)
    
                # Compile and train the model
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                   metrics=['accuracy'])
    
                self.model.fit(train_dataset, epochs=num_train_epochs, validation_data=test_dataset)
            except Exception as e:
                self.logger.error(f"Error during fine-tuning: {e}")
                raise

        # Example usage:
        config = {'base_model': 'bert-base-uncased'}
        arch_model = ArchitectureModel(config)
        arch_model.fine_tune('csv', './results')



    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """
        Evaluate the model on the test data and calculate evaluation metrics.

        Args:
            test_data (Any): The test data to evaluate the model on.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation metrics.
        """
        # Assuming the test data is in the format [(input, label), ...]
        # where input is the input data and label is the ground truth label.
        # Implement your evaluation logic here
        # For example, you can use the model to make predictions on the test data
        # and then calculate evaluation metrics like accuracy, precision, recall, etc.
        
        self.logger.info("Evaluating the model...")
        
        # Calculate evaluation metrics using the model and test data
        # This assumes the model has a predict() method that takes input data and returns the predicted label
        # and a metrics() method that calculates accuracy, precision, recall, and f1_score
        y_true = [label for _, label in test_data]
        y_pred = [self.model.predict(input_data) for input_data, _ in test_data]
        metrics = self.model.metrics(y_true, y_pred)
        
    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        # Implement your evaluation logic here
        # For example, you can use the model to make predictions on the test data
        # and then calculate evaluation metrics like accuracy, precision, recall, etc.
        self.logger.info("Evaluating the model...")
        
        # Placeholder for evaluation logic
        # Replace with actual evaluation code
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        return metrics

    def predict_architecture(self, project_description: str) -> List[float]:
        if isinstance(self.model, BertModel):
            return self._predict_with_bert(project_description)
        else:
            return self._predict_with_tf(project_description)

    def _predict_with_bert(self, project_description: str) -> List[float]:
        inputs = self.tokenizer(project_description, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        # You might need to add a classification layer here
        # For now, we'll just return the mean of the pooled output as a placeholder
        return pooled_output.mean(dim=1).tolist()[0]

    @tf.function(experimental_relax_shapes=True)
    def _predict_with_tf(self, project_description: str) -> List[float]:
        processed_input = self._preprocess_input(project_description)
        architecture_prediction = self.model(processed_input)
        return self._postprocess_output(architecture_prediction)

    @tf.function
    def _preprocess_input(self, project_description: str) -> tf.Tensor:
        processed_input = tf.strings.lower(project_description)
        processed_input = tf.strings.regex_replace(processed_input, r'\W+', ' ')
        processed_input = tf.strings.strip(processed_input)
        processed_input = tf.strings.split(processed_input)
        processed_input = tf.strings.reduce_join(processed_input)
        processed_input = tf.strings.unicode_decode(processed_input, 'UTF-8')
        processed_input = tf.strings.to_hash_bucket_fast(processed_input, 1024)
        return tf.expand_dims(processed_input, 0)  # Add batch dimension

    def _postprocess_output(self, architecture_prediction: tf.Tensor) -> List[float]:
        return tf.nn.softmax(architecture_prediction).numpy().tolist()[0]

    def fine_tune(self, new_data: tf.data.Dataset) -> None:
        try:
            if isinstance(self.model, BertModel):
                self.logger.info("Fine-tuning not implemented for BERT model")
                return
            epochs = self.config.get('fine_tune_epochs', 5)
            self.logger.info(f"Fine-tuning model for {epochs} epochs")
            self.model.fit(new_data, epochs=epochs)
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            raise

    def save_model(self, filepath: str) -> None:
        try:
            if isinstance(self.model, BertModel):
                self.model.save_pretrained(filepath)
                self.tokenizer.save_pretrained(filepath)
            else:
                self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'ArchitectureModel':
        try:
            instance = cls(config)
            if config['base_model'] == 'bert-base-uncased':
                instance.model = BertModel.from_pretrained(filepath)
                instance.tokenizer = BertTokenizer.from_pretrained(filepath)
            else:
                instance.model = tf.keras.models.load_model(filepath)
            instance.logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            instance.logger.error(f"Error loading model: {e}")
            raise