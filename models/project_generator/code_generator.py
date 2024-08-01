import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Any, Dict, List

class CodeGenerator:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_name = self.config.get('model_name', 'gpt2')
        self.max_length = self.config.get('max_length', 1000)
        
        self.logger.info(f"Initializing CodeGenerator with model: {self.model_name}")
        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            
            # Set pad_token to eos_token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_code(self, architecture: str, language: str) -> str:
        """
        Generate code based on the architecture and language.

        Args:
            architecture (str): Description of the architecture.
            language (str): Programming language.

        Returns:
            str: Generated code.
        """
        # Tokenize the input
        input_ids = self.tokenizer.encode(
            f"Generate code for {architecture} architecture in {language} language.",
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )

        # Generate code
        generated_ids = self.model.generate(
            input_ids,
            do_sample=True,
            top_p=0.95,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode the generated code
        generated_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        return generated_code

    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        self.logger.info("Evaluating the model...")
        
        total_samples = len(test_data)
        correct_predictions = 0
        
        for sample in test_data:
            input_text = sample['input']
            expected_output = sample['output']
            
            encoding = self.tokenizer(
                input_text, 
                return_tensors='pt', 
                max_length=self.max_length, 
                truncation=True, 
                padding='max_length'
            )
            inputs = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            max_new_tokens = max(1, self.max_length - inputs.shape[-1])
            
            outputs = self.model.generate(
                inputs, 
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True  # Enable sampling
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Placeholder for actual evaluation logic
            if generated_text == expected_output:  # This is just a placeholder condition
                correct_predictions += 1
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": 0.0,  # Placeholder
            "recall": 0.0,     # Placeholder
            "f1_score": 0.0    # Placeholder
        }
        
        return metrics

    def _create_prompt(self, architecture, language):
        if not architecture or not language:
            raise ValueError("Both 'architecture' and 'language' must be provided.")
        
        prompt = f"Generate code for {architecture} architecture in {language} language."
        return prompt

    def _postprocess_code(self, generated_code):
        if not generated_code:
            return generated_code
        
        cleaned_code = generated_code.strip()
        return cleaned_code
        
        # Remove redundant whitespace
        # whitespace_table = str.maketrans({'\n': ' ', '\t': ' '})
        # cleaned_code = ' '.join(cleaned_code.translate(whitespace_table).split())
        
        # return cleaned_code

    def fine_tune(self, new_data):
        if not new_data:
            raise ValueError("New data must be provided for fine-tuning.")
        
        self.logger.info("Starting fine-tuning process")
        try:
            # Placeholder for fine-tuning logic
            self.model.train(new_data)
            self.logger.info("Fine-tuning completed successfully")
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            raise
        
        self.logger.info("Fine-tuning process completed")