import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedIntentClassifier:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

    def predict(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy()
    
    def fine_tune(self, train_texts: List[str], train_labels: List[int], 
                  val_texts: List[str] = None, val_labels: List[int] = None,
                  epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
                  warmup_steps: int = 0, weight_decay: float = 0.01) -> Dict[str, List[float]]:
        
        if val_texts is None or val_labels is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.1, random_state=42
            )
        
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch in train_dataloader:
                self.model.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            history['train_loss'].append(avg_train_loss)
            
            val_loss, val_accuracy = self._evaluate(val_dataloader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return history
    
    def _create_dataset(self, texts: List[str], labels: List[int] = None) -> TensorDataset:
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
        if labels is not None:
            return TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
        return TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        
        for batch in dataloader:
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_eval_accuracy += (preds == labels).sum().item()
        
        avg_val_loss = total_eval_loss / len(dataloader)
        avg_val_accuracy = total_eval_accuracy / len(dataloader.dataset)
        
        return avg_val_loss, avg_val_accuracy
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")