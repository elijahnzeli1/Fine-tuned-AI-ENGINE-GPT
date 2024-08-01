import tensorflow as tf
# from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from typing import Optional, Any
from transformers import BertModel, GPT2Model, DistilBertModel
import spacy

def load_pretrained_model(model_name: str, trainable: bool = False) -> tf.keras.Model:
    """
    Load a pretrained model or create a simple default model if not found.
    
    Args:
        model_name (str): Name of the pretrained model to load.
        trainable (bool): Whether the loaded model should be trainable.
    
    Returns:
        tf.keras.Model: The loaded or created model.
    """

    model_name = model_name.lower()

    if model_name == 'bert-base-uncased':
        model = BertModel.from_pretrained(model_name)
    elif model_name == 'gpt2-medium':
        model = GPT2Model.from_pretrained(model_name)
    elif model_name == 'distilbert-base-uncased':
        model = DistilBertModel.from_pretrained(model_name)
    elif model_name == 'en_core_web_sm':
        model = spacy.load(model_name)
    # Include the other models from the previous version
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if hasattr(model, 'trainable'):
        model.trainable = trainable
    return model


def save_model(model: tf.keras.Model, filepath: str) -> None:
    """
    Save a TensorFlow model to the specified filepath.

    Args:
        model (tf.keras.Model): The model to save.
        filepath (str): The path where the model should be saved.
    """
    model.save(filepath)

def load_model(filepath: str) -> tf.keras.Model:
    """
    Load a TensorFlow model from the specified filepath.

    Args:
        filepath (str): The path from which to load the model.

    Returns:
        tf.keras.Model: The loaded model.
    """
    return tf.keras.models.load_model(filepath)

def compile_model(model: tf.keras.Model, optimizer: Any, loss: Any, metrics: list) -> None:
    """
    Compile a TensorFlow model with the specified optimizer, loss, and metrics.

    Args:
        model (tf.keras.Model): The model to compile.
        optimizer (Any): The optimizer to use.
        loss (Any): The loss function to use.
        metrics (list): A list of metrics to track during training.
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)