import spacy
from spacy.matcher import Matcher
from typing import Dict, List, Union, Optional
import logging

class EntityExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logging.error(f"Failed to load model '{model}'. Make sure it's installed.")
            raise

        self.matcher = Matcher(self.nlp.vocab)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from the given text using spaCy's named entity recognition.

        :param text: Input text to extract entities from.
        :return: Dictionary of entity types and their corresponding entities.
        """
        if not text.strip():
            return {}

        doc = self.nlp(text)
        entities: Dict[str, List[str]] = {}

        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)

        return entities

    def add_custom_patterns(self, patterns: Dict[str, List[List[Union[str, Dict]]]]) -> None:
        """
        Add custom patterns to the matcher.

        :param patterns: Dictionary of entity labels and their corresponding patterns.
        """
        for label, pattern_list in patterns.items():
            self.matcher.add(label, pattern_list)

    def extract_custom_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract custom entities from the given text using the added patterns.

        :param text: Input text to extract custom entities from.
        :return: Dictionary of custom entity types and their corresponding entities.
        """
        if not text.strip():
            return {}

        doc = self.nlp(text)
        matches = self.matcher(doc)

        custom_entities: Dict[str, List[str]] = {}
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            custom_entities.setdefault(label, []).append(doc[start:end].text)

        return custom_entities

    def extract_all_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract both named entities and custom entities from the given text.

        :param text: Input text to extract entities from.
        :return: Dictionary of all entity types and their corresponding entities.
        """
        named_entities = self.extract_entities(text)
        custom_entities = self.extract_custom_entities(text)

        all_entities = named_entities.copy()
        for label, entities in custom_entities.items():
            all_entities.setdefault(label, []).extend(entities)

        return all_entities

    def get_entity_spans(self, text: str, entity_type: Optional[str] = None) -> List[Dict[str, Union[str, int]]]:
        """
        Get the spans of entities in the text, optionally filtered by entity type.

        :param text: Input text to extract entity spans from.
        :param entity_type: Optional entity type to filter by.
        :return: List of dictionaries containing entity text, label, start, and end positions.
        """
        doc = self.nlp(text)
        spans = []

        for ent in doc.ents:
            if entity_type is None or ent.label_ == entity_type:
                spans.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        return spans