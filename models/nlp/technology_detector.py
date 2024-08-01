import re
from typing import List, Dict

class TechnologyDetector:
    def __init__(self):
        # Use a dictionary to store technologies with their common variations or aliases
        self.technologies: Dict[str, List[str]] = {
            "Python": ["python", "py"],
            "Java": ["java", "jvm"],
            "JavaScript": ["javascript", "js", "ecmascript"],
            "React": ["react", "reactjs", "react.js"],
            "Angular": ["angular", "angularjs", "angular.js"],
            "Django": ["django"],
            "Flask": ["flask"],
            "Node.js": ["node.js", "nodejs", "node"],
            "Spring Boot": ["spring boot", "springboot"],
            "MySQL": ["mysql", "my sql"],
            "PostgreSQL": ["postgresql", "postgres"],
            "MongoDB": ["mongodb", "mongo"],
            "Docker": ["docker"],
            "Kubernetes": ["kubernetes", "k8s"],
            "AWS": ["aws", "amazon web services"],
            "Azure": ["azure", "microsoft azure"],
            "GCP": ["gcp", "google cloud platform"],
            "TensorFlow": ["tensorflow", "tf"],
            "PyTorch": ["pytorch"],
            "OpenCV": ["opencv"]
        }

    def detect_technologies(self, text: str) -> List[str]:
        """
        Detects technologies mentioned in the given text.

        :param text: The text to search for technologies.
        :return: A list of detected technologies found in the text.
        """
        detected_technologies = set()
        text = text.lower()

        for tech, aliases in self.technologies.items():
            for alias in aliases:
                # Use word boundary regex to avoid partial matches
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text):
                    detected_technologies.add(tech)
                    break  # No need to check other aliases for this technology

        return list(detected_technologies)

    def add_technology(self, technology: str, aliases: List[str]) -> None:
        """
        Adds a new technology to the detector.

        :param technology: The main name of the technology.
        :param aliases: A list of aliases or variations for the technology.
        """
        self.technologies[technology] = [alias.lower() for alias in aliases]

    def remove_technology(self, technology: str) -> None:
        """
        Removes a technology from the detector.

        :param technology: The main name of the technology to remove.
        """
        self.technologies.pop(technology, None)

    def get_all_technologies(self) -> List[str]:
        """
        Returns a list of all technologies currently in the detector.

        :return: A list of technology names.
        """
        return list(self.technologies.keys())