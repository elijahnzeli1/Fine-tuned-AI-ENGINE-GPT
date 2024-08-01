from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

class FeedbackSystem:
    def __init__(self, mongodb_uri: str, db_name: str):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]

    def submit_feedback(self, project_id: str, user_id: str, rating: int, comments: str) -> None:
        feedback = {
            "project_id": project_id,
            "user_id": user_id,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now(timezone.utc)
        }
        self.db.feedback.insert_one(feedback)

    def get_project_feedback(self, project_id: str) -> List[Dict[str, Any]]:
        return list(self.db.feedback.find({"project_id": project_id}))

    def get_average_rating(self, project_id: str) -> Optional[float]:
        pipeline = [
            {"$match": {"project_id": project_id}},
            {"$group": {"_id": "$project_id", "avg_rating": {"$avg": "$rating"}}}
        ]
        result = list(self.db.feedback.aggregate(pipeline))
        return result[0]['avg_rating'] if result else None

    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self.db.feedback.find().sort("timestamp", -1).limit(limit))

    def analyze_feedback(self) -> Dict[str, Any]:
        """Analyzes the feedback data and returns a dictionary of analysis results."""
        average_rating = self.db.feedback.aggregate([
            {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
        ]).next()["avg_rating"]

        project_ratings = self.db.feedback.aggregate([
            {"$group": {"_id": "$project_id", "avg_rating": {"$avg": "$rating"}}}
        ])

        recent_comments = self.db.feedback.find().sort("timestamp", -1).limit(10)
        recent_comments = [comment["comments"] for comment in recent_comments]

        comments = [comment["comments"] for comment in self.db.feedback.find()]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(comments)
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        comment_clusters = {i: [] for i in range(10)}
        for comment, label in zip(comments, cluster_labels):
            comment_clusters[label].append(comment)

        common_issues = []
        for cluster in comment_clusters.values():
            words = ' '.join(comment.split() for comment in cluster)
            issue_words = sorted(set(words.split()), key=lambda x: words.count(x), reverse=True)[:3]
            common_issues.extend(issue_words)

        return {
            "average_rating": average_rating,
            "project_ratings": {project["_id"]: project["avg_rating"] for project in project_ratings},
            "recent_comments": recent_comments,
            "common_issues": common_issues
        }

    def process(self, feedback: Dict[str, Any], project_id: str) -> None:
        """Process feedback and store it in the database."""
        self.submit_feedback(
            project_id=project_id,
            user_id=feedback.get('user_id', 'anonymous'),
            rating=feedback.get('rating', 0),
            comments=feedback.get('comments', '')
        )