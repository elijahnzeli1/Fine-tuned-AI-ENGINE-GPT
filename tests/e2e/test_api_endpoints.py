import unittest
import json
from api.endpoints import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_generate_project_endpoint(self):
        payload = {
            "description": "Create a React web app for a blog",
            "language": "javascript"
        }
        response = self.app.post('/generate_project', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('intent', data)
        self.assertIn('entities', data)
        self.assertIn('architecture', data)
        self.assertIn('code', data)
        self.assertIn('import React from', data['code'])

    def test_analyze_requirements_endpoint(self):
        payload = {
            "requirements": [
                "The system should allow users to create accounts",
                "Users should be able to post blog articles",
                "The blog should have a commenting system"
            ]
        }
        response = self.app.post('/analyze_requirements', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('intents', data)
        self.assertIn('entities', data)
        self.assertEqual(len(data['intents']), 3)
        self.assertEqual(len(data['entities']), 3)

if __name__ == '__main__':
    unittest.main()