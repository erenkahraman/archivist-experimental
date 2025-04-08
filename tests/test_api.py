"""
Tests for the API routes and functionality.
"""
import json
import unittest
from unittest.mock import patch

from src.app import create_app


class TestAPI(unittest.TestCase):
    """Test cases for the API routes."""

    def setUp(self):
        """Set up the test environment."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_index_route(self):
        """Test the index route returns correct status."""
        response = self.client.get('/')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('api_version', data)
        self.assertIn('features', data)

    @patch('src.search_engine.search_engine.search')
    def test_search_route(self, mock_search):
        """Test the search route with mocked search function."""
        mock_search.return_value = {'results': []}
        
        response = self.client.post('/api/search', 
                               json={'query': 'test query'})
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', data)


if __name__ == '__main__':
    unittest.main() 