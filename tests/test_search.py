"""
Tests for search functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import json

from src.search.elasticsearch_client import ElasticsearchClient


class TestElasticsearchClient(unittest.TestCase):
    """Test cases for the ElasticsearchClient."""

    def setUp(self):
        """Set up the test environment."""
        # Create a mock Elasticsearch client
        with patch('elasticsearch.Elasticsearch'):
            self.es_client = ElasticsearchClient(hosts=['http://localhost:9200'])
    
    @patch('elasticsearch.Elasticsearch.indices.exists')
    def test_index_exists(self, mock_exists):
        """Test checking if index exists."""
        mock_exists.return_value = True
        
        result = self.es_client.index_exists()
        self.assertTrue(result)
        
        mock_exists.return_value = False
        result = self.es_client.index_exists()
        self.assertFalse(result)
    
    @patch('elasticsearch.Elasticsearch.search')
    def test_search(self, mock_search):
        """Test searching for documents."""
        # Setup mock return value
        mock_search.return_value = {
            'hits': {
                'total': {'value': 1},
                'hits': [{
                    '_id': '1',
                    '_source': {'filename': 'test.jpg', 'colors': [255, 0, 0]}
                }]
            }
        }
        
        results = self.es_client.search(query="red")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['filename'], 'test.jpg')


if __name__ == '__main__':
    unittest.main() 