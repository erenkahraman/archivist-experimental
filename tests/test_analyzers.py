"""
Tests for image analyzers functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

from src.analyzers.gemini_analyzer import GeminiAnalyzer


class TestGeminiAnalyzer(unittest.TestCase):
    """Test cases for the GeminiAnalyzer."""

    def setUp(self):
        """Set up the test environment."""
        self.analyzer = GeminiAnalyzer()
        
    @patch('src.analyzers.gemini_analyzer.GeminiAnalyzer._get_from_cache')
    def test_cache_check(self, mock_get_from_cache):
        """Test that the analyzer checks the cache first."""
        # Setup mock
        mock_result = {"main_theme": "Test Pattern"}
        mock_get_from_cache.return_value = mock_result
        
        # Call the analyze method
        result = self.analyzer.analyze_image("test_path.jpg")
        
        # Verify cache was checked
        mock_get_from_cache.assert_called_once_with("test_path.jpg")
        
        # Verify result matches mock
        self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main() 