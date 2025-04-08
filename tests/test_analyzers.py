"""
Tests for image analyzers functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from src.analyzers.color_analyzer import ColorAnalyzer


class TestColorAnalyzer(unittest.TestCase):
    """Test cases for the ColorAnalyzer."""

    def setUp(self):
        """Set up the test environment."""
        self.analyzer = ColorAnalyzer()
        
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
    def test_extract_colors(self):
        """Test color extraction from an image."""
        colors = self.analyzer.extract_colors(self.test_image)
        
        # Should extract at least one color
        self.assertGreaterEqual(len(colors), 1)
        
        # First color should be red (or close to it)
        first_color = colors[0]
        self.assertGreaterEqual(first_color[0], 200)  # R value
        self.assertLessEqual(first_color[1], 50)      # G value
        self.assertLessEqual(first_color[2], 50)      # B value


if __name__ == '__main__':
    unittest.main() 