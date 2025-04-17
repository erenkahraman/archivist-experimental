import json
import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class AIResponseParser:
    """
    Utility class for parsing and repairing JSON from AI responses
    """
    
    @staticmethod
    def extract_json_from_response(response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from response text with efficient error handling
        
        Args:
            response_text: The raw text response from the AI
            
        Returns:
            Extracted JSON as a dictionary, or an empty dict if extraction fails
        """
        try:
            # First attempt: direct JSON parsing
            try:
                return json.loads(response_text.strip())
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from markdown code blocks
            markdown_json = AIResponseParser._extract_json_from_markdown(response_text)
            if markdown_json:
                return markdown_json
            
            # Extract JSON from arbitrary text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_candidate = response_text[json_start:json_end]
                try:
                    # Fix common JSON format issues
                    json_candidate = AIResponseParser._fix_json_format(json_candidate)
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON candidate")
            
            logger.error("Failed to extract valid JSON from AI response")
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {str(e)}")
            return {}
    
    @staticmethod
    def _fix_json_format(json_str: str) -> str:
        """Fix common JSON formatting issues"""
        # Replace single quotes with double quotes
        fixed = re.sub(r'\'([^\']+)\'(\s*:)', r'"\1"\2', json_str)
        fixed = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed)
        
        # Fix trailing commas in arrays and objects
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix missing quotes around property names
        fixed = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', fixed)
        
        # Fix missing quotes around string values
        fixed = re.sub(r':\s*([a-zA-Z0-9_]+)(\s*[,}])', r': "\1"\2', fixed)
        
        return fixed
    
    @staticmethod
    def _extract_json_from_markdown(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks"""
        # Look for ```json or ``` blocks
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_block_pattern, text)
        
        for match in matches:
            try:
                # Try to parse this block as JSON
                json_candidate = AIResponseParser._fix_json_format(match)
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                continue
        
        return None
    
    @staticmethod
    def extract_lists_from_text(text: str) -> List[str]:
        """
        Extract lists from plain text responses, useful for keyword extraction
        """
        # Look for numbered lists
        numbered_list_pattern = r'\d+\.\s+([^\n]+)'
        numbered_items = re.findall(numbered_list_pattern, text)
        
        # Look for bullet lists
        bullet_list_pattern = r'[•\-\*]\s+([^\n]+)'
        bullet_items = re.findall(bullet_list_pattern, text)
        
        # Combine and clean up
        all_items = numbered_items + bullet_items
        
        # If no structured lists found, try extracting comma-separated items
        if not all_items and ',' in text:
            # Find sentences that might contain lists
            sentences = re.split(r'[.!?]\s+', text)
            for sentence in sentences:
                if ',' in sentence and len(sentence.split(',')) >= 3:
                    # Likely a list-like sentence
                    items = [item.strip() for item in sentence.split(',')]
                    all_items.extend(items)
        
        # Clean up items
        clean_items = []
        for item in all_items:
            # Remove any trailing punctuation and leading/trailing whitespace
            clean_item = re.sub(r'[,;.]$', '', item).strip()
            if clean_item:
                clean_items.append(clean_item)
        
        return clean_items[:10]  # Limit to 10 items 