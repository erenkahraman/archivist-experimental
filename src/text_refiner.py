"""Module for refining generated text prompts."""
import re
import logging

logger = logging.getLogger(__name__)

def refine_text(prompt: str) -> str:
    """
    Refine the prompt text to fix punctuation, smooth transitions, and improve readability.
    
    Args:
        prompt: The raw generated prompt
        
    Returns:
        Refined prompt text
    """
    try:
        if not prompt:
            return "No pattern detected."
            
        # Remove extra spaces
        refined = re.sub(r'\s+', ' ', prompt).strip()
        
        # Fix punctuation
        refined = re.sub(r'\s+\.', '.', refined)  # Remove spaces before periods
        refined = re.sub(r'\.+', '.', refined)    # Replace multiple periods with a single one
        refined = re.sub(r'\s+,', ',', refined)   # Remove spaces before commas
        
        # Ensure sentence ends with a period
        if not refined.endswith('.'):
            refined += '.'
            
        # Capitalize the first letter
        refined = refined[0].upper() + refined[1:] if refined else refined
        
        # Fix common pattern description issues
        refined = re.sub(r'pattern pattern', 'pattern', refined, flags=re.IGNORECASE)
        refined = re.sub(r'featuring featuring', 'featuring', refined, flags=re.IGNORECASE)
        
        # Remove redundant color mentions
        color_pattern = r'(\w+)\s+colored\s+\1'
        refined = re.sub(color_pattern, r'\1', refined, flags=re.IGNORECASE)
        
        logger.debug(f"Original prompt: {prompt}")
        logger.debug(f"Refined prompt: {refined}")
        
        return refined
        
    except Exception as e:
        logger.error(f"Error refining text: {str(e)}")
        return prompt  # Return original if refinement fails 