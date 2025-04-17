"""
Search module that provides unified search functionality.
"""
import logging
from typing import Dict, List, Any
import re
from src.config.config import DEFAULT_SEARCH_LIMIT, DEFAULT_MIN_SIMILARITY

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Search engine that provides pattern search functionality through metadata.
    This implementation works with PatternAnalyzer's metadata format.
    """
    
    def __init__(self):
        """Initialize the search engine"""
        self.metadata = {}
        self.search_index = {}
        logger.info("Search engine initialized")
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """
        Set the metadata and build the search index
        
        Args:
            metadata: Dictionary mapping file paths to metadata
        """
        self.metadata = metadata
        self._build_search_index(metadata)
        logger.info(f"Search index built with {len(metadata)} items")
    
    def save_metadata(self) -> bool:
        """
        Placeholder for compatibility with old code.
        In this implementation, metadata is managed by PatternAnalyzer.
        
        Returns:
            bool: Always True for compatibility
        """
        logger.info("SearchEngine.save_metadata called - metadata is managed by PatternAnalyzer")
        return True
    
    def clear_metadata(self) -> bool:
        """
        Clear all metadata and search index
        
        Returns:
            bool: Success status
        """
        self.metadata = {}
        self.search_index = {}
        logger.info("Cleared search engine metadata and index")
        return True
    
    def _build_search_index(self, metadata: Dict[str, Any]):
        """
        Build an inverted index for faster text search
        
        Args:
            metadata: Dictionary mapping file paths to metadata
        """
        self.search_index = {}
        
        # Weights for different fields (higher = more important)
        field_weights = {
            "patterns.primary_pattern": 5.0,
            "patterns.main_theme": 4.0, 
            "patterns.style_keywords": 3.0,
            "patterns.content_details": 2.0
        }
        
        # Process each item in metadata
        for path, item in metadata.items():
            # Index each field with its weight
            for field_path, weight in field_weights.items():
                field_value = self._get_nested_field(item, field_path)
                
                if not field_value:
                    continue
                
                # Handle different value types
                if isinstance(field_value, list):
                    # For lists (like keywords), index each item
                    for entry in field_value:
                        if isinstance(entry, dict) and "name" in entry:
                            self._index_text(entry["name"], path, weight)
                        elif isinstance(entry, str):
                            self._index_text(entry, path, weight)
                elif isinstance(field_value, str):
                    # For strings, index directly
                    self._index_text(field_value, path, weight)
                elif isinstance(field_value, dict) and "name" in field_value:
                    # For dictionaries with a name field
                    self._index_text(field_value["name"], path, weight)
    
    def _get_nested_field(self, data: Dict, field_path: str) -> Any:
        """
        Get a value from a nested dictionary using dot notation
        
        Args:
            data: Dictionary to extract value from
            field_path: Path to the field using dot notation (e.g., "patterns.main_theme")
            
        Returns:
            The value at the specified path or None if not found
        """
        if not data or not field_path:
            return None
            
        parts = field_path.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value
    
    def _index_text(self, text: str, doc_id: str, weight: float = 1.0):
        """
        Index text by tokenizing and mapping tokens to document IDs
        
        Args:
            text: Text to index
            doc_id: Document ID to associate with this text
            weight: Weight to assign to this text (higher = more important)
        """
        if not text or not isinstance(text, str):
            return
            
        # Normalize and tokenize
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Index individual words
        for word in words:
            if len(word) > 2:  # Skip very short words
                if word not in self.search_index:
                    self.search_index[word] = {}
                
                # Store with weight (use maximum if already exists)
                self.search_index[word][doc_id] = max(
                    weight, 
                    self.search_index[word].get(doc_id, 0)
                )
        
        # Index bigrams (pairs of words) for phrase matching
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in self.search_index:
                    self.search_index[bigram] = {}
                
                # Give bigrams slightly higher weight
                bigram_weight = weight * 1.2
                self.search_index[bigram][doc_id] = max(
                    bigram_weight, 
                    self.search_index[bigram].get(doc_id, 0)
                )
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a search using the search index
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching metadata items
        """
        if not query or not self.search_index:
            return []
            
        # Normalize query
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        terms = query.split()
        
        # Find matching documents with scores
        scores = {}
        
        # Process each term in the query
        for term in terms:
            if len(term) <= 2:
                continue
                
            if term in self.search_index:
                # Add scores for this term
                for doc_id, weight in self.search_index[term].items():
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += weight
        
        # Look for bigrams in the query
        for i in range(len(terms) - 1):
            if len(terms[i]) > 2 and len(terms[i+1]) > 2:
                bigram = f"{terms[i]} {terms[i+1]}"
                if bigram in self.search_index:
                    # Add scores for this bigram with a boost
                    for doc_id, weight in self.search_index[bigram].items():
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += weight * 1.5  # Give extra weight to bigram matches
        
        # Sort by score
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to list of metadata entries
        results = []
        for doc_id, score in ranked_results[:max_results]:
            if doc_id in self.metadata:
                # Add the score to the metadata for debugging
                metadata_copy = dict(self.metadata[doc_id])
                metadata_copy['search_score'] = score
                results.append(metadata_copy)
        
        return results

    def metadata_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> List[Dict[str, Any]]:
        """
        Perform a metadata-based search and filter by similarity
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of matching metadata items
        """
        logger.info(f"Performing metadata search with query: '{query}', limit: {limit}")
        
        # Basic validation
        if not query or not self.metadata:
            logger.warning("Empty query or no metadata to search")
            return []
        
        # Use the regular search method to get initial results
        results = self.search(query, max_results=limit*2)  # Get more results to filter by similarity
        
        # Filter results by minimum similarity
        if min_similarity > 0:
            filtered_results = []
            for result in results:
                if result.get('search_score', 0) >= min_similarity:
                    filtered_results.append(result)
            
            results = filtered_results
        
        # Return the top results up to the limit
        return results[:limit]

    def find_similar_images(self, image_path: str, text_query: str = "", k: int = 10, exclude_source: bool = True, image_weight: float = 0.7, text_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find images similar to the given image path (placeholder implementation).
        In this version, we just use text search based on the metadata.
        
        Args:
            image_path: Path to the image to find similar images for
            text_query: Additional text query to refine results
            k: Number of results to return
            exclude_source: Whether to exclude the source image from results
            image_weight: Weight for image similarity (not used in this implementation)
            text_weight: Weight for text similarity (not used in this implementation)
            
        Returns:
            List of similar images with metadata
        """
        logger.info(f"Finding similar images for {image_path} with text query: {text_query}")
        
        # Use text search if a text query is provided
        if text_query:
            logger.info(f"Using text search for similarity with query: {text_query}")
            return self.metadata_search(text_query, k)
            
        # If no text query, try to find the reference image metadata
        ref_metadata = None
        if image_path in self.metadata:
            ref_metadata = self.metadata[image_path]
        else:
            # Try to find by filename
            import os
            filename = os.path.basename(image_path)
            for path, metadata in self.metadata.items():
                if metadata.get('filename') == filename:
                    ref_metadata = metadata
                    break
        
        # If we found reference metadata, use its pattern information for search
        if ref_metadata and 'patterns' in ref_metadata:
            # Extract pattern info for search
            patterns = ref_metadata['patterns']
            search_terms = []
            
            # Add pattern information to search terms
            if isinstance(patterns, dict):
                if 'primary_pattern' in patterns:
                    search_terms.append(patterns['primary_pattern'])
                if 'main_theme' in patterns:
                    search_terms.append(patterns['main_theme'])
                if 'style_keywords' in patterns and isinstance(patterns['style_keywords'], list):
                    search_terms.extend(patterns['style_keywords'][:3])  # Add top keywords
            
            # Create a search query from the pattern information
            if search_terms:
                search_query = " ".join(search_terms)
                logger.info(f"Created search query from reference image: {search_query}")
                results = self.metadata_search(search_query, k+1 if exclude_source else k)
                
                # Filter out source image if needed
                if exclude_source:
                    results = [r for r in results if r.get('filename') != os.path.basename(image_path)]
                
                return results[:k]
        
        # Fallback: return random images from metadata
        logger.warning("No pattern data found for reference image, returning random results")
        import random
        random_keys = list(self.metadata.keys())
        if exclude_source and image_path in random_keys:
            random_keys.remove(image_path)
            
        # Randomize and take k items
        random.shuffle(random_keys)
        results = []
        for key in random_keys[:k]:
            result = dict(self.metadata[key])
            result['similarity'] = 0.5  # Add dummy similarity score
            results.append(result)
            
        return results

    def delete_image(self, image_path: str) -> bool:
        """
        Delete an image from the search engine metadata
        
        Args:
            image_path: Path or filename of the image to delete
            
        Returns:
            bool: Success status
        """
        logger.info(f"Deleting image from search engine: {image_path}")
        
        try:
            # Try to find by direct path first
            if image_path in self.metadata:
                del self.metadata[image_path]
                # Rebuild search index
                self._build_search_index(self.metadata)
                logger.info(f"Deleted image by direct path: {image_path}")
                return True
                
            # Try to find by filename
            import os
            filename = os.path.basename(image_path)
            for path in list(self.metadata.keys()):
                meta = self.metadata[path]
                if meta.get('filename') == filename or os.path.basename(path) == filename:
                    del self.metadata[path]
                    # Rebuild search index
                    self._build_search_index(self.metadata)
                    logger.info(f"Deleted image by filename: {filename}")
                    return True
                    
            logger.warning(f"No metadata found for image: {image_path}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            return False
            
    def cleanup_missing_files(self) -> int:
        """
        Remove metadata entries for files that no longer exist
        
        Returns:
            int: Number of entries cleaned up
        """
        import os
        cleaned_count = 0
        
        try:
            # Create a copy of keys to iterate over while deleting
            paths_to_check = list(self.metadata.keys())
            
            for path in paths_to_check:
                # Check if the file exists
                if not os.path.exists(path):
                    # Try to check with absolute path if it's a relative path
                    from src.config.config import UPLOAD_DIR
                    absolute_path = os.path.join(UPLOAD_DIR, os.path.basename(path))
                    
                    if not os.path.exists(absolute_path):
                        # File doesn't exist, remove from metadata
                        del self.metadata[path]
                        cleaned_count += 1
            
            # Rebuild search index if we cleaned any entries
            if cleaned_count > 0:
                self._build_search_index(self.metadata)
                logger.info(f"Cleaned up {cleaned_count} missing file entries")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up missing files: {str(e)}")
            return 0
            
    def process_image(self, image_path) -> Dict[str, Any]:
        """
        Process an image for the search engine
        This is a compatibility method that forwards to PatternAnalyzer
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict: Image metadata
        """
        logger.info(f"Processing image through SearchEngine: {image_path}")
        
        try:
            # Import here to avoid circular imports
            from src.core.pattern_analyzer import PatternAnalyzer
            from src.api import GEMINI_API_KEY
            
            # Create a new analyzer instance
            analyzer = PatternAnalyzer(api_key=GEMINI_API_KEY)
            
            # Process the image
            metadata = analyzer.process_image(image_path)
            
            # Update our metadata with the results
            if metadata:
                self.set_metadata(analyzer.get_all_metadata())
                
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None
            
    def create_thumbnail(self, image_path):
        """
        Create a thumbnail for an image
        This is a compatibility method that forwards to PatternAnalyzer
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path: Path to the thumbnail
        """
        logger.info(f"Creating thumbnail through SearchEngine: {image_path}")
        
        try:
            # Import here to avoid circular imports
            from src.core.pattern_analyzer import PatternAnalyzer
            from src.api import GEMINI_API_KEY
            
            # Create a new analyzer instance
            analyzer = PatternAnalyzer(api_key=GEMINI_API_KEY)
            
            # Create the thumbnail
            return analyzer.create_thumbnail(image_path)
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return None

# Create a single instance for import
search_engine = SearchEngine()

__all__ = ['search_engine', 'SearchEngine'] 