from typing import List, Dict
import torch
from transformers import CLIPProcessor, CLIPModel
from functools import lru_cache

class ImageSearcher:
    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: torch.device):
        self.model = model
        self.processor = processor
        self.device = device

    def search(self, query: str, metadata_list: List[Dict], k: int = 10) -> List[Dict]:
        """Enhanced search for images based on query with hierarchical ranking."""
        try:
            if not query.strip():
                return metadata_list[:k]
                
            query_terms = [term.lower().strip() for term in query.lower().split() if term.strip()]
            
            if not metadata_list:
                print("No images in database")
                return []

            # Get query features
            query_features = self._get_query_features(query)

            # Calculate scores for each image
            results = self._calculate_scores(query_terms, query_features, metadata_list)
            
            # Normalize and sort results
            results = self._normalize_results(results)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            if results:
                print(f"Search for '{query}' found {len(results)} results")
                print(f"Top match: {results[0]['patterns']['category']} with score {results[0]['similarity']:.2f}")
            
            return results[:k]

        except Exception as e:
            print(f"Error in search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    @lru_cache(maxsize=1000)
    def _get_query_features(self, query: str) -> torch.Tensor:
        """
        Cache query features for better performance.
        
        Args:
            query (str): Search query
            
        Returns:
            torch.Tensor: Query features
        """
        with torch.no_grad():
            text_inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            return self.model.get_text_features(**text_inputs).cpu()

    def _calculate_scores(self, query_terms: List[str], query_features: torch.Tensor, 
                         metadata_list: List[Dict]) -> List[Dict]:
        """Calculate scores for each image."""
        results = []
        max_scores = {
            'exact_match': 0.0,
            'semantic': 0.0,
            'pattern': 0.0,
            'color': 0.0,
            'element': 0.0,
            'attribute': 0.0,
            'term_matches': 0
        }
        
        for metadata in metadata_list:
            scores = self._calculate_individual_scores(query_terms, query_features, metadata)
            
            # Track maximum scores for normalization
            for key in max_scores:
                if key in scores and scores[key] > max_scores[key]:
                    max_scores[key] = scores[key]
            
            # Calculate final score
            final_score = self._calculate_final_score(scores)
            
            if final_score > 0:
                results.append({
                    **metadata,
                    'raw_score': final_score,
                    'scores': scores,
                    'matched_terms': scores['term_matches']
                })
                
        return results

    def _normalize_results(self, results: List[Dict]) -> List[Dict]:
        """Normalize search results."""
        if not results:
            return results
            
        max_raw_score = max(result['raw_score'] for result in results)
        
        for result in results:
            normalized_score = (result['raw_score'] / max_raw_score) * 0.95
            result['similarity'] = normalized_score
            del result['raw_score']
            del result['scores']
            
        return results 

    def _calculate_individual_scores(self, query_terms: List[str], query_features: torch.Tensor, metadata: Dict) -> Dict:
        """Calculate individual score components for a single image."""
        scores = {
            'exact_match': 0.0,
            'semantic': 0.0,
            'pattern': 0.0,
            'color': 0.0,
            'element': 0.0,
            'attribute': 0.0,
            'term_matches': 0
        }

        try:
            # 1. Pattern matching
            if 'patterns' in metadata:
                patterns = metadata['patterns']
                
                # Primary pattern matching
                if 'category' in patterns:
                    category = patterns['category'].lower()
                    confidence = patterns.get('category_confidence', 0.8)
                    
                    # Check for exact matches
                    for term in query_terms:
                        if term == category:
                            scores['exact_match'] += 5.0 * confidence
                            scores['term_matches'] += 1
                        elif term in category:
                            scores['pattern'] += 2.0 * confidence
                            scores['term_matches'] += 1
                
                # Secondary patterns matching
                if 'secondary_patterns' in patterns:
                    for pattern in patterns['secondary_patterns']:
                        pattern_name = pattern['name'].lower()
                        pattern_conf = pattern.get('confidence', 0.5)
                        
                        for term in query_terms:
                            if term == pattern_name:
                                scores['pattern'] += pattern_conf * 1.5
                                scores['term_matches'] += 1
                            elif term in pattern_name:
                                scores['pattern'] += pattern_conf
                                scores['term_matches'] += 1
                
                # NEW: Specific elements matching
                if 'elements' in patterns:
                    for element in patterns['elements']:
                        element_name = element['name'].lower()
                        element_conf = element.get('confidence', 0.5)
                        
                        for term in query_terms:
                            if term == element_name:
                                scores['element'] += element_conf * 2.0
                                scores['term_matches'] += 1
                            elif term in element_name or element_name in term:
                                scores['element'] += element_conf * 1.0
                                scores['term_matches'] += 1

            # 2. Color matching
            if 'colors' in metadata:
                colors = metadata['colors']
                if 'dominant_colors' in colors:
                    for color in colors['dominant_colors']:
                        color_name = color['name'].lower()
                        proportion = color['proportion']
                        
                        for term in query_terms:
                            if term == color_name:
                                scores['color'] += proportion * 2.0
                                scores['term_matches'] += 1
                            elif term in color_name:
                                scores['color'] += proportion
                                scores['term_matches'] += 1

            # 3. Semantic matching with prompt
            if 'patterns' in metadata and 'prompt' in metadata['patterns']:
                prompt = metadata['patterns']['prompt']
                if 'final_prompt' in prompt:
                    prompt_text = prompt['final_prompt']
                    
                    # Process prompt through CLIP
                    with torch.no_grad():
                        prompt_inputs = self.processor(
                            text=[prompt_text],
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                        
                        prompt_features = self.model.get_text_features(**prompt_inputs)
                        similarity = torch.nn.functional.cosine_similarity(
                            query_features.to(self.device),
                            prompt_features,
                            dim=1
                        )
                        scores['semantic'] = float(similarity[0].cpu()) * 3.0

            return scores

        except Exception as e:
            print(f"Error calculating individual scores: {str(e)}")
            return scores

    def _calculate_final_score(self, scores: Dict) -> float:
        """Calculate final score with weighted components."""
        weights = {
            'exact_match': 0.35,  # Exact matches are highest priority
            'semantic': 0.25,     # Semantic similarity is second priority
            'element': 0.15,      # NEW: Element matches are important
            'pattern': 0.10,      # Pattern matches are next priority
            'color': 0.10,        # Color matches are next priority
            'attribute': 0.05,    # Attribute matches are lowest priority
        }
        
        final_score = sum(scores[key] * weight 
                         for key, weight in weights.items() 
                         if key in scores)
        
        # Add bonus for matching multiple terms
        term_coverage = scores['term_matches'] / 10  # Normalize by assuming max 10 terms
        final_score += term_coverage * 0.05
        
        return final_score 