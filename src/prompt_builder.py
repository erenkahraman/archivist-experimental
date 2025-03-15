import json
import random
import logging
from pathlib import Path
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Dedicated class for building detailed pattern prompts using templates"""
    
    def __init__(self, template_path=None):
        """Initialize with template path"""
        if not template_path:
            template_path = Path(__file__).parent / "templates" / "pattern_templates.json"
        
        self.templates = self._load_templates(template_path)
        self.fallback_template = "{pattern_type} pattern featuring {elements} in {colors} on a {background} background."
    
    def _load_templates(self, template_path):
        """Load pattern templates from JSON file"""
        try:
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Template file not found: {template_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return {}
    
    def build_prompt(self, pattern_info: Dict, color_info: Dict, style_info: Dict = None) -> Dict:
        """Build a detailed prompt based on pattern analysis, color information, and style analysis"""
        try:
            # Get basic pattern information
            pattern_type = pattern_info.get('category', '').lower()
            pattern_confidence = pattern_info.get('category_confidence', 0.0)
            
            # Extract colors with better background detection
            colors = self._extract_colors(color_info)
            
            # Use the most dominant color as background (usually the first one by proportion)
            if len(colors) > 1:
                background_color = colors[0]  # Most dominant color is likely background
                foreground_colors = colors[1:]  # Other colors are foreground elements
            else:
                background_color = colors[0] if colors else "neutral"
                foreground_colors = []
            
            # Get pattern elements and details
            elements = self._get_elements(pattern_info)
            specific_details = pattern_info.get('specific_details', [])
            density = pattern_info.get('density', {}).get('type', 'regular')
            
            # Get style information
            layout = style_info.get('layout', {}).get('type', 'balanced') if style_info else 'balanced'
            scale = style_info.get('scale', {}).get('type', 'medium') if style_info else 'medium'
            texture = style_info.get('texture_type', {}).get('type', 'smooth') if style_info else 'smooth'
            
            # Determine the pattern category for template selection
            category = self._determine_category(pattern_type)
            
            # Build the prompt based on the category
            if category == "floral":
                return self._build_floral_prompt(pattern_type, elements, foreground_colors, background_color, 
                                               layout, scale, texture, specific_details)
            elif category == "leaf_botanical":
                return self._build_botanical_prompt(pattern_type, elements, foreground_colors, background_color, 
                                                  layout, scale, texture, specific_details)
            elif category == "animal_print":
                return self._build_animal_print_prompt(pattern_type, elements, foreground_colors, background_color, 
                                                     layout, scale, texture, specific_details)
            elif category == "geometric":
                return self._build_geometric_prompt(pattern_type, elements, foreground_colors, background_color, 
                                                  layout, scale, texture, specific_details)
            elif category == "abstract":
                return self._build_abstract_prompt(pattern_type, elements, foreground_colors, background_color, 
                                                 layout, scale, texture, specific_details)
            else:
                return self._build_generic_prompt(pattern_type, elements, foreground_colors, background_color, 
                                                layout, scale, texture, specific_details)
        
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Fallback to a simple prompt
            return {
                "final_prompt": f"Pattern with {pattern_type if pattern_type else 'various'} elements.",
                "components": {},
                "completeness_score": 0.3
            }
    
    def _determine_category(self, pattern_type: str) -> str:
        """Determine the pattern category based on pattern type"""
        pattern_type = pattern_type.lower()
        
        if any(floral in pattern_type for floral in ["floral", "flower", "rose", "tulip", "daisy", "peony", "lily", "blossom"]):
            return "floral"
        elif any(leaf in pattern_type for leaf in ["leaf", "botanical", "palm", "fern", "tropical", "plant", "foliage"]):
            return "leaf_botanical"
        elif any(animal in pattern_type for animal in ["animal", "leopard", "zebra", "tiger", "giraffe", "snake", "crocodile"]):
            return "animal_print"
        elif any(geo in pattern_type for geo in ["geometric", "circle", "square", "triangle", "diamond", "hexagon", "line"]):
            return "geometric"
        elif any(abstract in pattern_type for abstract in ["abstract", "modern", "contemporary", "artistic", "expressionist"]):
            return "abstract"
        else:
            # Try to infer from elements if pattern_type is not specific enough
            return "generic"
    
    def _extract_colors(self, color_info: Dict) -> List[str]:
        """Extract color information from color analysis"""
        colors = []
        if color_info and 'dominant_colors' in color_info:
            colors = [c['name'].lower() for c in color_info['dominant_colors']]
        return colors if colors else ["neutral"]
    
    def _get_elements(self, pattern_info: Dict) -> List[Dict]:
        """Extract element information with enhanced confidence handling"""
        elements = []
        if 'elements' in pattern_info:
            # Get high confidence elements first
            high_conf = [e for e in pattern_info['elements'] if e['confidence'] > 0.35]
            # Then medium confidence
            med_conf = [e for e in pattern_info['elements'] if 0.35 >= e['confidence'] > 0.25 and e['name'] not in [h['name'] for h in high_conf]]
            
            # Combine with confidence markers
            elements = [{'name': e['name'], 'confidence': e['confidence'], 'prominence': 'high'} for e in high_conf]
            elements.extend([{'name': e['name'], 'confidence': e['confidence'], 'prominence': 'medium'} for e in med_conf])
        
        return elements
    
    def _get_specific_element_details(self, element_name: str, category: str, subcategory: str) -> Dict:
        """Get specific details for an element from templates"""
        try:
            if category in self.templates.get("pattern_types", {}):
                category_data = self.templates["pattern_types"][category]
                
                if subcategory in category_data:
                    subcategory_data = category_data[subcategory]
                    
                    for key, details in subcategory_data.items():
                        if any(term in element_name.lower() for term in [key.lower(), key.lower()[:-1]]):  # Handle singular/plural
                            return {
                                "type": random.choice(details.get("types", [""])) if "types" in details else "",
                                "descriptor": random.choice(details.get("descriptors", [""])) if "descriptors" in details else "",
                                "colors": details.get("colors", []),
                                "details": details.get("details", [])
                            }
            
            # Fallback to generic description
            return {
                "type": "",
                "descriptor": "",
                "colors": [],
                "details": []
            }
        
        except Exception as e:
            logger.error(f"Error getting element details: {e}")
            return {
                "type": "",
                "descriptor": "",
                "colors": [],
                "details": []
            }
    
    def _build_floral_prompt(self, pattern_type, elements, colors, background, layout, scale, texture, specific_details):
        """Build a detailed floral pattern prompt with improved color matching"""
        try:
            # Get template data
            floral_data = self.templates.get("pattern_types", {}).get("floral", {})
            
            # Select an appropriate adjective
            adjectives = floral_data.get("adjectives", ["Elegant", "Delicate"])
            adjective = random.choice(adjectives)
            
            # Process floral elements
            floral_descriptions = []
            for element in elements:
                element_name = element['name']
                
                # Get specific flower details
                flower_details = self._get_specific_element_details(element_name, "floral", "specific_flowers")
                
                # Build flower description
                flower_type = flower_details["type"]
                descriptor = flower_details["descriptor"]
                
                # Improved color matching
                element_color = self._find_best_color_match(flower_details["colors"], colors)
                
                # Build the description
                if flower_type and descriptor and element_color:
                    floral_descriptions.append(f"{descriptor} {element_color} {flower_type}")
                elif flower_type and element_color:
                    floral_descriptions.append(f"{element_color} {flower_type}")
                elif descriptor and element_name and element_color:
                    floral_descriptions.append(f"{descriptor} {element_color} {element_name}")
                elif element_name and element_color:
                    floral_descriptions.append(f"{element_color} {element_name}")
                elif element_name:
                    floral_descriptions.append(element_name)
            
            # If no specific flowers were detected, use generic description
            if not floral_descriptions and "floral" in pattern_type:
                if colors:
                    floral_descriptions.append(f"{colors[0]} floral motifs")
                else:
                    floral_descriptions.append("floral motifs")
            
            # Build the prompt
            prompt_parts = []
            
            # Start with pattern type and adjective
            prompt_parts.append(f"{adjective} floral pattern")
            
            # Add floral descriptions
            if floral_descriptions:
                prompt_parts.append(f"featuring {', '.join(floral_descriptions)}")
            
            # Use analyzed layout instead of random choice
            prompt_parts.append(f"arranged in a {layout} layout")
            
            # Add scale and texture from style analysis
            prompt_parts.append(f"with {texture} textural details at {scale} scale")
            
            # Add background
            prompt_parts.append(f"on a {background} background")
            
            # Add specific details if available
            if specific_details:
                prompt_parts.append(f"with {', '.join(specific_details[:2])}")
            
            # Combine all parts
            final_prompt = ". ".join([p.strip() for p in prompt_parts if p.strip()])
            if not final_prompt.endswith('.'):
                final_prompt += '.'
            
            # Capitalize first letter
            final_prompt = final_prompt[0].upper() + final_prompt[1:]
            
            return {
                "final_prompt": final_prompt,
                "components": {
                    "pattern_type": "floral",
                    "adjective": adjective,
                    "elements": floral_descriptions,
                    "layout": layout,
                    "scale": scale,
                    "texture": texture,
                    "background": background,
                    "specific_details": specific_details
                },
                "completeness_score": min(1.0, len(prompt_parts) / 5)
            }
        
        except Exception as e:
            logger.error(f"Error building floral prompt: {e}")
            return self._build_generic_prompt(pattern_type, elements, colors, background, layout, scale, texture, specific_details)
    
    def _find_best_color_match(self, element_colors, available_colors):
        """Find the best matching color from available colors for an element"""
        if not element_colors or not available_colors:
            return available_colors[0] if available_colors else ""
        
        # Try exact matches first
        for color in available_colors:
            if color in element_colors:
                return color
        
        # Try partial matches
        for avail_color in available_colors:
            for elem_color in element_colors:
                if avail_color in elem_color or elem_color in avail_color:
                    return avail_color
        
        # Default to first available color
        return available_colors[0]
    
    def _build_botanical_prompt(self, pattern_type, elements, colors, background, layout, scale, texture, specific_details):
        """Build a detailed botanical/leaf pattern prompt"""
        try:
            # Get template data
            botanical_data = self.templates.get("pattern_types", {}).get("leaf_botanical", {})
            
            # Select an appropriate adjective
            adjectives = botanical_data.get("adjectives", ["Natural", "Organic"])
            adjective = random.choice(adjectives)
            
            # Process leaf elements
            leaf_descriptions = []
            for element in elements:
                element_name = element['name']
                
                # Get specific leaf details
                leaf_details = self._get_specific_element_details(element_name, "leaf_botanical", "specific_leaves")
                
                # Build leaf description
                leaf_type = leaf_details["type"]
                descriptor = leaf_details["descriptor"]
                
                # Match with appropriate color
                element_color = ""
                if leaf_details["colors"] and colors:
                    matching_colors = [c for c in colors if c in leaf_details["colors"]]
                    element_color = matching_colors[0] if matching_colors else colors[0]
                elif colors:
                    element_color = colors[0]
                
                # Build the description
                if leaf_type and descriptor and element_color:
                    leaf_descriptions.append(f"{descriptor} {element_color} {leaf_type}")
                elif leaf_type and element_color:
                    leaf_descriptions.append(f"{element_color} {leaf_type}")
                elif descriptor and element_name and element_color:
                    leaf_descriptions.append(f"{descriptor} {element_color} {element_name}")
                elif element_name and element_color:
                    leaf_descriptions.append(f"{element_color} {element_name}")
                elif element_name:
                    leaf_descriptions.append(element_name)
            
            # If no specific leaves were detected, use generic description
            if not leaf_descriptions and any(term in pattern_type for term in ["leaf", "botanical", "tropical"]):
                if colors and any(c in ["green", "emerald", "olive", "teal"] for c in colors):
                    green_color = next((c for c in colors if c in ["green", "emerald", "olive", "teal"]), "green")
                    leaf_descriptions.append(f"{green_color} leaves")
                elif colors:
                    leaf_descriptions.append(f"{colors[0]} botanical elements")
                else:
                    leaf_descriptions.append("botanical elements")
            
            # Get arrangement style
            arrangements = botanical_data.get("arrangements", ["scattered", "layered"])
            arrangement = random.choice(arrangements)
            
            # Get texture details
            textures = botanical_data.get("textures", ["detailed", "veined"])
            texture = random.choice(textures)
            
            # Build the prompt
            prompt_parts = []
            
            # Start with pattern type and adjective
            prompt_parts.append(f"{adjective} botanical pattern")
            
            # Add density if available
            if layout and layout != "balanced":
                prompt_parts.append(f"with {layout}")
            
            # Add leaf descriptions
            if leaf_descriptions:
                prompt_parts.append(f"featuring {', '.join(leaf_descriptions)}")
            
            # Add texture details
            prompt_parts.append(f"with {texture} details")
            
            # Add arrangement
            prompt_parts.append(f"arranged in {arrangement} composition")
            
            # Add background
            prompt_parts.append(f"on a {background} background")
            
            # Add specific details if available
            if specific_details:
                prompt_parts.append(f"with {', '.join(specific_details[:2])}")
            
            # Combine all parts
            final_prompt = ". ".join([p.strip() for p in prompt_parts if p.strip()])
            if not final_prompt.endswith('.'):
                final_prompt += '.'
            
            # Capitalize first letter
            final_prompt = final_prompt[0].upper() + final_prompt[1:]
            
            return {
                "final_prompt": final_prompt,
                "components": {
                    "pattern_type": "botanical",
                    "adjective": adjective,
                    "elements": leaf_descriptions,
                    "texture": texture,
                    "arrangement": arrangement,
                    "background": background,
                    "specific_details": specific_details
                },
                "completeness_score": min(1.0, len(prompt_parts) / 5)
            }
        
        except Exception as e:
            logger.error(f"Error building botanical prompt: {e}")
            return self._build_generic_prompt(pattern_type, elements, colors, background, layout, scale, texture, specific_details)
    
    def _build_animal_print_prompt(self, pattern_type, elements, colors, background, layout, scale, texture, specific_details):
        """Build a detailed animal print pattern prompt"""
        try:
            # Get template data
            animal_data = self.templates.get("pattern_types", {}).get("animal_print", {})
            
            # Determine specific animal print type
            animal_type = "animal"
            for specific_animal in ["leopard", "zebra", "tiger", "giraffe", "snake", "crocodile"]:
                if specific_animal in pattern_type:
                    animal_type = specific_animal
                    break
            
            # Get specific print details
            specific_print = animal_data.get("specific_prints", {}).get(animal_type, {})
            
            # Get pattern details
            pattern_details = specific_print.get("pattern_details", [])
            pattern_detail = random.choice(pattern_details) if pattern_details else ""
            
            # Get color combinations
            color_combos = specific_print.get("colors", [])
            color_combo = ""
            if color_combos:
                color_combo = random.choice(color_combos)
            elif colors and len(colors) > 1:
                color_combo = f"{colors[0]} and {colors[1]}"
            elif colors:
                color_combo = f"{colors[0]} and black"
            else:
                color_combo = "natural tones"
            
            # Get style
            styles = specific_print.get("styles", [])
            style = random.choice(styles) if styles else ""
            
            # Build the prompt
            prompt_parts = []
            
            # Start with animal type
            prompt_parts.append(f"{animal_type.capitalize()} print pattern")
            
            # Add density if available
            if layout and layout != "balanced":
                prompt_parts.append(f"with {layout}")
            
            # Add pattern details
            if pattern_detail:
                prompt_parts.append(f"featuring {pattern_detail}")
            
            # Add color information
            prompt_parts.append(f"in {color_combo}")
            
            # Add style if available
            if style:
                prompt_parts.append(f"with {style} styling")
            
            # Add background
            prompt_parts.append(f"on a {background} background")
            
            # Add specific details if available
            if specific_details:
                prompt_parts.append(f"with {', '.join(specific_details[:2])}")
            
            # Combine all parts
            final_prompt = ". ".join([p.strip() for p in prompt_parts if p.strip()])
            if not final_prompt.endswith('.'):
                final_prompt += '.'
            
            # Capitalize first letter
            final_prompt = final_prompt[0].upper() + final_prompt[1:]
            
            return {
                "final_prompt": final_prompt,
                "components": {
                    "pattern_type": f"{animal_type} print",
                    "pattern_detail": pattern_detail,
                    "color_combo": color_combo,
                    "style": style,
                    "background": background,
                    "specific_details": specific_details
                },
                "completeness_score": min(1.0, len(prompt_parts) / 5)
            }
        
        except Exception as e:
            logger.error(f"Error building animal print prompt: {e}")
            return self._build_generic_prompt(pattern_type, elements, colors, background, layout, scale, texture, specific_details)
    
    def _build_geometric_prompt(self, pattern_type, elements, colors, background, layout, scale, texture, specific_details):
        """Build a detailed geometric pattern prompt"""
        # Implementation similar to other pattern types
        # ...
        
    def _build_abstract_prompt(self, pattern_type, elements, colors, background, layout, scale, texture, specific_details):
        """Build a detailed abstract pattern prompt"""
        # Implementation similar to other pattern types
        # ...
    
    def _build_generic_prompt(self, pattern_type, elements, colors, background, layout, scale, texture, specific_details):
        """Build a generic pattern prompt when specific category can't be determined"""
        try:
            # Extract element names
            element_names = [e['name'] for e in elements]
            
            # Build the prompt
            prompt_parts = []
            
            # Start with pattern type
            prompt_parts.append(f"{pattern_type.capitalize() if pattern_type else 'Decorative'} pattern")
            
            # Add density if available
            if layout and layout != "balanced":
                prompt_parts.append(f"with {layout}")
            
            # Add elements
            if element_names:
                prompt_parts.append(f"featuring {', '.join(element_names[:3])}")
            
            # Add color information
            if colors:
                prompt_parts.append(f"in shades of {', '.join(colors[:2])}")
            
            # Add background
            prompt_parts.append(f"on a {background} background")
            
            # Add specific details if available
            if specific_details:
                prompt_parts.append(f"with {', '.join(specific_details[:2])}")
            
            # Combine all parts
            final_prompt = ". ".join([p.strip() for p in prompt_parts if p.strip()])
            if not final_prompt.endswith('.'):
                final_prompt += '.'
            
            # Capitalize first letter
            final_prompt = final_prompt[0].upper() + final_prompt[1:]
            
            return {
                "final_prompt": final_prompt,
                "components": {
                    "pattern_type": pattern_type,
                    "elements": element_names,
                    "colors": colors,
                    "background": background,
                    "specific_details": specific_details
                },
                "completeness_score": min(1.0, len(prompt_parts) / 5)
            }
        
        except Exception as e:
            logger.error(f"Error building generic prompt: {e}")
            return {
                "final_prompt": f"Pattern with {pattern_type if pattern_type else 'various'} elements.",
                "components": {},
                "completeness_score": 0.3
            } 