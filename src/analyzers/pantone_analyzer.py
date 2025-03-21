import os
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class PantoneAnalyzer:
    """Class to handle RGB to Pantone color conversion"""
    
    def __init__(self, catalogs_dir: str = None):
        """Initialize with catalogs directory"""
        self.catalogs_dir = catalogs_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "catalogs")
        logger.info(f"Initializing PantoneAnalyzer with catalogs directory: {self.catalogs_dir}")
        
        # Ensure the catalogs directory exists
        os.makedirs(self.catalogs_dir, exist_ok=True)
        
        self.catalogs = {}
        self.load_catalogs()
        
        # Log the loaded catalogs
        if self.catalogs:
            logger.info(f"Loaded {len(self.catalogs)} catalogs: {', '.join(self.catalogs.keys())}")
        else:
            logger.warning("No catalogs were loaded")
    
    def load_catalogs(self) -> None:
        """Load all available Pantone catalogs"""
        try:
            catalog_files = [f for f in os.listdir(self.catalogs_dir) if f.endswith('.cat')]
            for catalog_file in catalog_files:
                catalog_path = os.path.join(self.catalogs_dir, catalog_file)
                catalog_name = os.path.splitext(catalog_file)[0]
                self.load_catalog(catalog_path, catalog_name)
        except Exception as e:
            logger.error(f"Error loading catalogs: {str(e)}")
    
    def load_catalog(self, catalog_path: str, catalog_name: str) -> bool:
        """
        Load a single Pantone catalog file (.cat format)
        
        Args:
            catalog_path: Path to the catalog file
            catalog_name: Name to identify the catalog
            
        Returns:
            bool: True if catalog was loaded successfully
        """
        try:
            logger.info(f"Loading catalog file: {catalog_path}")
            if not os.path.exists(catalog_path):
                logger.error(f"Catalog file not found: {catalog_path}")
                return False
                
            colors = []
            valid_lines = 0
            invalid_lines = 0
            
            # Try different encodings if needed
            encodings = ['utf-8', 'latin-1', 'cp1252']
            file_content = None
            
            # First, try to read the entire file content with different encodings
            for encoding in encodings:
                try:
                    with open(catalog_path, 'r', encoding=encoding, errors='ignore') as f:
                        file_content = f.read()
                    # If we successfully read the file, break the encoding loop
                    break
                except UnicodeDecodeError:
                    # Try the next encoding
                    logger.warning(f"Failed to decode with {encoding}, trying next encoding")
                    continue
            
            if not file_content:
                logger.error(f"Failed to read catalog file with any encoding: {catalog_path}")
                return False
            
            # Clean up the file content to remove any non-printable characters
            import string
            printable = set(string.printable)
            file_content = ''.join(c for c in file_content if c in printable)
            
            # Process the file line by line
            lines = file_content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                # Skip comments and empty lines
                if not line.strip() or line.strip().startswith('#'):
                    continue
                
                # Try to extract color information
                # Format is typically: NAME R G B
                try:
                    # First, try to find RGB values at the end of the line
                    rgb_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)$', line.strip())
                    if rgb_match:
                        r = int(rgb_match.group(1))
                        g = int(rgb_match.group(2))
                        b = int(rgb_match.group(3))
                        
                        # Validate RGB values
                        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                            logger.warning(f"Invalid RGB values in line {line_num}: {r},{g},{b}")
                            invalid_lines += 1
                            continue
                        
                        # Name is everything before the RGB values
                        name_part = line[:rgb_match.start()].strip()
                        
                        # Add to colors list
                        colors.append({
                            'name': name_part,
                            'rgb': [r, g, b],
                            'hex': f'#{r:02x}{g:02x}{b:02x}'
                        })
                        valid_lines += 1
                    else:
                        # Try to find RGB values with slash format (e.g., "RGB 252/249/81")
                        rgb_slash_match = re.search(r'RGB\s+(\d+)/(\d+)/(\d+)', line.strip())
                        if rgb_slash_match:
                            r = int(rgb_slash_match.group(1))
                            g = int(rgb_slash_match.group(2))
                            b = int(rgb_slash_match.group(3))
                            
                            # Validate RGB values
                            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                                logger.warning(f"Invalid RGB values in line {line_num}: {r},{g},{b}")
                                invalid_lines += 1
                                continue
                            
                            # Extract the name - typically before "RGB" or at the beginning of the line
                            name_parts = line[:rgb_slash_match.start()].strip().split()
                            # Take the first part as the name (usually a code like "12-0645")
                            name_part = name_parts[0] if name_parts else f"Color-{line_num}"
                            
                            # Add to colors list
                            colors.append({
                                'name': name_part,
                                'rgb': [r, g, b],
                                'hex': f'#{r:02x}{g:02x}{b:02x}'
                            })
                            valid_lines += 1
                        else:
                            # Try to find TCXRGB format (e.g., "12-0645 TCXRGB 252/249/81")
                            tcx_match = re.search(r'TCXRGB\s+(\d+)/(\d+)/(\d+)', line.strip())
                            if tcx_match:
                                r = int(tcx_match.group(1))
                                g = int(tcx_match.group(2))
                                b = int(tcx_match.group(3))
                                
                                # Validate RGB values
                                if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                                    logger.warning(f"Invalid RGB values in line {line_num}: {r},{g},{b}")
                                    invalid_lines += 1
                                    continue
                                
                                # Extract the name - typically before "TCXRGB"
                                name_parts = line[:tcx_match.start()].strip().split()
                                # Take the first part as the name (usually a code like "12-0645")
                                name_part = name_parts[0] if name_parts else f"Color-{line_num}"
                                
                                # Add to colors list
                                colors.append({
                                    'name': name_part,
                                    'rgb': [r, g, b],
                                    'hex': f'#{r:02x}{g:02x}{b:02x}'
                                })
                                valid_lines += 1
                            else:
                                # Try alternative format: split by whitespace and take last 3 as RGB
                                parts = re.split(r'\s+', line.strip())
                                if len(parts) >= 4:
                                    try:
                                        r = int(parts[-3])
                                        g = int(parts[-2])
                                        b = int(parts[-1])
                                        
                                        # Validate RGB values
                                        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                                            logger.warning(f"Invalid RGB values in line {line_num}: {r},{g},{b}")
                                            invalid_lines += 1
                                            continue
                                        
                                        # Name is everything before the RGB values
                                        name = ' '.join(parts[:-3])
                                        
                                        # Add to colors list
                                        colors.append({
                                            'name': name,
                                            'rgb': [r, g, b],
                                            'hex': f'#{r:02x}{g:02x}{b:02x}'
                                        })
                                        valid_lines += 1
                                    except ValueError as ve:
                                        logger.warning(f"Error parsing line {line_num}: {line.strip()} - {str(ve)}")
                                        invalid_lines += 1
                                        continue
                                else:
                                    logger.warning(f"Line {line_num} does not have enough parts: {line.strip()}")
                                    invalid_lines += 1
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {line.strip()} - {str(e)}")
                    invalid_lines += 1
            
            if colors:
                # If we're reloading an existing catalog, replace it
                if catalog_name in self.catalogs:
                    self.catalogs[catalog_name] = colors
                    logger.info(f"Replaced existing catalog: {catalog_name} with {len(colors)} colors (valid: {valid_lines}, invalid: {invalid_lines})")
                else:
                    self.catalogs[catalog_name] = colors
                    logger.info(f"Loaded new catalog: {catalog_name} with {len(colors)} colors (valid: {valid_lines}, invalid: {invalid_lines})")
                return True
            else:
                logger.warning(f"No colors found in catalog: {catalog_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading catalog {catalog_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_available_catalogs(self) -> List[str]:
        """Get list of available catalog names"""
        return list(self.catalogs.keys())
    
    def get_catalog_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about available catalogs
        
        Returns:
            List of dictionaries with catalog information
        """
        result = []
        for catalog_name, colors in self.catalogs.items():
            catalog_path = os.path.join(self.catalogs_dir, f"{catalog_name}.cat")
            file_size = os.path.getsize(catalog_path) if os.path.exists(catalog_path) else 0
            
            result.append({
                "name": catalog_name,
                "colors_count": len(colors),
                "file_size": file_size,
                "file_path": catalog_path
            })
            
        return sorted(result, key=lambda x: x["name"])
    
    def rgb_to_pantone(self, rgb: List[int], catalog_name: str = None) -> Dict[str, Any]:
        """
        Convert RGB color to closest Pantone color
        
        Args:
            rgb: RGB color as [r, g, b] list
            catalog_name: Name of catalog to use (if None, uses all catalogs)
            
        Returns:
            Dictionary with Pantone color information
        """
        if not self.catalogs:
            return {"error": "No Pantone catalogs loaded"}
        
        catalogs_to_check = [catalog_name] if catalog_name and catalog_name in self.catalogs else self.catalogs.keys()
        
        closest_color = None
        min_distance = float('inf')
        source_catalog = None
        
        for cat_name in catalogs_to_check:
            for color in self.catalogs[cat_name]:
                # Calculate color distance using Euclidean distance
                pantone_rgb = color['rgb']
                distance = np.sqrt(
                    (rgb[0] - pantone_rgb[0])**2 + 
                    (rgb[1] - pantone_rgb[1])**2 + 
                    (rgb[2] - pantone_rgb[2])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color
                    source_catalog = cat_name
        
        if closest_color:
            return {
                "pantone_name": closest_color['name'],
                "pantone_rgb": closest_color['rgb'],
                "pantone_hex": closest_color['hex'],
                "source_catalog": source_catalog,
                "distance": float(min_distance),
                "match_quality": max(0, min(100, 100 - (min_distance / 4.42)))  # Normalize to 0-100%
            }
        else:
            return {"error": "No matching Pantone color found"}
    
    def analyze_colors(self, colors: List[Dict], catalog_name: str = None) -> List[Dict]:
        """
        Analyze a list of RGB colors and find Pantone matches
        
        Args:
            colors: List of color dictionaries with 'rgb' key
            catalog_name: Name of catalog to use (optional)
            
        Returns:
            List of colors with added Pantone information
        """
        result = []
        
        for color in colors:
            if 'rgb' not in color:
                result.append(color)
                continue
                
            rgb = color['rgb']
            pantone_info = self.rgb_to_pantone(rgb, catalog_name)
            
            # Create a copy of the original color and add Pantone info
            color_with_pantone = color.copy()
            color_with_pantone['pantone'] = pantone_info
            
            result.append(color_with_pantone)
            
        return result
    
    def upload_catalog(self, file_path: str, catalog_name: str = None) -> Dict[str, Any]:
        """
        Upload and process a new Pantone catalog
        
        Args:
            file_path: Path to the uploaded catalog file
            catalog_name: Optional name for the catalog (defaults to filename)
            
        Returns:
            Dictionary with status information
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return {"success": False, "error": "File not found"}
                
            if not file_path.endswith('.cat'):
                logger.error(f"Invalid file format: {file_path}")
                return {"success": False, "error": "Invalid file format. Only .cat files are supported"}
            
            # Use filename as catalog name if not provided
            if not catalog_name:
                catalog_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Ensure catalog name is valid
            catalog_name = re.sub(r'[^\w\-\.]', '_', catalog_name)
            
            # Copy file to catalogs directory
            import shutil
            dest_path = os.path.join(self.catalogs_dir, f"{catalog_name}.cat")
            
            # Ensure the catalogs directory exists
            os.makedirs(self.catalogs_dir, exist_ok=True)
            
            # Check if destination file already exists
            if os.path.exists(dest_path):
                # If the files are the same, just reload the catalog
                try:
                    if os.path.samefile(file_path, dest_path):
                        logger.info(f"File {file_path} and {dest_path} are the same file. Reloading catalog.")
                        # Reload the catalog
                        success = self.load_catalog(dest_path, catalog_name)
                        if success:
                            logger.info(f"Successfully reloaded catalog: {catalog_name} with {len(self.catalogs[catalog_name])} colors")
                            return {
                                "success": True, 
                                "catalog_name": catalog_name,
                                "colors_count": len(self.catalogs[catalog_name]),
                                "message": "Catalog reloaded successfully"
                            }
                        else:
                            logger.error(f"Failed to reload catalog: {catalog_name}")
                            return {"success": False, "error": "Failed to reload catalog"}
                except OSError:
                    # If samefile check fails, continue with the upload
                    pass
                
                # If the files are different, create a unique name
                import time
                timestamp = int(time.time())
                catalog_name = f"{catalog_name}_{timestamp}"
                dest_path = os.path.join(self.catalogs_dir, f"{catalog_name}.cat")
                logger.info(f"File with same name exists. Creating unique name: {catalog_name}")
            
            logger.info(f"Copying catalog from {file_path} to {dest_path}")
            shutil.copy2(file_path, dest_path)
            
            # Load the catalog
            logger.info(f"Loading catalog: {dest_path}")
            success = self.load_catalog(dest_path, catalog_name)
            
            if success:
                logger.info(f"Successfully loaded catalog: {catalog_name} with {len(self.catalogs[catalog_name])} colors")
                return {
                    "success": True, 
                    "catalog_name": catalog_name,
                    "colors_count": len(self.catalogs[catalog_name])
                }
            else:
                logger.error(f"Failed to load catalog: {catalog_name}")
                return {"success": False, "error": "Failed to load catalog"}
                
        except Exception as e:
            logger.error(f"Error uploading catalog: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)} 