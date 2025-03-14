from typing import Dict
import numpy as np
import colorsys
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    def __init__(self, max_clusters: int = 10):
        self.max_clusters = max_clusters
        self.color_references = {
            'Red': (255, 0, 0),
            'Dark Red': (139, 0, 0),
            'Pink': (255, 192, 203),
            'Orange': (255, 165, 0),
            'Yellow': (255, 255, 0),
            'Green': (0, 128, 0),
            'Light Green': (144, 238, 144),
            'Blue': (0, 0, 255),
            'Light Blue': (173, 216, 230),
            'Purple': (128, 0, 128),
            'Brown': (165, 42, 42),
            'Gray': (128, 128, 128),
            'Light Gray': (211, 211, 211),
            'Black': (0, 0, 0),
            'White': (255, 255, 255),
            'Beige': (245, 245, 220),
            'Navy': (0, 0, 128),
            'Teal': (0, 128, 128),
            'Maroon': (128, 0, 0),
            'Gold': (255, 215, 0)
        }

    def analyze_colors(self, image: np.ndarray) -> Dict:
        """Analyze colors in the image."""
        try:
            logger.info("Starting color analysis...")
            
            # Convert image to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]

            # Reshape image for clustering
            pixels = image.reshape(-1, 3)
            
            # Determine optimal number of clusters
            n_colors = min(max(3, len(np.unique(pixels, axis=0)) // 100), self.max_clusters)
            logger.info(f"Using {n_colors} clusters for color analysis")
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers and proportions
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            proportions = counts / len(labels)

            # Process colors
            dominant_colors = []
            for color, proportion in zip(colors, proportions):
                rgb = color.astype(int)
                hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                color_name = self._get_color_name(rgb)
                
                dominant_colors.append({
                    'rgb': rgb.tolist(),
                    'hex': hex_color,
                    'name': color_name,
                    'proportion': float(proportion)
                })

            # Sort by proportion
            dominant_colors.sort(key=lambda x: x['proportion'], reverse=True)
            
            logger.info(f"Found {len(dominant_colors)} dominant colors")
            for color in dominant_colors:
                logger.info(f"Color: {color['name']}, Proportion: {color['proportion']:.2%}")

            return {
                'dominant_colors': dominant_colors,
                'overall_brightness': float(np.mean(image) / 255.0),
                'color_contrast': float(np.std(image) / 255.0)
            }

        except Exception as e:
            logger.error(f"Error in color analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _get_color_name(self, rgb):
        """Find closest color name."""
        min_distance = float('inf')
        closest_color = 'Unknown'
        
        hsv = self._rgb_to_hsv(rgb)
        
        for name, ref_rgb in self.color_references.items():
            ref_hsv = self._rgb_to_hsv(ref_rgb)
            
            h_diff = min(abs(hsv[0] - ref_hsv[0]), 1 - abs(hsv[0] - ref_hsv[0])) * 2.0
            s_diff = abs(hsv[1] - ref_hsv[1])
            v_diff = abs(hsv[2] - ref_hsv[2])
            
            distance = (h_diff * 2) + s_diff + v_diff
            
            if distance < min_distance:
                min_distance = distance
                closest_color = name

        return closest_color

    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV."""
        return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255) 