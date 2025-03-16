/**
 * Gets the thumbnail URL for an image
 * @param {string} path - The thumbnail path
 * @returns {string} The full thumbnail URL
 */
export const getThumbnailUrl = (path) => {
  if (!path) {
    console.log('Warning: Empty thumbnail path');
    return '';
  }
  return `http://localhost:8000/api/thumbnails/${path.split('/').pop()}`
}

/**
 * Extracts the image name from a path
 * @param {string} path - The image path
 * @returns {string} The image filename
 */
export const getImageName = (path) => {
  if (!path) return 'Unknown'
  return path.split('/').pop()
}

/**
 * Truncates a prompt text for display in the gallery
 * @param {string|object} prompt - The prompt text or object
 * @returns {string} The processed prompt text
 */
export const truncatePrompt = (prompt) => {
  return getPromptText(prompt, true);
}

/**
 * Gets the full prompt text from a prompt object or string
 * @param {string|object} prompt - The prompt text or object
 * @param {boolean} truncate - Whether to truncate the text
 * @returns {string} The processed prompt text
 */
export const getPromptText = (prompt, truncate = false) => {
  if (!prompt) return 'No description available';
  
  // Handle both string and object formats
  const promptText = typeof prompt === 'string' 
    ? prompt 
    : (prompt.final_prompt || 'No description available');
    
  if (truncate && promptText.length > 100) {
    return promptText.substring(0, 100) + '...';
  }
  
  return promptText;
}

/**
 * Calculates the aspect ratio of an image
 * @param {object} dimensions - The image dimensions object with width and height
 * @returns {string} The aspect ratio as a string (e.g., "16:9")
 */
export const calculateAspectRatio = (dimensions) => {
  if (!dimensions?.width || !dimensions?.height) return 'N/A'
  const gcd = (a, b) => b ? gcd(b, a % b) : a
  const divisor = gcd(dimensions.width, dimensions.height)
  return `${dimensions.width/divisor}:${dimensions.height/divisor}`
} 