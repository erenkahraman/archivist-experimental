<template>
  <div v-if="props.selectedImage" class="modal-container" :class="{ 'is-open': isOpen }">
    <div class="modal-backdrop" @click="closeModal"></div>
    <div class="modal">
      <div class="modal-header">
        <h3>{{ getImageTitle() }}</h3>
        <button class="close-button" @click="closeModal">×</button>
      </div>
      <div class="modal-content">
        <!-- Image display - always visible -->
        <div class="image-section">
          <div class="image-container" :class="{ 'is-loading': isLoading }">
            <img
              id="modal-image"
              :src="getImageUrl(props.selectedImage.path || props.selectedImage.file_path || props.selectedImage.image_path || props.selectedImage.original_path)"
              :alt="getImageName(props.selectedImage.file_path || props.selectedImage.image_path || props.selectedImage.id)"
              class="main-image"
              @load="handleImageLoaded"
              @error="handleImageError"
              :class="{ 'hidden': isLoading }"
              ref="imageElement"
            >
            <div v-if="imageLoadError" class="image-error">
              <div class="error-message">{{ errorMessage }}</div>
              <button @click="tryLoadThumbnail" class="try-thumbnail-btn">
                Try Thumbnail
              </button>
            </div>
            <div v-if="isLoading" class="loading-spinner">
              <div class="spinner"></div>
              <p>Loading image...</p>
            </div>
          </div>
        </div>
        
        <!-- Tabs for metadata -->
        <div class="tabs">
          <button
            v-for="tab in tabs"
            :key="tab.id"
            :class="['tab-button', { active: activeTab === tab.id }]"
            @click="activeTab = tab.id"
          >
            {{ tab.label }}
          </button>
        </div>
        
        <div class="tab-content">
          <!-- Patterns Tab -->
          <div v-if="activeTab === 'patterns'" class="tab-pane">
            <!-- Primary Pattern -->
            <div class="pattern-section" v-if="getPrimaryPattern()">
              <h3 class="section-title">Primary Pattern</h3>
              <div class="pattern-primary">
                {{ getPrimaryPattern() }}
                <span class="inline-confidence" v-if="getPatternConfidence() !== null">
                  {{ formatConfidence(getPatternConfidence()) }}
                </span>
              </div>
            </div>
            
            <!-- Secondary Patterns -->
            <div class="pattern-section" v-if="hasSecondaryPatterns">
              <h3 class="section-title">Secondary Patterns</h3>
              <p class="confidence-note">Percentages indicate pattern confidence level</p>
              <div class="tag-container">
                <span 
                  v-for="pattern in getSecondaryPatterns()" 
                  :key="typeof pattern === 'string' ? pattern : JSON.stringify(pattern)"
                  class="tag"
                >
                  {{ typeof pattern === 'string' ? pattern : 
                     (pattern.name ? `${pattern.name} (${Math.round((pattern.confidence || 0) * 100)}%)` : JSON.stringify(pattern)) }}
                </span>
              </div>
            </div>
            
            <!-- Pattern Description -->
            <div class="pattern-section" v-if="getPatternDescription()">
              <h3 class="section-title">Pattern Description</h3>
              <div class="image-data">{{ getPatternDescription() }}</div>
            </div>

            <!-- Style Keywords -->
            <div class="pattern-section" v-if="hasStyleKeywords">
              <h3 class="section-title">Style Keywords</h3>
              <div class="keyword-tags">
                <span 
                  v-for="(keyword, index) in getStyleKeywords()" 
                  :key="index"
                  class="keyword-tag"
                >
                  {{ keyword }}
                </span>
              </div>
            </div>
          </div>
          
          <!-- Colors Tab -->
          <div v-if="activeTab === 'colors'" class="tab-pane">
            <!-- Colors debug info -->
            <div v-if="isDev" class="debug-info">
              <pre>{{ JSON.stringify(props.selectedImage.colors, null, 2) }}</pre>
            </div>

            <!-- Dominant Color -->
            <div class="color-section" v-if="getDominantColor()">
              <h3 class="section-title">Primary Color</h3>
              <div class="color-display">
                <div 
                  class="color-preview" 
                  :style="{ backgroundColor: extractHexColor(getDominantColor()) }"
                ></div>
                <div class="color-info">
                  <span class="color-value">{{ extractHexColor(getDominantColor()) }}</span>
                  <span class="color-name" v-if="extractColorName(getDominantColor())">{{ extractColorName(getDominantColor()) }}</span>
                  <span class="color-proportion" v-if="extractProportion(getDominantColor())">{{ formatProportion(extractProportion(getDominantColor())) }}</span>
                </div>
              </div>
            </div>
            
            <!-- Color Palette -->
            <div class="color-section" v-if="getColorPalette().length > 0">
              <h3 class="section-title">Color Palette</h3>
              <div class="color-palette-grid">
                <div 
                  v-for="(color, idx) in getColorPalette()" 
                  :key="idx"
                  class="palette-item"
                >
                  <div class="color-preview" :style="{ backgroundColor: extractHexColor(color) }"></div>
                  <div class="color-info">
                    <span class="color-value">{{ extractHexColor(color) }}</span>
                    <span class="color-name" v-if="extractColorName(color)">{{ extractColorName(color) }}</span>
                    <span class="color-proportion" v-if="extractProportion(color)">{{ formatProportion(extractProportion(color)) }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Details Tab -->
          <div v-if="activeTab === 'details'" class="tab-pane">
            <div class="details-section">
              <h3 class="section-title">File Information</h3>
              <table class="details-table">
                <tr>
                  <td class="details-label">Filename:</td>
                  <td>{{ getImageName(props.selectedImage.file_path || props.selectedImage.image_path || props.selectedImage.id) }}</td>
                </tr>
                <tr v-if="props.selectedImage.metadata?.width && props.selectedImage.metadata?.height">
                  <td class="details-label">Dimensions:</td>
                  <td>{{ props.selectedImage.metadata.width }} × {{ props.selectedImage.metadata.height }} px</td>
                </tr>
                <tr v-if="props.selectedImage.created_at">
                  <td class="details-label">Added:</td>
                  <td>{{ formatDate(props.selectedImage.created_at) }}</td>
                </tr>
                <tr>
                  <td class="details-label">Path:</td>
                  <td class="file-path">{{ props.selectedImage.file_path || props.selectedImage.image_path || props.selectedImage.thumbnail_path }}</td>
                </tr>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps, ref, computed, onMounted } from 'vue'
import { useImageStore } from '../../stores/imageStore'

const props = defineProps({
  selectedImage: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['close'])
const imageStore = useImageStore()

// Use the API_BASE_URL from the store for consistency
const isDev = false; // Set to true when debugging is needed

// Image loading state
const imageElement = ref(null)
const isLoading = ref(true)
const imageLoadError = ref(false)
const useThumbnail = ref(false)
const errorMessage = ref('')

// Tab management
const tabs = [
  { id: 'patterns', label: 'Patterns' },
  { id: 'colors', label: 'Colors' },
  { id: 'details', label: 'Details' }
]
const activeTab = ref('patterns')

// Data availability checks
const hasSecondaryPatterns = computed(() => {
  // Check search results format
  if (props.selectedImage.pattern?.secondary && 
      Array.isArray(props.selectedImage.pattern.secondary) && 
      props.selectedImage.pattern.secondary.length > 0) {
    return true;
  }
  
  // Check regular format
  return props.selectedImage.patterns?.secondary_patterns && 
    (Array.isArray(props.selectedImage.patterns.secondary_patterns) || 
    typeof props.selectedImage.patterns.secondary_patterns === 'object')
})

const hasStyleKeywords = computed(() => {
  return !!getStyleKeywords()?.length;
});

// Core data functions
const getImageTitle = () => {
  // If there's a filename, use that first
  if (props.selectedImage.filename) {
    return props.selectedImage.filename;
  }
  
  // Otherwise try to use one of the path fields or ID
  return getImageName(props.selectedImage.file_path || 
                      props.selectedImage.image_path || 
                      props.selectedImage.original_path || 
                      props.selectedImage.thumbnail_path || 
                      props.selectedImage.id);
}

const getImageUrl = (path) => {
  if (!path) return '';
  
  // If it's already a full URL, use it
  if (path.startsWith('http')) {
    return path;
  }
  
  // Get just the filename
  const filename = imageStore.getFileName(path);
  
  // If the path contains 'uploads/', it's a reference to the uploads directory
  if (path.includes('uploads/')) {
    // Construct API URL for full image
    return `${imageStore.API_BASE_URL}/images/${filename}`;
  }
  
  // For paths that have thumbnail in them, use the thumbnail endpoint
  if (path.includes('thumbnail')) {
    return `${imageStore.API_BASE_URL}/thumbnails/${filename}`;
  }
  
  // Default to full image endpoint
  return `${imageStore.API_BASE_URL}/images/${filename}`;
};

const getThumbnailUrl = (path) => {
  if (!path) return '';
  
  // Get just the filename
  const filename = imageStore.getFileName(path);
  
  // Construct API URL
  return `${imageStore.API_BASE_URL}/thumbnails/${filename}`;
};

const getImageName = (path) => {
  if (!path) return 'Unknown Image'
  return path.split('/').pop()
}

const getPromptText = (prompt) => {
  if (!prompt) return 'No description available'
  return typeof prompt === 'string' ? prompt : (prompt.final_prompt || 'No description available')
}

// Pattern handling
const getPrimaryPattern = () => {
  if (!props.selectedImage) return null;
  
  // Regular format: props.selectedImage.patterns.primary_pattern
  if (props.selectedImage.patterns?.primary_pattern) {
    const pattern = props.selectedImage.patterns.primary_pattern;
    if (typeof pattern === 'object' && pattern.name) {
      return pattern.name;
    }
    return pattern;
  }
  
  // Search results format: props.selectedImage.pattern.primary
  if (props.selectedImage.pattern?.primary) {
    return props.selectedImage.pattern.primary;
  }
  
  return null;
}

const getPatternConfidence = () => {
  if (!props.selectedImage) return null;
  
  // Regular format
  if (props.selectedImage.patterns?.pattern_confidence !== undefined) {
    return props.selectedImage.patterns.pattern_confidence;
  }
  
  if (props.selectedImage.patterns?.category_confidence !== undefined) {
    return props.selectedImage.patterns.category_confidence;
  }
  
  // Search results format
  if (props.selectedImage.pattern?.confidence !== undefined) {
    return props.selectedImage.pattern.confidence;
  }
  
  return null;
}

const formatConfidence = (confidence) => {
  if (confidence === null || confidence === undefined) return '';
  return `${Math.round(confidence * 100)}%`;
}

const getPatternDescription = () => {
  if (!props.selectedImage) return null;
  
  // Regular format
  if (props.selectedImage.patterns?.prompt) {
    return getPromptText(props.selectedImage.patterns.prompt);
  }
  
  // Search results format
  if (props.selectedImage.prompt) {
    return getPromptText(props.selectedImage.prompt);
  }
  
  return null;
}

const getSecondaryPatterns = () => {
  if (!props.selectedImage) return [];
  
  // Search results format
  if (props.selectedImage.pattern?.secondary && Array.isArray(props.selectedImage.pattern.secondary)) {
    return props.selectedImage.pattern.secondary;
  }
  
  // Regular format
  if (!props.selectedImage.patterns?.secondary_patterns) return []
  
  // Handle when it's an array of strings
  if (Array.isArray(props.selectedImage.patterns.secondary_patterns)) {
    return props.selectedImage.patterns.secondary_patterns
  }
  
  // Handle when it's an object (JSON)
  if (typeof props.selectedImage.patterns.secondary_patterns === 'object') {
    try {
      let patterns = []
      const secondaryPatterns = props.selectedImage.patterns.secondary_patterns
      
      // Case 1: Array-like object with numeric keys (coming from JSON.parse of array)
      if (Object.keys(secondaryPatterns).some(key => !isNaN(parseInt(key)))) {
        patterns = Object.values(secondaryPatterns)
      } 
      // Case 2: Object with pattern names as keys
      else {
        patterns = Object.entries(secondaryPatterns).map(([key, value]) => {
          return { name: key, confidence: value }
        })
      }
      
      // Process each pattern object to extract name and confidence
      return patterns
        .map(pattern => {
          if (!pattern) return null
          
          // If it's already a string, return as is
          if (typeof pattern === 'string') return pattern
          
          // Extract name and confidence from object
          let name = ''
          let confidence = null
          
          if (typeof pattern === 'object') {
            name = pattern.name || ''
            confidence = pattern.confidence
          }
          
          // Format with confidence as percentage if available
          if (name && confidence !== undefined && confidence !== null) {
            return `${name} (${Math.round(confidence * 100)}%)`
          } else {
            return name || null
          }
        })
        .filter(Boolean)
        .sort((a, b) => {
          // Sort by confidence (extract percentage)
          const confA = a.match(/\((\d+)%\)/)
          const confB = b.match(/\((\d+)%\)/)
          if (confA && confB) {
            return parseInt(confB[1]) - parseInt(confA[1])
          }
          return 0
        })
      
    } catch (e) {
      console.error('Error parsing secondary patterns:', e)
      // Last resort fallback
      return Object.values(props.selectedImage.patterns.secondary_patterns)
    }
  }
  
  return []
}

const getStyleKeywords = () => {
  if (!props.selectedImage) return [];
  
  // Check different possible locations for style keywords
  if (props.selectedImage.style_keywords) {
    return Array.isArray(props.selectedImage.style_keywords) 
      ? props.selectedImage.style_keywords 
      : [props.selectedImage.style_keywords];
  }
  
  if (props.selectedImage.patterns?.style_keywords) {
    return Array.isArray(props.selectedImage.patterns.style_keywords) 
      ? props.selectedImage.patterns.style_keywords 
      : [props.selectedImage.patterns.style_keywords];
  }
  
  if (props.selectedImage.metadata?.style_keywords) {
    return Array.isArray(props.selectedImage.metadata.style_keywords) 
      ? props.selectedImage.metadata.style_keywords 
      : [props.selectedImage.metadata.style_keywords];
  }
  
  return [];
};

// Color handling
const getDominantColor = () => {
  // Check for direct colors array in the search results
  if (props.selectedImage.colors && Array.isArray(props.selectedImage.colors)) {
    // If colors is an array (common in search results), use the first one
    if (props.selectedImage.colors.length > 0) {
      return props.selectedImage.colors[0];
    }
  }
  
  // Try various possible locations for color data
  if (props.selectedImage.colors?.dominant_colors && 
      Array.isArray(props.selectedImage.colors.dominant_colors)) {
    return props.selectedImage.colors.dominant_colors[0];
  }
  
  if (props.selectedImage.colors?.dominant_color) {
    return props.selectedImage.colors.dominant_color;
  }
  
  // For search results, it might be in metadata or different structures
  if (props.selectedImage.metadata?.colors?.dominant_color) {
    return props.selectedImage.metadata.colors.dominant_color;
  }
  
  // Try palette as fallback
  if (props.selectedImage.colors?.palette && props.selectedImage.colors.palette.length > 0) {
    return props.selectedImage.colors.palette[0];
  }
  
  if (props.selectedImage.patterns?.colors?.palette && 
      props.selectedImage.patterns.colors.palette.length > 0) {
    return props.selectedImage.patterns.colors.palette[0];
  }
  
  if (props.selectedImage.metadata?.colors?.palette && 
      props.selectedImage.metadata.colors.palette.length > 0) {
    return props.selectedImage.metadata.colors.palette[0];
  }
  
  return null;
};

const getColorPalette = () => {
  // Check for direct colors array in the search results
  if (props.selectedImage.colors && Array.isArray(props.selectedImage.colors)) {
    return props.selectedImage.colors;
  }
  
  // Try various possible locations for palette data
  if (props.selectedImage.colors?.dominant_colors && 
      Array.isArray(props.selectedImage.colors.dominant_colors)) {
    return props.selectedImage.colors.dominant_colors;
  }
  
  if (props.selectedImage.colors?.palette && props.selectedImage.colors.palette.length > 0) {
    return props.selectedImage.colors.palette;
  }
  
  if (props.selectedImage.patterns?.colors?.palette && 
      props.selectedImage.patterns.colors.palette.length > 0) {
    return props.selectedImage.patterns.colors.palette;
  }
  
  if (props.selectedImage.metadata?.colors?.palette && 
      props.selectedImage.metadata.colors.palette.length > 0) {
    return props.selectedImage.metadata.colors.palette;
  }
  
  return [];
};

const extractHexColor = (color) => {
  if (!color) return '#cccccc'; // Default gray
  
  // If it's a simple string, return it directly
  if (typeof color === 'string') return color;
  
  // If it's an object, try to find the hex code
  if (typeof color === 'object') {
    // Direct hex property
    if (color.hex) return color.hex;
    if (color.color) return color.color;
    
    // Try to find RGB values
    if (color.rgb && Array.isArray(color.rgb) && color.rgb.length >= 3) {
      return `rgb(${color.rgb.join(', ')})`;
    }
  }
  
  // Last resort - try to extract from a stringified representation
  try {
    const colorStr = JSON.stringify(color);
    
    // Look for hex code pattern first
    const hexPattern = /#[0-9a-f]{6}/i;
    const foundHex = colorStr.match(hexPattern);
    if (foundHex) {
      return foundHex[0];
    }
    
    const hexMatch = colorStr.match(/"hex":"(#[0-9a-f]+)"/i);
    if (hexMatch && hexMatch[1]) {
      return hexMatch[1];
    }
    
    const colorMatch = colorStr.match(/"color":"(#[0-9a-f]+)"/i);
    if (colorMatch && colorMatch[1]) {
      return colorMatch[1];
    }
  } catch (e) {}
  
  return '#cccccc'; // Default gray
}

const extractColorName = (color) => {
  if (!color) return null;
  
  if (typeof color === 'object' && color.name) {
    return color.name;
  }
  
  try {
    const colorStr = JSON.stringify(color);
    const nameMatch = colorStr.match(/"name":"([^"]+)"/i);
    if (nameMatch && nameMatch[1]) {
      return nameMatch[1];
    }
  } catch (e) {}
  
  return null;
}

const extractRGB = (color) => {
  if (!color) return null;
  
  if (typeof color === 'object' && Array.isArray(color.rgb)) {
    return color.rgb.join(', ');
  }
  
  try {
    const colorStr = JSON.stringify(color);
    const rgbMatch = colorStr.match(/"rgb":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/i);
    if (rgbMatch && rgbMatch.length >= 4) {
      return `${rgbMatch[1]}, ${rgbMatch[2]}, ${rgbMatch[3]}`;
    }
  } catch (e) {}
  
  return null;
}

const extractProportion = (color) => {
  if (!color) return null;
  
  if (typeof color === 'object' && typeof color.proportion === 'number') {
    return color.proportion;
  }
  
  try {
    const colorStr = JSON.stringify(color);
    const proportionMatch = colorStr.match(/"proportion":([0-9.]+)/i);
    if (proportionMatch && proportionMatch[1]) {
      return parseFloat(proportionMatch[1]);
    }
  } catch (e) {}
  
  return null;
}

const formatProportion = (proportion) => {
  if (proportion === null || proportion === undefined) return '';
  return `${Math.round(proportion * 100)}%`;
}

// Utility functions
const formatDate = (dateString) => {
  if (!dateString) return 'Unknown'
  try {
    const date = new Date(dateString)
    return date.toLocaleString()
  } catch (e) {
    return dateString
  }
}

// Image loading handlers
const handleImageLoaded = () => {
  console.log('Image loaded successfully');
  isLoading.value = false;
  imageLoadError.value = false;
  errorMessage.value = '';
};

// Track thumbnail load attempts
const attemptCount = ref(0);

const handleImageError = (error) => {
  console.error('Error loading image:', error);
  console.log('Image that failed to load:', props.selectedImage);
  console.log('Attempted URL:', imageElement.value?.src);
  isLoading.value = false;
  imageLoadError.value = true;
  
  // If we've tried multiple times, just close the modal
  if (useThumbnail.value || attemptCount.value > 1) {
    console.log("Image seems to be deleted from the server, closing modal");
    
    // Close the modal
    setTimeout(() => {
      emit('close');
      
      // Clean the gallery to remove any invalid images
      imageStore.cleanGallery();
    }, 500);
    
    return;
  }
  
  // Try loading the thumbnail
  errorMessage.value = 'Failed to load image. Trying thumbnail instead...';
  
  if (props.selectedImage.thumbnail_path) {
    console.log('Attempting to load thumbnail instead:', props.selectedImage.thumbnail_path);
    tryLoadThumbnail();
  } else {
    console.log('No thumbnail available, trying to generate one from path:', props.selectedImage.path);
    tryLoadThumbnail();
  }
};

const tryLoadThumbnail = () => {
  // Try different sources for the thumbnail
  const thumbnailPath = props.selectedImage.thumbnail_path || 
                        (props.selectedImage.path && props.selectedImage.path.replace('uploads/', 'thumbnails/'));
  
  if (!thumbnailPath) {
    errorMessage.value = 'No thumbnail available for this image.';
    return;
  }
  
  // Increase attempt count
  attemptCount.value++;
  
  // Prevent infinite retries
  if (attemptCount.value > 2) {
    errorMessage.value = 'Failed to load both image and thumbnail. The image may no longer exist.';
    isLoading.value = false;
    
    // Close modal
    setTimeout(() => {
      emit('close');
      imageStore.cleanGallery();
    }, 1000);
    
    return;
  }
  
  useThumbnail.value = true;
  isLoading.value = true;
  
  // Log the source we're trying
  console.log('Trying to load thumbnail from:', thumbnailPath);
  
  // Change the image source to the thumbnail
  if (imageElement.value) {
    imageElement.value.src = getThumbnailUrl(thumbnailPath);
  }
};

// Modal state
const isOpen = ref(true)

const closeModal = () => {
  isOpen.value = false
  setTimeout(() => {
    emit('close')
  }, 300) // Wait for animation to complete
}

onMounted(() => {
  // Set image element for load handling
  imageElement.value = document.getElementById('modal-image');
  
  // Reset state for new image
  attemptCount.value = 0;
  useThumbnail.value = false;
  isLoading.value = true;
  imageLoadError.value = false;
  
  // Log initial image path being attempted
  const imagePath = props.selectedImage.path || 
                   props.selectedImage.file_path || 
                   props.selectedImage.image_path || 
                   props.selectedImage.original_path;
  console.log('Initial image path attempt:', imagePath);
  console.log('Using URL:', getImageUrl(imagePath));
  
  // Handle image loading events
  if (imageElement.value) {
    imageElement.value.onload = () => {
      console.log('Image loaded successfully via onload event');
      isLoading.value = false;
      imageLoadError.value = false;
      errorMessage.value = '';
    };
    
    imageElement.value.onerror = (error) => {
      console.error('Image load error via onerror event:', error);
      
      // If we're not already using the thumbnail, try to use it
      if (!useThumbnail.value) {
        tryLoadThumbnail();
      } else {
        // If we're already using the thumbnail and it failed, show an error
        isLoading.value = false;
        imageLoadError.value = true;
        errorMessage.value = 'Failed to load image.';
      }
    };
  }
});
</script>

<style scoped>
.modal-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.3s ease;
}

.modal-container.is-open {
  opacity: 1;
  pointer-events: all;
}

.modal-backdrop {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(5px);
}

.modal {
  position: relative;
  width: 95%;
  max-width: 1200px;
  max-height: 90vh;
  background-color: white;
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-xl);
  z-index: 1001;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-4);
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.close-button {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  color: #333;
  border: none;
  padding: 0;
  cursor: pointer;
  box-shadow: var(--shadow-md);
}

.close-button svg {
  width: 18px;
  height: 18px;
}

.close-button:hover {
  background-color: white;
  transform: scale(1.05);
}

.modal-layout {
  display: flex;
  height: 100%;
  max-height: 90vh;
}

.image-column {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: var(--space-4);
  border-right: 1px solid rgba(0, 0, 0, 0.1);
  background-color: rgba(255, 255, 255, 0.5);
}

.image-section {
  margin-bottom: var(--space-4);
  border-radius: var(--radius-lg);
  overflow: hidden;
  background-color: rgba(0, 0, 0, 0.02);
}

.image-container {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  min-height: 300px;
  max-height: 500px;
  overflow: hidden;
}

.main-image {
  max-width: 100%;
  max-height: 500px;
  object-fit: contain;
}

.image-name {
  font-family: var(--font-heading);
  font-size: 1.3rem;
  margin: 0;
  padding: var(--space-2) 0;
  text-align: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #333;
}

.image-loading, .image-error {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.8);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(79, 70, 229, 0.2);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s infinite linear;
  margin-bottom: var(--space-4);
}

.try-thumbnail-btn {
  background-color: var(--color-primary);
  color: white;
  border: none;
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-weight: 500;
  box-shadow: var(--shadow-sm);
  transition: all 0.2s;
}

.try-thumbnail-btn:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.details-column {
  width: 40%;
  min-width: 400px;
  max-width: 500px;
  display: flex;
  flex-direction: column;
  background-color: white;
}

/* Tabs */
.tabs-nav {
  display: flex;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  background-color: #f5f5f7;
}

.tab-button {
  flex: 1;
  padding: var(--space-3) var(--space-2);
  background: transparent;
  color: #555;
  border: none;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.tab-button.active {
  color: var(--color-primary);
  background-color: white;
}

.tab-button.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: var(--color-primary);
}

.tab-button:hover:not(.active) {
  background-color: rgba(0, 0, 0, 0.03);
}

.tab-content {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-4);
}

.tab-pane {
  animation: fadeIn 0.3s ease;
}

/* Pattern tab */
.pattern-section {
  margin-bottom: var(--space-5);
  padding: var(--space-3);
  background-color: #f9f9fb;
  border-radius: var(--radius-md);
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.confidence-note {
  font-size: 0.8rem;
  font-style: italic;
  color: #666;
  margin: var(--space-1) 0 var(--space-1) 0;
}

.pattern-primary {
  font-size: 1.2rem;
  font-weight: 600;
  color: #333;
  padding: var(--space-2) var(--space-3);
  background-color: rgba(79, 70, 229, 0.07);
  border-radius: var(--radius-md);
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  margin-top: var(--space-2);
}

.inline-confidence {
  margin-left: 8px;
  color: var(--color-primary);
  font-weight: 700;
  font-size: 1rem;
}

.section-title {
  font-family: var(--font-heading);
  font-size: 1rem;
  color: #555;
  margin-bottom: var(--space-2);
}

.tag-container {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin-top: var(--space-2);
}

.tag {
  display: inline-block;
  padding: var(--space-1) var(--space-2);
  background-color: rgba(79, 70, 229, 0.1);
  border-radius: var(--radius-full);
  font-size: 0.85rem;
  color: var(--color-primary);
  font-weight: 500;
}

.image-data {
  line-height: 1.6;
  color: #333;
  font-size: 0.95rem;
  white-space: pre-line;
  padding: var(--space-3);
  border-radius: var(--radius-md);
  background-color: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.05);
  margin-top: var(--space-2);
}

/* Colors tab */
.color-section {
  margin-bottom: var(--space-5);
  padding: var(--space-4);
  background-color: #f9f9fb;
  border-radius: var(--radius-md);
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.color-display {
  display: flex;
  align-items: center;
  margin-top: var(--space-3);
}

.color-preview {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  margin-right: var(--space-3);
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.color-info {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.color-value {
  font-weight: 600;
  font-size: 0.9rem;
}

.color-name {
  font-size: 0.85rem;
  color: var(--color-text-light);
}

.color-rgb {
  font-family: monospace;
  font-size: 0.85rem;
  color: #666;
}

.color-proportion {
  font-size: 0.8rem;
  color: var(--color-text-light);
  font-weight: 500;
}

.color-palette-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: var(--space-4);
  margin-top: var(--space-3);
}

.palette-item {
  display: flex;
  align-items: center;
  padding: var(--space-3);
  border-radius: var(--radius-md);
  background-color: white;
  box-shadow: var(--shadow-xs);
  transition: all 0.2s;
}

.palette-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}

.color-swatch {
  width: 36px;
  height: 36px;
  min-width: 36px;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.color-hex {
  font-family: monospace;
  font-size: 0.9rem;
  font-weight: 600;
  color: #333;
}

/* Details tab */
.details-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: var(--space-2);
}

.details-table tr {
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.details-table tr:last-child {
  border-bottom: none;
}

.details-table td {
  padding: var(--space-2);
  vertical-align: top;
}

.details-label {
  font-weight: 600;
  color: #555;
  white-space: nowrap;
  width: 30%;
}

.file-path {
  font-family: monospace;
  font-size: 0.85rem;
  word-break: break-all;
  color: #555;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 900px) {
  .modal-layout {
    flex-direction: column;
  }
  
  .image-column {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  }
  
  .details-column {
    width: 100%;
    min-width: unset;
    max-width: unset;
  }
  
  .color-palette-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  
  .modal-content {
    width: 100%;
    height: 100%;
    max-height: 100vh;
    border-radius: 0;
  }
}

@media (max-width: 600px) {
  .color-palette-grid {
    grid-template-columns: 1fr;
  }
  
  .tab-button {
    font-size: 0.9rem;
    padding: var(--space-2) var(--space-1);
  }
}

.keyword-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.keyword-tag {
  display: inline-block;
  background-color: rgba(var(--primary-rgb), 0.1);
  color: var(--primary-color);
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: default;
  transition: all 0.2s ease;
}

.keyword-tag:hover {
  background-color: rgba(var(--primary-rgb), 0.2);
}

.modal-content {
  padding: var(--space-4);
  height: calc(90vh - 60px); /* Adjust for header height */
  overflow-y: auto;
}

.main-image.hidden {
  opacity: 0;
  transition: opacity 0.3s;
}

.image-container.is-loading {
  min-height: 300px;
}

.image-error {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.9);
  padding: var(--space-4);
  text-align: center;
}

.error-message {
  color: var(--color-error);
  margin-bottom: var(--space-4);
  font-weight: 500;
}

.loading-spinner {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.7);
}

.loading-spinner .spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(79, 70, 229, 0.2);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s infinite linear;
  margin-bottom: var(--space-3);
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.debug-info {
  margin-bottom: var(--space-4);
  padding: var(--space-3);
  background-color: #f0f0f0;
  border-radius: var(--radius-md);
  font-family: monospace;
  font-size: 12px;
  overflow: auto;
  max-height: 200px;
}
</style> 