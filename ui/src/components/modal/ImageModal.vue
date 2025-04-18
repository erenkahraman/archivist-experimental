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
  { id: 'patterns', label: 'Patterns' }
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
  // Try various possible locations for style keywords
  if (props.selectedImage.patterns?.style_keywords) {
    return props.selectedImage.patterns.style_keywords;
  }
  
  if (props.selectedImage.pattern?.keywords) {
    return props.selectedImage.pattern.keywords;
  }
  
  if (props.selectedImage.keywords) {
    return props.selectedImage.keywords;
  }
  
  if (props.selectedImage.metadata?.style_keywords) {
    return Array.isArray(props.selectedImage.metadata.style_keywords) 
      ? props.selectedImage.metadata.style_keywords 
      : [props.selectedImage.metadata.style_keywords];
  }
  
  return [];
};

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

<style>
.modal-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.3s ease, visibility 0.3s ease;
}

.modal-container.is-open {
  opacity: 1;
  visibility: visible;
}

.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(5px);
  z-index: -1;
}

.modal {
  width: 90%;
  max-width: 1000px;
  max-height: 90vh;
  background-color: rgba(18, 18, 18, 0.95);
  border-radius: var(--radius-lg);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 
              0 0 0 1px rgba(30, 30, 30, 0.3), 
              0 0 40px rgba(0, 0, 0, 0.4);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  animation: modalEnter 0.3s ease forwards;
  border: 1px solid #252525;
}

@keyframes modalEnter {
  0% {
    opacity: 0;
    transform: scale(0.95);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-4) var(--space-6);
  border-bottom: 1px solid rgba(40, 40, 40, 0.3);
  background-color: rgba(25, 25, 25, 0.5);
}

.modal-header h3 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #aaaaaa;
}

.close-button {
  background: none;
  border: none;
  color: var(--color-text-light);
  font-size: 1.5rem;
  cursor: pointer;
  line-height: 1;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.2s ease;
}

.close-button:hover {
  color: #aaaaaa;
  background-color: rgba(40, 40, 40, 0.3);
}

.modal-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: var(--space-6);
}

/* Firefox fix for scrolling */
@-moz-document url-prefix() {
  .modal-content {
    scrollbar-width: thin;
    scrollbar-color: var(--color-primary-light) transparent;
  }
}

.modal-content::-webkit-scrollbar {
  width: 6px;
}

.modal-content::-webkit-scrollbar-track {
  background: transparent;
}

.modal-content::-webkit-scrollbar-thumb {
  background-color: rgba(58, 58, 58, 0.3);
  border-radius: 20px;
}

.image-section {
  margin-bottom: var(--space-6);
}

.image-container {
  width: 100%;
  position: relative;
  text-align: center;
  min-height: 200px;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background-color: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.main-image {
  max-width: 100%;
  max-height: 500px;
  border-radius: var(--radius-md);
  object-fit: contain;
}

.image-error {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
  color: var(--color-text-light);
}

.error-message {
  font-size: 1rem;
  max-width: 300px;
  text-align: center;
}

.try-thumbnail-btn {
  padding: var(--space-2) var(--space-4);
  background-color: rgba(58, 58, 58, 0.2);
  border: 1px solid rgba(58, 58, 58, 0.3);
  color: var(--color-primary-light);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.try-thumbnail-btn:hover {
  background-color: rgba(58, 58, 58, 0.3);
  border-color: var(--color-primary);
}

.loading-spinner {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
  color: var(--color-text-light);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(58, 58, 58, 0.1);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s infinite linear;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.hidden {
  display: none;
}

/* Tabs */
.tabs {
  display: flex;
  gap: var(--space-1);
  margin-bottom: var(--space-6);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: var(--space-2);
}

.tab-button {
  padding: var(--space-2) var(--space-4);
  background: none;
  border: none;
  color: var(--color-text-light);
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  border-radius: var(--radius-md);
  transition: all 0.2s ease;
}

.tab-button:hover {
  background-color: rgba(40, 40, 40, 0.2);
  color: var(--color-text);
  transform: none;
  box-shadow: none;
}

.tab-button.active {
  background-color: rgba(30, 30, 30, 0.5);
  color: #aaaaaa;
  border-bottom: 2px solid #3a3a3a;
}

.tab-content {
  flex: 1;
}

.tab-pane {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Patterns Tab */
.pattern-section {
  margin-bottom: var(--space-6);
}

.section-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: var(--space-3);
  color: #aaaaaa;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid rgba(40, 40, 40, 0.3);
}

.pattern-primary {
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--color-text);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-3);
  background: rgba(30, 30, 30, 0.3);
  border-radius: var(--radius-md);
  border: 1px solid #3a3a3a;
}

.inline-confidence {
  font-size: 0.8rem;
  padding: var(--space-1) var(--space-2);
  background-color: rgba(40, 40, 40, 0.4);
  color: #aaaaaa;
  border-radius: var(--radius-full);
  font-weight: normal;
}

.confidence-note {
  font-size: 0.85rem;
  color: var(--color-text-light);
  margin-bottom: var(--space-3);
}

.tag-container {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.tag {
  display: inline-block;
  padding: var(--space-2) var(--space-3);
  background-color: rgba(30, 30, 30, 0.3);
  border: 1px solid rgba(40, 40, 40, 0.3);
  color: var(--color-text);
  border-radius: var(--radius-full);
  font-size: 0.85rem;
}

.image-data {
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--color-text);
  padding: var(--space-3);
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: var(--radius-md);
  border: 1px solid rgba(40, 40, 40, 0.3);
}

.keyword-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.keyword-tag {
  padding: var(--space-1) var(--space-3);
  background-color: rgba(40, 40, 40, 0.3);
  color: #aaaaaa;
  border-radius: var(--radius-full);
  font-size: 0.8rem;
  border: 1px solid rgba(40, 40, 40, 0.3);
}

/* Colors Tab */
.color-section {
  margin-bottom: var(--space-6);
}

.color-display {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  padding: var(--space-3);
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: var(--radius-md);
  border: 1px solid rgba(40, 40, 40, 0.3);
}

.color-preview {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.color-info {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.color-value {
  font-family: monospace;
  font-size: 0.9rem;
  color: var(--color-text);
}

.color-name {
  font-size: 0.85rem;
  color: var(--color-text-light);
  font-weight: 500;
}

.color-proportion {
  font-size: 0.8rem;
  color: var(--color-text-light);
}

.color-palette-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--space-4);
}

.palette-item {
  padding: var(--space-3);
  background-color: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

/* Details Tab */
.details-section {
  margin-bottom: var(--space-6);
}

.details-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
}

.details-table tbody {
  display: table;
  width: 100%;
}

.details-table tr {
  display: table-row;
}

.details-table td {
  padding: var(--space-3);
  border-bottom: 1px solid rgba(40, 40, 40, 0.2);
  display: table-cell;
}

.details-table tr:last-child td {
  border-bottom: none;
}

.details-label {
  font-weight: 500;
  color: #777777;
  width: 120px;
}

.file-path {
  font-family: monospace;
  word-break: break-all;
  font-size: 0.9rem;
}

.debug-info {
  padding: var(--space-3);
  background-color: rgba(0, 0, 0, 0.3);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-4);
  overflow-x: auto;
}

.debug-info pre {
  margin: 0;
  font-family: monospace;
  font-size: 0.8rem;
  color: var(--color-text-light);
}

@media (max-width: 768px) {
  .modal {
    width: 95%;
    max-height: 95vh;
  }
  
  .modal-header {
    padding: var(--space-3) var(--space-4);
  }
  
  .modal-content {
    padding: var(--space-4);
  }
  
  .main-image {
    max-height: 300px;
  }
  
  .tabs {
    overflow-x: auto;
    padding-bottom: var(--space-1);
  }
  
  .tab-button {
    padding: var(--space-2) var(--space-3);
    white-space: nowrap;
  }
  
  .color-palette-grid {
    grid-template-columns: 1fr;
  }
  
  .pattern-primary {
    font-size: 1rem;
  }
  
  .details-label {
    width: 100px;
  }
}
</style> 