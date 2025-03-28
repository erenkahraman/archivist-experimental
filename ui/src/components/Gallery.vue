<template>
  <div class="gallery-container">
    <!-- Empty state -->
    <div v-if="loading" class="gallery-placeholder animate__animated animate__pulse">
      <div class="loading-container">
        <div class="loading-spinner"></div>
        <p class="loading-text">Discovering images...</p>
      </div>
    </div>
    
    <!-- Empty state -->
    <div v-else-if="images.length === 0" class="gallery-empty animate__animated animate__fadeIn">
      <div class="empty-illustration">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M4 16L8.58579 11.4142C9.36684 10.6332 10.6332 10.6332 11.4142 11.4142L16 16M14 14L15.5858 12.4142C16.3668 11.6332 17.6332 11.6332 18.4142 12.4142L20 14M14 8H14.01M6 20H18C19.1046 20 20 19.1046 20 18V6C20 4.89543 19.1046 4 18 4H6C4.89543 4 4 4.89543 4 6V18C4 19.1046 4.89543 20 6 20Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <h3 v-if="searchActive" class="empty-title">No images found</h3>
      <h3 v-else class="empty-title">Your gallery is empty</h3>
      <p v-if="searchActive" class="empty-description">Try adjusting your search criteria or upload new images</p>
      <p v-else class="empty-description">Upload images to begin your collection</p>
    </div>
    
    <!-- Image grid -->
    <div v-else class="gallery-grid animate__animated animate__fadeIn">
      <div 
        v-for="image in images" 
        :key="image.id || image.timestamp || (image.thumbnail_path ? image.thumbnail_path : Math.random())"
        class="gallery-item"
        :class="{
          'is-uploading': image.isUploading,
          'is-searching': image.isSearching
        }"
      >
        <!-- Uploading state -->
        <template v-if="image.isUploading">
          <div class="upload-placeholder">
            <div class="upload-spinner"></div>
            <div class="upload-progress">
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: `${image.uploadProgress || 0}%` }"></div>
              </div>
              <p class="progress-text">{{ image.uploadProgress || 0 }}%</p>
              <p class="upload-status">{{ image.uploadStatus || 'Uploading...' }}</p>
            </div>
          </div>
        </template>
        
        <!-- Regular image display -->
        <template v-else>
          <div class="image-actions">
            <button 
              class="delete-button"
              @click.stop="confirmDelete(image)"
              title="Delete image"
              aria-label="Delete image"
            >
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
          
          <div class="image-container" @click="handleImageClick(image)">
            <img 
              :src="getThumbnailUrl(image.thumbnail_path)" 
              :alt="getImageName(image)"
              class="gallery-image"
              loading="lazy"
            >
            
            <!-- Hover overlay with quick info -->
            <div class="image-overlay">
              <div class="overlay-content">
                <p class="overlay-title">
                  {{ getPatternName(image) }}
                  <span v-if="getPatternConfidence(image)" class="confidence-badge">
                    {{ formatConfidence(getPatternConfidence(image)) }}
                  </span>
                </p>
                
                <!-- Color chips -->
                <div class="color-chips" v-if="image.colors && image.colors.length > 0">
                  <div 
                    v-for="(color, idx) in image.colors.slice(0, 5)" 
                    :key="idx"
                    class="color-chip"
                    :style="{ backgroundColor: color.hex }"
                    :title="`${color.name} (${Math.round(color.proportion * 100)}%)`"
                  ></div>
                </div>
                
                <!-- Search score when in search mode -->
                <div v-if="searchActive && image.similarity !== undefined" class="search-score-info">
                  <span class="score-label">Match score: {{ (image.similarity * 100).toFixed(0) }}%</span>
                </div>
              </div>
            </div>
          </div>
          
          <div class="image-metadata">
            <div class="pattern-type">
              <span class="type-label">Pattern:</span>
              <span class="type-value">{{ getPatternName(image) }}</span>
            </div>
            
            <!-- Secondary patterns if available -->
            <div class="secondary-patterns" v-if="hasSecondaryPatterns(image)">
              <span class="sec-pattern-label">Also:</span>
              <div class="sec-pattern-tags">
                <span 
                  v-for="(pattern, idx) in getSecondaryPatterns(image)" 
                  :key="idx"
                  class="sec-pattern-tag"
                >
                  {{ pattern }}
                </span>
              </div>
            </div>
            
            <!-- Style keywords -->
            <div class="style-keywords" v-if="hasStyleKeywords(image)">
              <div class="keyword-tags">
                <span 
                  v-for="(keyword, idx) in getStyleKeywords(image)" 
                  :key="idx"
                  class="keyword-tag"
                >
                  {{ keyword }}
                </span>
              </div>
            </div>
            
            <!-- Display search score when in search mode -->
            <div v-if="searchActive && image.similarity !== undefined" class="search-score">
              <div class="score-bar" :style="{ width: `${image.similarity * 100}%` }"></div>
              <span class="score-label">Match: {{ (image.similarity * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Use the modular ImageModal component -->
    <ImageModal 
      v-if="selectedImage" 
      :selected-image="selectedImage" 
      @close="selectedImage = null"
    />

    <!-- Delete Confirmation Modal -->
    <div v-if="showDeleteConfirm" class="delete-modal" @click="showDeleteConfirm = false">
      <div class="delete-modal-content" @click.stop>
        <h3>Delete Image</h3>
        <p>Are you sure you want to delete this image?</p>
        <div class="delete-modal-actions">
          <button class="cancel-button" @click="showDeleteConfirm = false">Cancel</button>
          <button class="confirm-button" @click="handleDelete">Delete</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useImageStore } from '../stores/imageStore'
import ImageModal from './modal/ImageModal.vue'

const imageStore = useImageStore()
const selectedImage = ref(null)
const showDeleteConfirm = ref(false)
const imageToDelete = ref(null)

const images = computed(() => {
  if (!imageStore.images) return []
  
  // Filter out images without thumbnails - these are likely deleted images
  const validImages = imageStore.images.filter(img => {
    // Keep uploading images
    if (img.isUploading) return true
    
    // Filter out images with missing thumbnails or original paths
    return img.thumbnail_path && img.original_path
  })
  
  // Sort images by timestamp in descending order (newest first)
  return [...validImages].sort((a, b) => {
    const timeA = a.timestamp || 0
    const timeB = b.timestamp || 0
    return timeB - timeA
  })
})
const loading = computed(() => imageStore.loading)
const searchActive = computed(() => imageStore.searchQuery !== '')

onMounted(async () => {
  try {
    // Clear all local storage immediately
    localStorage.removeItem('gallery-images')
    
    // Fetch fresh data directly from server
    await imageStore.purgeDeletedImages()
    
    // Always clear uploading states
    imageStore.clearUploadingStates()
  } catch (error) {
    console.error("Failed to initialize gallery:", error)
  }
})

// Helper functions for pattern information
const getPatternName = (image) => {
  if (image.pattern && image.pattern.primary) {
    return image.pattern.primary;
  }
  if (image.patterns && image.patterns.primary_pattern) {
    return image.patterns.primary_pattern;
  }
  return 'Unknown pattern';
}

const getPatternConfidence = (image) => {
  if (image.pattern && image.pattern.confidence) {
    return image.pattern.confidence;
  }
  if (image.patterns && image.patterns.pattern_confidence) {
    return image.patterns.pattern_confidence;
  }
  return null;
}

const formatConfidence = (confidence) => {
  const percent = Math.round(confidence * 100);
  return `${percent}%`;
}

const hasSecondaryPatterns = (image) => {
  if (image.pattern && image.pattern.secondary && image.pattern.secondary.length > 0) {
    return true;
  }
  if (image.patterns && image.patterns.secondary_patterns && image.patterns.secondary_patterns.length > 0) {
    return true;
  }
  return false;
}

const getSecondaryPatterns = (image) => {
  if (image.pattern && image.pattern.secondary) {
    return image.pattern.secondary;
  }
  if (image.patterns && image.patterns.secondary_patterns) {
    return image.patterns.secondary_patterns.map(p => p.name);
  }
  return [];
}

const hasStyleKeywords = (image) => {
  if (image.style_keywords && image.style_keywords.length > 0) {
    return true;
  }
  if (image.patterns && image.patterns.style_keywords && image.patterns.style_keywords.length > 0) {
    return true;
  }
  return false;
}

const getStyleKeywords = (image) => {
  if (image.style_keywords) {
    return image.style_keywords;
  }
  if (image.patterns && image.patterns.style_keywords) {
    return image.patterns.style_keywords;
  }
  return [];
}

// Helper function to get thumbnail URL
const getThumbnailUrl = (path) => {
  if (!path) return '';
  
  // If it's already a full URL, use it
  if (path.startsWith('http')) {
    return path;
  }
  
  // If it already has the API path, use it
  if (path.includes('/api/')) {
    return path;
  }
  
  // Construct the API URL with just the filename
  const filename = path.split('/').pop();
  return `http://localhost:8000/api/thumbnails/${filename}`;
}

const getImageName = (image) => {
  // Check various possible path properties
  const path = image.file_path || image.image_path || image.original_path || 
              image.path || image.thumbnail_path || '';
  
  if (!path) return 'Unknown image';
  
  return path.split('/').pop();
}

// Handle image click to open modal
const handleImageClick = (image) => {
  // Just directly set the selected image without modifications
  selectedImage.value = image;
}

const confirmDelete = (image) => {
  console.log("Confirm delete called with image:", image)
  if (image && image.original_path) {
    console.log("Image has valid path:", image.original_path)
  } else {
    console.warn("Image is missing original_path:", image)
  }
  imageToDelete.value = image
  showDeleteConfirm.value = true
}

const handleDelete = async () => {
  if (!imageToDelete.value) {
    console.error("Cannot delete: Image is undefined")
    showDeleteConfirm.value = false
    return
  }

  // Try to find a valid path using the same logic as getImageName
  const image = imageToDelete.value
  const path = image.original_path || image.file_path || image.image_path || 
              image.path || image.thumbnail_path

  if (path) {
    try {
      console.log("Deleting image with path:", path)
      await imageStore.deleteImage(path)
      showDeleteConfirm.value = false
      imageToDelete.value = null
      if (selectedImage.value === imageToDelete.value) {
        selectedImage.value = null
      }
    } catch (error) {
      console.error("Failed to delete image:", error)
    }
  } else {
    console.error("Cannot delete: Image path is undefined")
    showDeleteConfirm.value = false
    imageToDelete.value = null
  }
}

const truncatePrompt = (prompt) => {
  return getPromptText(prompt, true);
}

const getPromptText = (prompt, truncate = false) => {
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
</script>

<style scoped>
.gallery-container {
  min-height: 300px;
}

.gallery-placeholder,
.gallery-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  padding: var(--space-8);
  text-align: center;
  border-radius: var(--radius-lg);
  background-color: rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(4px);
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
}

.loading-text {
  color: var(--color-text-light);
  font-size: 1.1rem;
}

.empty-illustration {
  width: 100px;
  height: 100px;
  color: var(--color-text-light);
  opacity: 0.6;
  margin-bottom: var(--space-4);
}

.empty-title {
  font-family: var(--font-heading);
  font-size: 1.8rem;
  color: var(--color-text);
  margin-bottom: var(--space-2);
}

.empty-description {
  color: var(--color-text-light);
  max-width: 400px;
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--space-6);
}

.gallery-item {
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background-color: var(--color-surface);
  box-shadow: var(--shadow-sm);
  transition: all var(--transition-fast);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.gallery-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.gallery-item.search-result {
  border: 2px solid transparent;
}

.gallery-item.search-result:hover {
  border-color: var(--color-primary);
}

.image-container {
  position: relative;
  overflow: hidden;
  aspect-ratio: 1 / 1;
  cursor: pointer;
}

.gallery-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform var(--transition-fast);
}

.image-container:hover .gallery-image {
  transform: scale(1.05);
}

.image-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent);
  opacity: 0;
  transition: opacity var(--transition-fast);
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: var(--space-3);
  color: white;
}

.image-container:hover .image-overlay {
  opacity: 1;
}

.overlay-title {
  font-weight: 600;
  margin-bottom: var(--space-2);
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.confidence-badge {
  font-size: 0.7rem;
  background-color: rgba(255, 255, 255, 0.2);
  padding: 2px 6px;
  border-radius: 10px;
  font-weight: normal;
}

.color-chips {
  display: flex;
  gap: var(--space-1);
  margin-bottom: var(--space-2);
}

.color-chip {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.image-metadata {
  padding: var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  flex: 1;
}

.pattern-type {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.type-label {
  font-size: 0.7rem;
  font-weight: 600;
  opacity: 0.7;
}

.type-value {
  font-size: 0.85rem;
  font-weight: 600;
}

.secondary-patterns {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
  align-items: center;
}

.sec-pattern-label {
  font-size: 0.7rem;
  opacity: 0.7;
}

.sec-pattern-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.sec-pattern-tag {
  font-size: 0.7rem;
  background-color: var(--color-surface-light);
  padding: 2px 6px;
  border-radius: 10px;
}

.style-keywords {
  margin-top: var(--space-1);
}

.keyword-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.keyword-tag {
  font-size: 0.65rem;
  background-color: var(--color-surface-lighter);
  color: var(--color-text-light);
  padding: 1px 5px;
  border-radius: 6px;
}

.search-score {
  margin-top: auto;
  padding-top: var(--space-2);
  position: relative;
  height: 6px;
  background-color: var(--color-surface-lighter);
  border-radius: 10px;
  overflow: hidden;
}

.score-bar {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background-color: var(--color-primary);
  border-radius: 10px;
}

.score-label {
  position: absolute;
  right: 0;
  top: -16px;
  font-size: 0.7rem;
  color: var(--color-text-light);
}

.search-score-info {
  margin-top: var(--space-1);
  font-size: 0.8rem;
}

.image-actions {
  position: absolute;
  top: var(--space-2);
  right: var(--space-2);
  z-index: 10;
  opacity: 0;
  transition: opacity var(--transition-fast);
}

.gallery-item:hover .image-actions {
  opacity: 1;
}

.delete-button {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.delete-button:hover {
  background: rgba(255, 0, 0, 0.8);
}

.delete-button svg {
  width: 14px;
  height: 14px;
}

/* Upload placeholder styles */
.upload-placeholder {
  height: 100%;
  min-height: 250px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-4);
  background-color: rgba(79, 70, 229, 0.05);
}

.upload-spinner {
  margin-bottom: var(--space-4);
}

.upload-progress {
  width: 100%;
  text-align: center;
}

.progress-bar {
  height: 6px;
  background-color: var(--color-border);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-bottom: var(--space-2);
}

.progress-fill {
  height: 100%;
  background: var(--gradient-primary);
  transition: width 0.3s ease;
}

.progress-text {
  font-weight: 600;
  color: var(--color-primary);
  margin-bottom: var(--space-1);
}

.upload-status {
  font-size: 0.85rem;
  color: var(--color-text-light);
}

/* Delete modal */
.delete-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-4);
}

.delete-modal-content {
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  width: 100%;
  max-width: 400px;
  box-shadow: var(--shadow-xl);
}

.delete-modal-content h3 {
  font-family: var(--font-heading);
  margin-bottom: var(--space-3);
  color: var(--color-error);
}

.delete-modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-3);
  margin-top: var(--space-6);
}

.cancel-button {
  background-color: transparent;
  border: 1px solid var(--color-border);
  color: var(--color-text);
}

.cancel-button:hover {
  background-color: var(--color-background);
  transform: none;
}

.confirm-button {
  background-color: var(--color-error);
}

.confirm-button:hover {
  background-color: #dc2626;
}

@media (max-width: 768px) {
  .gallery-grid {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: var(--space-4);
  }
  
  .image-container {
    height: 200px;
  }
  
  .upload-placeholder {
    min-height: 200px;
  }
  
  .empty-illustration {
    width: 80px;
    height: 80px;
  }
  
  .empty-title {
    font-size: 1.5rem;
  }
}
</style> 