<template>
  <div class="gallery-container">
    <!-- Reset button -->
    <div class="reset-container">
      <button 
        class="reset-button" 
        @click="confirmReset"
        title="WARNING: This will delete ALL images"
      >
        Reset All Images
      </button>
    </div>
    
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
        draggable="true"
        @dragstart="handleDragStart(image, $event)"
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
                <path d="M3 6H5H21" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
          
          <div class="image-container" @click="handleImageClick(image)">
            <img 
              :src="getThumbnailUrl(image.thumbnail_path)" 
              :alt="getImageName(image)"
              class="gallery-image"
              loading="lazy"
              @error="handleImageError(image)"
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

    <!-- Add the reset confirmation modal -->
    <div v-if="showResetConfirm" class="reset-modal" @click="showResetConfirm = false">
      <div class="reset-modal-content" @click.stop>
        <h3>⚠️ WARNING: Delete ALL Images</h3>
        <p>This will permanently delete ALL images from the server.</p>
        <p>This action <strong>cannot be undone</strong>.</p>
        <div class="reset-modal-actions">
          <button class="cancel-button" @click="showResetConfirm = false">Cancel</button>
          <button class="confirm-button" @click="handleReset">Delete Everything</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useImageStore } from '../stores/imageStore'
import ImageModal from './modal/ImageModal.vue'

// Helper to control log level
const isDev = process.env.NODE_ENV === 'development'
const log = (message, ...args) => {
  if (isDev) {
    console.log(message, ...args)
  }
}

const imageStore = useImageStore()
const selectedImage = ref(null)
const showDeleteConfirm = ref(false)
const imageToDelete = ref(null)
const showResetConfirm = ref(false)
const isResetting = ref(false)

const images = computed(() => {
  // Get valid images from the store
  const validImages = imageStore.getValidImages() || []
  
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
    // Reset the store to get fresh data - only once on mount
    await imageStore.resetStore()
    
    // No need for additional real-time polling here
    // This will reduce the number of requests to the backend
  } catch (error) {
    // Only log errors in development
    if (isDev) {
      console.error("Failed to initialize gallery:", error)
    }
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

// Handle image error
const handleImageError = (image) => {
  if (!image || !image.thumbnail_path) return;
  
  if (isDev) {
    console.warn('Thumbnail failed to load:', image.thumbnail_path);
  }
  
  // Remove invalid images from the gallery
  imageStore.cleanGallery();
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
  if (!image) return
  imageToDelete.value = image
  showDeleteConfirm.value = true
}

const handleDelete = async () => {
  if (!imageToDelete.value) {
    if (isDev) {
      console.error("Cannot delete: Image is undefined")
    }
    showDeleteConfirm.value = false
    return
  }

  // Try to find a valid path using the same logic as getImageName
  const image = imageToDelete.value
  const path = image.original_path || image.file_path || image.image_path || 
              image.path || image.thumbnail_path

  if (path) {
    try {
      await imageStore.deleteImage(path)
      showDeleteConfirm.value = false
      imageToDelete.value = null
      if (selectedImage.value === imageToDelete.value) {
        selectedImage.value = null
      }
    } catch (error) {
      if (isDev) {
        console.error("Failed to delete image:", error)
      }
    }
  } else {
    if (isDev) {
      console.error("Cannot delete: Image path is undefined")
    }
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

// Replace getThumbnailUrl function
const getThumbnailUrl = (path) => {
  if (!path) return '';
  
  // Get just the filename
  const filename = path.includes('/') ? 
    imageStore.getFileName(path) : 
    path; // Use path directly if it's already just a filename
  
  // Construct API URL
  return `${imageStore.API_BASE_URL}/thumbnails/${filename}`;
}

// Add these functions for the reset functionality
const confirmReset = () => {
  showResetConfirm.value = true
}

const handleReset = async () => {
  try {
    isResetting.value = true
    
    // Call the nuclear option
    await imageStore.nukeEverything()
    
    showResetConfirm.value = false
    
    // Fetch fresh empty state
    await imageStore.fetchImages()
  } catch (error) {
    if (isDev) {
      console.error("Failed to reset images:", error)
    }
  } finally {
    isResetting.value = false
  }
}

// Add draggable attribute and event handler to the gallery item
const handleDragStart = (image, event) => {
  // Store the image ID (or path) in the drag event
  const imageId = image.id || getFileName(image.thumbnail_path || image.path || image.original_path || '');
  event.dataTransfer.setData('text/plain', imageId);
  event.dataTransfer.effectAllowed = 'copy';
  
  // Optional: you can add a CSS class for visual feedback
  event.target.closest('.gallery-item').classList.add('is-dragging');
  
  // Add a dragend event listener to remove the class when dragging ends
  event.target.closest('.gallery-item').addEventListener('dragend', () => {
    event.target.closest('.gallery-item').classList.remove('is-dragging');
  }, { once: true });
}

// Helper to get filename from path
const getFileName = (path) => {
  if (!path) return '';
  return path.split('/').pop();
}
</script>

<style scoped>
.gallery-container {
  width: 100%;
  margin: 0 var(--space-1);
}

.reset-container {
  display: flex;
  justify-content: flex-end;
  margin-bottom: var(--space-4);
}

.reset-button {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  background-color: rgba(251, 113, 133, 0.1);
  color: #fb7185;
  border: 1px solid rgba(251, 113, 133, 0.2);
  border-radius: var(--radius-md);
  padding: var(--space-2) var(--space-4);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.reset-button:hover {
  background-color: rgba(251, 113, 133, 0.15);
  color: #f43f5e;
}

.gallery-placeholder, 
.gallery-empty {
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  padding: var(--space-8);
  text-align: center;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(79, 70, 229, 0.1);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-text {
  font-size: 1.1rem;
  color: var(--color-text-light);
  margin: 0;
}

.empty-illustration {
  width: 80px;
  height: 80px;
  color: var(--color-text-lighter);
  margin-bottom: var(--space-4);
  opacity: 0.6;
}

.empty-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0 0 var(--space-2) 0;
  color: var(--color-text);
}

.empty-description {
  font-size: 1rem;
  color: var(--color-text-light);
  margin: 0;
  max-width: 400px;
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: var(--space-4);
  width: 100%;
  justify-content: center;
  margin: 0 auto;
}

.gallery-item {
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
}

.gallery-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
  border-color: rgba(79, 70, 229, 0.3);
}

.image-actions {
  position: absolute;
  top: var(--space-2);
  right: var(--space-2);
  z-index: 3;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.gallery-item:hover .image-actions {
  opacity: 1;
}

.delete-button {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background-color: rgba(0, 0, 0, 0.5);
  border: none;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  backdrop-filter: blur(4px);
}

.delete-button:hover {
  background-color: rgba(0, 0, 0, 0.7);
}

.delete-button svg {
  width: 16px;
  height: 16px;
}

.image-container {
  position: relative;
  height: 200px;
  overflow: hidden;
  cursor: pointer;
}

.gallery-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s ease;
}

.gallery-item:hover .gallery-image {
  transform: scale(1.05);
}

.image-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.7) 0%, rgba(0, 0, 0, 0) 60%);
  display: flex;
  align-items: flex-end;
  padding: var(--space-3);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.gallery-item:hover .image-overlay {
  opacity: 1;
}

.overlay-content {
  width: 100%;
}

.overlay-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: white;
  margin: 0 0 var(--space-2) 0;
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.confidence-badge {
  font-size: 0.7rem;
  background-color: rgba(79, 70, 229, 0.4);
  color: white;
  padding: 2px 6px;
  border-radius: var(--radius-full);
  backdrop-filter: blur(4px);
}

.color-chips {
  display: flex;
  gap: 4px;
}

.color-chip {
  width: 15px;
  height: 15px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.search-score-info {
  margin-top: var(--space-2);
  font-size: 0.75rem;
  color: white;
}

.image-metadata {
  padding: var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  background: rgba(255, 255, 255, 0.02);
  flex: 1;
}

.pattern-type {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  font-size: 0.85rem;
}

.type-label {
  color: var(--color-text-light);
  font-weight: 500;
  font-size: 0.75rem;
}

.type-value {
  color: var(--color-text);
  font-weight: 600;
}

.secondary-patterns {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-2);
  align-items: center;
  font-size: 0.75rem;
}

.sec-pattern-label {
  color: var(--color-text-light);
}

.sec-pattern-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.sec-pattern-tag {
  background-color: rgba(79, 70, 229, 0.1);
  color: var(--color-primary);
  padding: 2px 6px;
  border-radius: var(--radius-full);
  font-size: 0.7rem;
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
  background-color: rgba(34, 211, 238, 0.1);
  color: #22d3ee;
  padding: 2px 6px;
  border-radius: var(--radius-full);
  font-size: 0.7rem;
}

.search-score {
  margin-top: var(--space-2);
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: var(--radius-full);
  height: 6px;
  overflow: hidden;
  position: relative;
}

.score-bar {
  height: 100%;
  background: linear-gradient(to right, #4f46e5, #3b82f6);
  border-radius: var(--radius-full);
}

.score-label {
  font-size: 0.7rem;
  color: var(--color-text-light);
  margin-top: 4px;
  display: block;
}

/* Upload placeholder */
.upload-placeholder {
  height: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-4);
  gap: var(--space-4);
  background-color: rgba(255, 255, 255, 0.03);
}

.upload-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(79, 70, 229, 0.1);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.upload-progress {
  width: 100%;
  text-align: center;
}

.progress-bar {
  height: 6px;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-bottom: var(--space-2);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(to right, #4f46e5, #3b82f6);
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--color-primary);
  margin: 0;
}

.upload-status {
  font-size: 0.8rem;
  color: var(--color-text-light);
  margin: var(--space-1) 0 0 0;
}

/* Modal styles */
.delete-modal,
.reset-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.delete-modal-content,
.reset-modal-content {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  width: 90%;
  max-width: 400px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
  animation: scaleIn 0.3s ease;
}

@keyframes scaleIn {
  from { transform: scale(0.9); }
  to { transform: scale(1); }
}

.delete-modal-content h3,
.reset-modal-content h3 {
  margin-top: 0;
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: var(--space-3);
  color: var(--color-text);
}

.reset-modal-content h3 {
  color: #f43f5e;
}

.delete-modal-content p,
.reset-modal-content p {
  color: var(--color-text-light);
  margin-bottom: var(--space-4);
}

.delete-modal-actions,
.reset-modal-actions {
  display: flex;
  gap: var(--space-3);
  justify-content: flex-end;
}

.cancel-button {
  padding: var(--space-2) var(--space-4);
  background-color: transparent;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: var(--color-text);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-button:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.confirm-button {
  padding: var(--space-2) var(--space-4);
  background-color: rgba(251, 113, 133, 0.1);
  color: #fb7185;
  border: 1px solid rgba(251, 113, 133, 0.2);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.confirm-button:hover {
  background-color: rgba(251, 113, 133, 0.2);
  color: #f43f5e;
}

/* Responsive design */
@media (max-width: 1400px) {
  .gallery-grid {
    grid-template-columns: repeat(5, 1fr);
  }
}

@media (max-width: 1200px) {
  .gallery-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

@media (max-width: 992px) {
  .gallery-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .gallery-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: var(--space-3);
  }
  
  .image-container {
    height: 150px;
  }
  
  .pattern-type {
    font-size: 0.8rem;
  }
  
  .sec-pattern-label,
  .sec-pattern-tag {
    font-size: 0.7rem;
  }
}
</style> 