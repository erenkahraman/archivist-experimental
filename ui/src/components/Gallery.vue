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
        :key="image.isUploading ? `uploading-${image.original_path}` : image.thumbnail_path"
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
          
          <div class="image-container" @click="selectImage(image)">
            <img 
              :src="getThumbnailUrl(image.thumbnail_path)" 
              :alt="getImageName(image.original_path)"
              class="gallery-image"
              loading="lazy"
            >
            
            <!-- Hover overlay with quick info -->
            <div class="image-overlay">
              <div class="overlay-content">
                <p class="overlay-title">{{ image.patterns?.primary_pattern || 'Unknown pattern' }}</p>
                <div class="color-chips" v-if="image.colors?.palette">
                  <div 
                    v-for="(color, idx) in image.colors.palette.slice(0, 5)" 
                    :key="idx"
                    class="color-chip"
                    :style="{ backgroundColor: color }"
                  ></div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="image-metadata">
            <div class="pattern-type">
              <span class="type-value">{{ image.patterns?.primary_pattern || 'Unknown' }}</span>
            </div>
            <div class="pattern-prompt">
              {{ truncatePrompt(image.patterns?.prompt) }}
            </div>
            
            <!-- Display search score when in search mode -->
            <div v-if="searchActive && image.searchScore !== undefined" class="search-score">
              <div class="score-bar" :style="{ width: `${image.searchScore * 100}%` }"></div>
              <span class="score-label">Match: {{ (image.searchScore * 100).toFixed(0) }}%</span>
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

const images = computed(() => imageStore.images || [])
const loading = computed(() => imageStore.loading)
const searchActive = computed(() => imageStore.searchQuery !== '')

onMounted(() => {
  imageStore.fetchImages()
  imageStore.clearUploadingStates()
})

const getThumbnailUrl = (path) => {
  if (!path) {
    console.log('Warning: Empty thumbnail path');
    return '';
  }
  return `http://localhost:8000/api/thumbnails/${path.split('/').pop()}`
}

const getImageName = (path) => {
  if (!path) return 'Unknown'
  return path.split('/').pop()
}

const selectImage = (image) => {
  console.log('Selected image data:', image);
  
  // Check if this is an uploading image
  if (image.isUploading) {
    console.log('Cannot select an image that is still uploading');
    return;
  }
  
  // Ensure the image has a valid path
  if (!image.original_path) {
    console.error('Image has no path:', image);
    
    // Try to fix the path if possible
    if (image.path) {
      console.log('Using image.path instead:', image.path);
      image.original_path = image.path;
    } else if (image.thumbnail_path) {
      // Derive original path from thumbnail path
      const filename = image.thumbnail_path.split('/').pop();
      image.original_path = `uploads/${filename}`;
      console.log('Derived path from thumbnail:', image.original_path);
    } else {
      alert('This image cannot be selected because it has no path. Please try refreshing the page.');
      return;
    }
  }
  
  selectedImage.value = image
}

const confirmDelete = (image) => {
  imageToDelete.value = image
  showDeleteConfirm.value = true
}

const handleDelete = async () => {
  if (imageToDelete.value) {
    await imageStore.deleteImage(imageToDelete.value.original_path)
    showDeleteConfirm.value = false
    imageToDelete.value = null
    if (selectedImage.value === imageToDelete.value) {
      selectedImage.value = null
    }
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
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.gallery-item:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.image-container {
  position: relative;
  overflow: hidden;
  cursor: pointer;
  height: 250px;
}

.gallery-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform var(--transition-normal);
}

.gallery-item:hover .gallery-image {
  transform: scale(1.05);
}

.image-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.7), transparent 70%);
  opacity: 0;
  transition: opacity var(--transition-normal);
  display: flex;
  align-items: flex-end;
  padding: var(--space-4);
}

.gallery-item:hover .image-overlay {
  opacity: 1;
}

.overlay-content {
  color: white;
  width: 100%;
}

.overlay-title {
  font-weight: 600;
  margin-bottom: var(--space-2);
  font-size: 1.1rem;
}

.color-chips {
  display: flex;
  gap: var(--space-1);
}

.color-chip {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.8);
}

.image-metadata {
  padding: var(--space-4);
  flex-grow: 1;
  display: flex;
  flex-direction: column;
}

.pattern-type {
  margin-bottom: var(--space-2);
}

.type-value {
  font-family: var(--font-heading);
  font-weight: 600;
  color: var(--color-primary);
  font-size: 1.1rem;
}

.pattern-prompt {
  color: var(--color-text-light);
  font-size: 0.9rem;
  line-height: 1.4;
  flex-grow: 1;
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
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.9);
  color: var(--color-error);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  padding: 0;
  border: none;
}

.delete-button svg {
  width: 16px;
  height: 16px;
}

.delete-button:hover {
  background-color: var(--color-error);
  color: white;
  transform: none;
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

/* Search score display */
.search-score {
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px solid var(--color-border);
}

.score-bar {
  height: 4px;
  background: var(--gradient-primary);
  border-radius: var(--radius-full);
  margin-bottom: var(--space-1);
}

.score-label {
  font-size: 0.8rem;
  color: var(--color-primary);
  font-weight: 500;
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