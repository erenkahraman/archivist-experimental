<template>
  <div class="gallery-container">
    <div v-if="loading" class="gallery-loading">
      <div class="loading-spinner"></div>
      <p>Loading images...</p>
    </div>
    
    <div v-else-if="images.length === 0" class="gallery-empty">
      <p v-if="searchActive">No images found for your search.</p>
      <p v-else>Upload pattern images to get started</p>
    </div>
    
    <div v-else class="gallery-grid">
      <div 
        v-for="image in images" 
        :key="image.isUploading ? `uploading-${image.original_path}` : image.thumbnail_path"
        class="gallery-item"
        :class="{
          'is-uploading': image.isUploading,
          'is-searching': image.isSearching
        }"
      >
        <template v-if="image.isUploading">
          <div class="upload-placeholder">
            <div class="upload-spinner"></div>
            <div class="upload-progress">
              <div class="progress-bar">
                <div 
                  class="progress-fill"
                  :style="{ width: `${image.uploadProgress || 0}%` }"
                ></div>
              </div>
              <p class="progress-text">{{ image.uploadProgress || 0 }}%</p>
              <p class="upload-status">{{ image.uploadStatus || 'Uploading...' }}</p>
            </div>
          </div>
        </template>
        <template v-else>
          <div class="image-actions">
            <button 
              class="delete-button"
              @click.stop="confirmDelete(image)"
              title="Delete image"
            >×</button>
          </div>
          <img 
            :src="getThumbnailUrl(image.thumbnail_path)" 
            :alt="getImageName(image.original_path)"
            class="gallery-image"
            @click="selectImage(image)"
          >
          <div class="image-metadata">
            <div class="pattern-type">
              <span class="type-label">Pattern:</span>
              <span class="type-value">{{ image.patterns?.primary_pattern || 'Unknown' }}</span>
            </div>
            <div class="pattern-prompt">
              {{ truncatePrompt(image.patterns?.prompt) }}
            </div>
            <!-- Display search score when in search mode -->
            <div v-if="searchActive && image.searchScore !== undefined" class="search-score">
              <div class="score-bar" :style="{ width: `${image.searchScore * 100}%` }"></div>
              <span class="score-label">Match: {{ (image.searchScore * 100).toFixed(0) }}%</span>
              <span v-if="image.matchedTerms" class="terms-matched">
                {{ image.matchedTerms }} term{{ image.matchedTerms !== 1 ? 's' : '' }} matched
              </span>
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
  console.log('Getting thumbnail for path:', path);
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
  
  console.log('Using image path:', image.original_path);
  console.log('Pattern analysis:', image.patterns);
  console.log('Color analysis:', image.colors);
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
  min-height: 200px;
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem;
}

.gallery-item {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
  cursor: pointer;
}

.gallery-item:hover {
  transform: translateY(-4px);
}

.gallery-image {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.image-metadata {
  padding: 0.8rem;
  background: rgba(255, 255, 255, 0.95);
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.pattern-type {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.type-label {
  font-weight: 500;
  color: #666;
  font-size: 0.9rem;
}

.type-value {
  color: #2c3e50;
  font-weight: 600;
}

.pattern-prompt {
  font-size: 0.85rem;
  color: #666;
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
}

.gallery-loading,
.gallery-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  color: #666;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #4CAF50;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

.delete-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.delete-modal-content {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  max-width: 400px;
  width: 90%;
}

.delete-modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  margin-top: 1.5rem;
}

.cancel-button {
  padding: 0.5rem 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  cursor: pointer;
}

.confirm-button {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  background: #dc3545;
  color: white;
  cursor: pointer;
}

.confirm-button:hover {
  background: #c82333;
}

.is-uploading {
  opacity: 0.7;
  pointer-events: none;
}

.upload-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(76, 175, 80, 0.05);
  border-radius: 8px;
  padding: 1rem;
}

.upload-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #4CAF50;
  border-radius: 50%;
  margin-bottom: 1rem;
  animation: spin 1s linear infinite;
}

.upload-progress {
  width: 100%;
  text-align: center;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background-color: #eee;
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.9rem;
  color: #666;
  margin: 0.25rem 0;
}

.upload-status {
  font-size: 0.8rem;
  color: #666;
  margin: 0;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.is-searching {
  position: relative;
}

.search-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
}

.search-spinner {
  width: 30px;
  height: 30px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #4CAF50;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.image-actions {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 2;
}

.delete-button {
  background: rgba(255, 0, 0, 0.7);
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
}

.delete-button:hover {
  background: rgba(255, 0, 0, 0.9);
}

.search-score {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #555;
  position: relative;
}

.score-bar {
  height: 4px;
  background-color: #4CAF50;
  border-radius: 2px;
  margin-bottom: 4px;
}

.score-label {
  font-weight: 500;
  color: #4CAF50;
}

.terms-matched {
  margin-left: 0.5rem;
  font-size: 0.75rem;
  color: #777;
}
</style> 