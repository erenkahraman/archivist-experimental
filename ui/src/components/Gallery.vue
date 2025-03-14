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

    <!-- Modal -->
    <div v-if="selectedImage" class="modal" @click="selectedImage = null">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h2>{{ selectedImage.patterns?.primary_pattern || 'Unknown Pattern' }}</h2>
          <button class="close-button" @click="selectedImage = null">×</button>
        </div>

        <div class="modal-body">
          <div class="modal-grid">
            <!-- Left side: Image and basic information -->
            <div class="modal-left">
              <div class="image-preview">
                <img 
                  :src="getThumbnailUrl(selectedImage.thumbnail_path)" 
                  :alt="selectedImage.patterns?.primary_pattern"
                  class="modal-image"
                >
              </div>
              
              <div class="basic-info">
                <div class="pattern-header">
                  <h3>{{ selectedImage.patterns?.primary_pattern }}</h3>
                  <div class="confidence">
                    {{ (selectedImage.patterns?.confidence * 100).toFixed(1) }}% confidence
                  </div>
                </div>
              </div>
            </div>

            <!-- Right side: Detailed analysis -->
            <div class="modal-right">
              <div class="analysis-tabs">
                <button 
                  v-for="tab in ['Pattern', 'Colors', 'Details']" 
                  :key="tab"
                  :class="{ active: activeTab === tab }"
                  @click="activeTab = tab"
                >
                  {{ tab }}
                </button>
              </div>

              <div class="tab-content">
                <!-- Pattern Analysis Tab -->
                <div v-if="activeTab === 'Pattern'" class="pattern-analysis">
                  <section class="analysis-section">
                    <h4>Primary Pattern</h4>
                    <div class="pattern-chips">
                      <div class="pattern-chip primary">
                        {{ selectedImage.patterns?.primary_pattern }}
                      </div>
                    </div>
                  </section>

                  <section class="analysis-section">
                    <h4>Secondary Patterns</h4>
                    <div class="pattern-chips">
                      <div 
                        v-for="pattern in selectedImage.patterns?.secondary_patterns" 
                        :key="pattern.name"
                        class="pattern-chip"
                      >
                        <span>{{ pattern.name }}</span>
                        <span class="confidence-badge">
                          {{ (pattern.confidence * 100).toFixed(0) }}%
                        </span>
                      </div>
                    </div>
                  </section>

                  <section class="analysis-section">
                    <h4>Pattern Description</h4>
                    <p>{{ getPromptText(selectedImage.patterns?.prompt) }}</p>
                  </section>
                </div>

                <!-- Colors Analysis Tab -->
                <div v-if="activeTab === 'Colors'" class="color-analysis">
                  <section class="analysis-section">
                    <h4>Color Palette</h4>
                    <div class="color-grid">
                      <div 
                        v-for="color in selectedImage.colors?.dominant_colors" 
                        :key="color.hex"
                        class="color-item"
                      >
                        <div 
                          class="color-preview" 
                          :style="{ backgroundColor: color.hex }"
                        ></div>
                        <div class="color-info">
                          <span class="color-name">{{ color.name }}</span>
                          <span class="color-percentage">
                            {{ (color.proportion * 100).toFixed(1) }}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </section>
                </div>

                <!-- Details Tab -->
                <div v-if="activeTab === 'Details'" class="details-analysis">
                  <!-- Pattern Analysis -->
                  <section class="analysis-section">
                    <h4>Pattern Characteristics</h4>
                    <div class="characteristics-grid">
                      <div class="characteristic-item">
                        <span class="label">Layout:</span>
                        <span>{{ selectedImage.patterns?.layout?.type }}</span>
                        <div class="confidence-bar" :style="{ width: `${selectedImage.patterns?.layout?.confidence * 100}%` }"></div>
                      </div>
                      <div class="characteristic-item">
                        <span class="label">Scale:</span>
                        <span>{{ selectedImage.patterns?.scale?.type }}</span>
                        <div class="confidence-bar" :style="{ width: `${selectedImage.patterns?.scale?.confidence * 100}%` }"></div>
                      </div>
                      <div class="characteristic-item">
                        <span class="label">Texture:</span>
                        <span>{{ selectedImage.patterns?.texture_type?.type }}</span>
                        <div class="confidence-bar" :style="{ width: `${selectedImage.patterns?.texture_type?.confidence * 100}%` }"></div>
                      </div>
                    </div>
                  </section>

                  <!-- Design Information -->
                  <section class="analysis-section">
                    <h4>Design Information</h4>
                    <div class="design-info-grid">
                      <div class="info-item">
                        <span class="label">Dimensions:</span>
                        <span>{{ selectedImage.dimensions?.width }}x{{ selectedImage.dimensions?.height }}px</span>
                      </div>
                      <div class="info-item">
                        <span class="label">Aspect Ratio:</span>
                        <span>{{ calculateAspectRatio(selectedImage.dimensions) }}</span>
                      </div>
                      <div class="info-item">
                        <span class="label">File Name:</span>
                        <span>{{ getImageName(selectedImage.original_path) }}</span>
                      </div>
                    </div>
                  </section>

                  <!-- Color Analysis -->
                  <section class="analysis-section">
                    <h4>Color Analysis</h4>
                    <div class="color-analysis-grid">
                      <!-- Color Distribution -->
                      <div class="color-stats">
                        <h5>Color Distribution</h5>
                        <div class="color-bars">
                          <div v-for="color in selectedImage.colors?.dominant_colors" :key="color.hex" class="color-bar-item">
                            <div class="color-bar-label">
                              <span class="color-preview" :style="{ backgroundColor: color.hex }"></span>
                              <span>{{ color.name }}</span>
                            </div>
                            <div class="color-bar-container">
                              <div class="color-bar-fill" 
                                   :style="{ width: `${color.proportion * 100}%`, backgroundColor: color.hex }">
                              </div>
                              <span class="color-percentage">{{ (color.proportion * 100).toFixed(1) }}%</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <!-- RGB Values -->
                      <div class="rgb-values" v-if="selectedImage.colors?.dominant_colors">
                        <h5>RGB Values</h5>
                        <div class="rgb-grid">
                          <div v-for="color in selectedImage.colors.dominant_colors" :key="color.hex" class="rgb-item">
                            <div class="color-preview" :style="{ backgroundColor: color.hex }"></div>
                            <div class="rgb-info">
                              <span>R: {{ color.rgb[0] }}</span>
                              <span>G: {{ color.rgb[1] }}</span>
                              <span>B: {{ color.rgb[2] }}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </section>

                  <!-- Style Analysis -->
                  <section class="analysis-section">
                    <h4>Style Analysis</h4>
                    <div class="style-grid">
                      <div class="style-item">
                        <span class="label">Cultural Influence:</span>
                        <span>{{ selectedImage.patterns?.cultural_influence?.type }}</span>
                        <div class="confidence-bar" 
                             :style="{ width: `${selectedImage.patterns?.cultural_influence?.confidence * 100}%` }">
                        </div>
                      </div>
                      <div class="style-item">
                        <span class="label">Historical Period:</span>
                        <span>{{ selectedImage.patterns?.historical_period?.type }}</span>
                        <div class="confidence-bar" 
                             :style="{ width: `${selectedImage.patterns?.historical_period?.confidence * 100}%` }">
                        </div>
                      </div>
                      <div class="style-item">
                        <span class="label">Mood:</span>
                        <span>{{ selectedImage.patterns?.mood?.type }}</span>
                        <div class="confidence-bar" 
                             :style="{ width: `${selectedImage.patterns?.mood?.confidence * 100}%` }">
                        </div>
                      </div>
                      <div class="style-item">
                        <span class="label">Style Keywords:</span>
                        <div class="keyword-chips">
                          <span v-for="keyword in selectedImage.patterns?.style_keywords" 
                                :key="keyword" 
                                class="keyword-chip">
                            {{ keyword }}
                          </span>
                        </div>
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

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

const imageStore = useImageStore()
const selectedImage = ref(null)
const showDeleteConfirm = ref(false)
const imageToDelete = ref(null)
const activeTab = ref('Pattern')

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
  console.log('Pattern analysis:', image.patterns);
  console.log('Color analysis:', image.colors);
  selectedImage.value = image
  activeTab.value = 'Pattern'
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

const calculateAspectRatio = (dimensions) => {
  if (!dimensions?.width || !dimensions?.height) return 'N/A'
  const gcd = (a, b) => b ? gcd(b, a % b) : a
  const divisor = gcd(dimensions.width, dimensions.height)
  return `${dimensions.width/divisor}:${dimensions.height/divisor}`
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

.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  overflow: hidden;
}

.modal-content {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 1200px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}

.modal-header {
  padding: 1.5rem;
  background: #f8f9fa;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 10;
}

.modal-header h2 {
  margin: 0;
  font-size: 1.5rem;
  color: #2c3e50;
}

.close-button {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #666;
  cursor: pointer;
  padding: 0.5rem;
}

.modal-body {
  padding: 2rem;
  overflow-y: auto;
  flex: 1;
}

.modal-grid {
  display: grid;
  grid-template-columns: minmax(300px, 1fr) 2fr;
  gap: 2rem;
}

.modal-left {
  position: sticky;
  top: 1rem;
  height: fit-content;
}

.image-preview {
  margin-bottom: 1.5rem;
}

.modal-image {
  width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.modal-right {
  display: flex;
  flex-direction: column;
}

.analysis-tabs {
  position: sticky;
  top: 0;
  background: white;
  padding: 1rem 0;
  margin-bottom: 1rem;
  z-index: 5;
  border-bottom: 1px solid #eee;
}

.analysis-tabs button {
  padding: 0.5rem 1.5rem;
  border: none;
  background: none;
  color: #666;
  cursor: pointer;
  font-size: 1rem;
  position: relative;
}

.analysis-tabs button.active {
  color: #2c3e50;
  font-weight: 600;
}

.analysis-tabs button.active::after {
  content: '';
  position: absolute;
  bottom: -1rem;
  left: 0;
  right: 0;
  height: 2px;
  background: #4CAF50;
}

.pattern-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.pattern-chip {
  padding: 0.5rem 1rem;
  background: #f1f1f1;
  border-radius: 16px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.pattern-chip.primary {
  background: #4CAF50;
  color: white;
}

.confidence-badge {
  background: rgba(0,0,0,0.1);
  padding: 0.2rem 0.5rem;
  border-radius: 10px;
  font-size: 0.8rem;
}

.color-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
}

.color-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 8px;
  position: relative;
}

.color-info {
  display: flex;
  align-items: center;
  flex: 1;
  position: relative;
  justify-content: space-between;
  padding-right: 10px;
}

.color-name {
  margin-right: 40px;
}

.color-percentage {
  color: #000000;
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
}

.color-preview {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.characteristics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.characteristic-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.characteristic-item .label {
  color: #666;
  font-size: 0.9rem;
  display: block;
  margin-bottom: 0.3rem;
}

.analysis-section {
  margin-bottom: 2rem;
}

.analysis-section h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.pattern-description {
  line-height: 1.6;
  color: #2c3e50;
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
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

@media (max-width: 1024px) {
  .modal-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }

  .modal-left {
    position: relative;
    top: 0;
  }

  .modal-image {
    max-height: 400px;
    object-fit: contain;
  }
}

@media (max-width: 768px) {
  .modal-content {
    width: 95%;
    max-height: 95vh;
  }

  .modal-body {
    padding: 1rem;
  }

  .analysis-tabs {
    padding: 0.5rem 0;
  }

  .analysis-tabs button {
    padding: 0.5rem;
    font-size: 0.9rem;
  }

  .characteristics-grid,
  .design-info-grid {
    grid-template-columns: 1fr;
  }

  .color-grid {
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  }
}

.tab-content {
  padding-top: 1rem;
}

.analysis-section {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.modal-body::-webkit-scrollbar {
  width: 8px;
}

.modal-body::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.modal-body::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

.modal-body::-webkit-scrollbar-thumb:hover {
  background: #666;
}

.full-prompt {
  font-size: 1rem;
  line-height: 1.6;
  color: #2c3e50;
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin-top: 1rem;
  white-space: pre-wrap;
}

.prompt-analysis {
  padding: 1rem;
}

.prompt-analysis h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.confidence-bar {
  height: 4px;
  background: #4CAF50;
  border-radius: 2px;
  margin-top: 4px;
  transition: width 0.3s ease;
}

.color-analysis-grid {
  display: grid;
  gap: 2rem;
}

.color-bars {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.color-bar-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.color-bar-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.color-bar-container {
  height: 24px;
  background: #f1f1f1;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}

.color-bar-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.color-percentage {
  color: #000000;
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
}

.rgb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 1rem;
}

.rgb-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.rgb-info {
  display: flex;
  flex-direction: column;
  font-size: 0.8rem;
  color: #666;
}

.keyword-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.keyword-chip {
  padding: 0.3rem 0.8rem;
  background: #f1f1f1;
  border-radius: 16px;
  font-size: 0.8rem;
  color: #666;
}

.design-info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.info-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

h5 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-size: 1rem;
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