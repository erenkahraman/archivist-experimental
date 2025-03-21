<template>
  <div v-if="selectedImage" class="modal" @click="$emit('close')">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h2>{{ selectedImage.patterns?.primary_pattern || 'Unknown Pattern' }}</h2>
        <button class="close-button" @click="$emit('close')">×</button>
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
                  {{ (selectedImage.patterns?.pattern_confidence * 100).toFixed(1) }}% confidence
                </div>
              </div>
            </div>
          </div>

          <!-- Right side: Detailed analysis -->
          <div class="modal-right">
            <div class="analysis-tabs">
              <button 
                v-for="tab in ['Pattern', 'Colors', 'Pantone', 'Details']" 
                :key="tab"
                :class="{ active: activeTab === tab }"
                @click="activeTab = tab"
              >
                {{ tab }}
              </button>
            </div>

            <div class="tab-content">
              <!-- Pattern Analysis Tab -->
              <PatternModal
                v-if="activeTab === 'Pattern'"
                :primary-pattern="selectedImage.patterns?.primary_pattern"
                :secondary-patterns="selectedImage.patterns?.secondary_patterns || []"
                :pattern-description="getPromptText(selectedImage.patterns?.prompt)"
              />

              <!-- Colors Analysis Tab -->
              <ColorModal
                v-if="activeTab === 'Colors'"
                :colors="selectedImage.colors?.dominant_colors || []"
              />

              <!-- Pantone Analysis Tab -->
              <PantoneModal
                v-if="activeTab === 'Pantone'"
                :selected-image="selectedImage"
              />

              <!-- Details Tab -->
              <DetailsModal
                v-if="activeTab === 'Details'"
                :image-patterns="selectedImage.patterns"
                :dimensions="selectedImage.dimensions"
                :image-name="getImageName(selectedImage.original_path)"
                :colors="selectedImage.colors?.dominant_colors || []"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import PatternModal from './PatternModal.vue'
import ColorModal from './ColorModal.vue'
import PantoneModal from './PantoneModal.vue'
import DetailsModal from './DetailsModal.vue'

const props = defineProps({
  selectedImage: {
    type: Object,
    default: null
  }
})

defineEmits(['close'])

const activeTab = ref('Pattern')

const getThumbnailUrl = (path) => {
  if (!path) {
    console.log('Warning: Empty thumbnail path')
    return ''
  }
  return `http://localhost:8000/api/thumbnails/${path.split('/').pop()}`
}

const getImageName = (path) => {
  if (!path) return 'Unknown'
  return path.split('/').pop()
}

const getPromptText = (prompt, truncate = false) => {
  if (!prompt) return 'No description available'
  
  // Handle both string and object formats
  const promptText = typeof prompt === 'string' 
    ? prompt 
    : (prompt.final_prompt || 'No description available')
    
  if (truncate && promptText.length > 100) {
    return promptText.substring(0, 100) + '...'
  }
  
  return promptText
}
</script>

<style scoped>
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

.tab-content {
  padding-top: 1rem;
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

.basic-info {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.pattern-header {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.pattern-header h3 {
  margin: 0;
  color: #2c3e50;
}

.confidence {
  color: #4CAF50;
  font-weight: 500;
  font-size: 0.9rem;
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
}
</style> 