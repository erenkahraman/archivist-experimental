<template>
  <div class="modal" @click="$emit('close')">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h2>{{ image.patterns?.primary_pattern || 'Unknown Pattern' }}</h2>
        <button class="close-button" @click="$emit('close')">×</button>
      </div>

      <div class="modal-body">
        <div class="modal-grid">
          <!-- Left side: Image and basic information -->
          <div class="modal-left">
            <div class="image-preview">
              <img 
                :src="getThumbnailUrl(image.thumbnail_path)" 
                :alt="image.patterns?.primary_pattern || 'Pattern preview'"
                class="modal-image"
              >
            </div>
            
            <div class="basic-info">
              <div class="pattern-header">
                <h3>{{ image.patterns?.primary_pattern }}</h3>
                <div class="confidence">
                  {{ (image.patterns?.confidence * 100).toFixed(1) }}% confidence
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
              <PatternTab v-if="activeTab === 'Pattern'" :image="image" />
              <ColorsTab v-if="activeTab === 'Colors'" :image="image" />
              <DetailsTab v-if="activeTab === 'Details'" :image="image" />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { getThumbnailUrl } from '../utils/helpers.js'
import PatternTab from './tabs/pattern-tab.vue'
import ColorsTab from './tabs/colors-tab.vue'
import DetailsTab from './tabs/details-tab.vue'

defineProps({
  image: {
    type: Object,
    required: true
  }
})

defineEmits(['close'])

const activeTab = ref('Pattern')
</script>

<style scoped>
.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 1200px;
  max-height: 90vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  padding: 1.5rem;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
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
  cursor: pointer;
  color: #666;
}

.modal-body {
  padding: 1.5rem;
  overflow-y: auto;
  flex: 1;
}

.modal-grid {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 2rem;
}

.modal-left {
  position: sticky;
  top: 0;
}

.modal-image {
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.pattern-header {
  margin-top: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.pattern-header h3 {
  margin: 0;
  color: #2c3e50;
}

.confidence {
  background: #f8f9fa;
  padding: 0.3rem 0.8rem;
  border-radius: 16px;
  font-size: 0.8rem;
  color: #4CAF50;
}

.analysis-tabs {
  display: flex;
  border-bottom: 1px solid #eee;
  margin-bottom: 1.5rem;
  padding: 0.5rem 0;
}

.analysis-tabs button {
  background: none;
  border: none;
  padding: 0.5rem 1rem;
  margin-right: 1rem;
  cursor: pointer;
  font-size: 1rem;
  color: #666;
  border-bottom: 2px solid transparent;
}

.analysis-tabs button.active {
  color: #4CAF50;
  border-bottom: 2px solid #4CAF50;
}

.tab-content {
  min-height: 300px;
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