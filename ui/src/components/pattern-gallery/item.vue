<template>
  <div 
    class="gallery-item"
    :class="{
      'is-uploading': image.isUploading,
      'is-searching': image.isSearching
    }"
  >
    <UploadingItem 
      v-if="image.isUploading" 
      :progress="image.uploadProgress" 
      :status="image.uploadStatus" 
    />
    
    <template v-else>
      <div class="image-actions">
        <button 
          class="delete-button"
          @click.stop="$emit('delete')"
          title="Delete image"
        >×</button>
      </div>
      
      <img 
        :src="getThumbnailUrl(image.thumbnail_path)" 
        :alt="getImageName(image.original_path)"
        class="gallery-image"
        @click="$emit('select')"
      >
      
      <div class="image-metadata">
        <div class="pattern-type">
          <span class="type-label">Pattern:</span>
          <span class="type-value">{{ image.patterns?.primary_pattern || 'Unknown' }}</span>
        </div>
        <div class="pattern-prompt">
          {{ truncatePrompt(image.patterns?.prompt) }}
        </div>
        
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
</template>

<script setup>
import UploadingItem from './uploading-item.vue'
import { getThumbnailUrl, getImageName, truncatePrompt } from './utils/helpers.js'

defineProps({
  image: {
    type: Object,
    required: true
  },
  searchActive: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select', 'delete'])
</script>

<style scoped>
.gallery-item {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
  cursor: pointer;
  position: relative;
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

.is-uploading {
  opacity: 0.7;
  pointer-events: none;
}

.is-searching {
  position: relative;
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