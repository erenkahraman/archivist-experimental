<template>
  <div class="modal" @click="$emit('close')">
    <div class="modal-content glass animate__animated animate__fadeIn" @click.stop>
      <button class="close-button" @click="$emit('close')" aria-label="Close modal">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      
      <div class="modal-layout">
        <!-- Image Column -->
        <div class="image-column">
          <div class="image-container">
            <div v-if="isLoading" class="image-loading">
              <div class="loading-spinner"></div>
              <p>Loading image...</p>
            </div>
            <img 
              :src="getImageUrl(selectedImage.original_path)" 
              :alt="getImageName(selectedImage.original_path)"
              class="main-image"
              @load="handleImageLoaded"
              @error="handleImageError"
              :class="{ 'hidden': isLoading }"
              ref="imageElement"
            >
            <div v-if="loadError" class="image-error">
              <button @click="tryLoadThumbnail" class="try-thumbnail-btn">
                Try Thumbnail
              </button>
            </div>
          </div>
          <h2 class="image-name">{{ getImageName(selectedImage.original_path) }}</h2>
        </div>
        
        <!-- Details Column -->
        <div class="details-column">
          <!-- Tabs Navigation -->
          <div class="tabs-nav">
            <button 
              v-for="tab in tabs" 
              :key="tab.id"
              class="tab-button"
              :class="{ 'active': activeTab === tab.id }"
              @click="activeTab = tab.id"
            >
              {{ tab.label }}
            </button>
          </div>
          
          <!-- Tab Content -->
          <div class="tab-content">
            <!-- Patterns Tab -->
            <div v-if="activeTab === 'patterns'" class="tab-pane">
              <div class="pattern-section" v-if="selectedImage.patterns?.primary_pattern">
                <h3 class="section-title">Primary Pattern</h3>
                <div class="pattern-primary">
                  {{ typeof selectedImage.patterns.primary_pattern === 'object' ? 
                     selectedImage.patterns.primary_pattern.name : 
                     selectedImage.patterns.primary_pattern }}
                  <span class="inline-confidence">{{ Math.round((selectedImage.patterns.category_confidence || 0) * 100) }}%</span>
                </div>
              </div>
              
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
              
              <div class="pattern-section" v-if="selectedImage.patterns?.prompt">
                <h3 class="section-title">Pattern Description</h3>
                <p class="prompt-text">{{ getPromptText(selectedImage.patterns.prompt) }}</p>
              </div>
            </div>
            
            <!-- Colors Tab -->
            <div v-if="activeTab === 'colors'" class="tab-pane">
              <div class="color-section" v-if="getDominantColor()">
                <h3 class="section-title">Dominant Color</h3>
                <div class="color-display">
                  <div 
                    class="color-preview" 
                    :style="{ backgroundColor: extractHexColor(getDominantColor()) }"
                  ></div>
                  <div class="color-info">
                    <span class="color-value">{{ extractHexColor(getDominantColor()) }}</span>
                    <span class="color-name" v-if="extractColorName(getDominantColor())">{{ extractColorName(getDominantColor()) }}</span>
                    <span class="color-rgb" v-if="extractRGB(getDominantColor())">RGB: {{ extractRGB(getDominantColor()) }}</span>
                  </div>
                </div>
              </div>
              
              <div class="color-section" v-if="getColorPalette().length > 0">
                <h3 class="section-title">Color Palette</h3>
                <div class="color-palette-grid">
                  <div 
                    v-for="(color, index) in getColorPalette()" 
                    :key="index"
                    class="palette-color"
                  >
                    <div 
                      class="color-swatch" 
                      :style="{ backgroundColor: extractHexColor(color) }"
                    ></div>
                    <div class="color-info">
                      <span class="color-hex">{{ extractHexColor(color) }}</span>
                      <span class="color-name" v-if="extractColorName(color)">{{ extractColorName(color) }}</span>
                      <span class="color-rgb" v-if="extractRGB(color)">RGB: {{ extractRGB(color) }}</span>
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
                    <td>{{ getImageName(selectedImage.original_path) }}</td>
                  </tr>
                  <tr v-if="selectedImage.metadata?.width && selectedImage.metadata?.height">
                    <td class="details-label">Dimensions:</td>
                    <td>{{ selectedImage.metadata.width }} × {{ selectedImage.metadata.height }} px</td>
                  </tr>
                  <tr v-if="selectedImage.created_at">
                    <td class="details-label">Added:</td>
                    <td>{{ formatDate(selectedImage.created_at) }}</td>
                  </tr>
                  <tr>
                    <td class="details-label">Path:</td>
                    <td class="file-path">{{ selectedImage.original_path }}</td>
                  </tr>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps, defineEmits, ref, computed, onMounted } from 'vue'

const props = defineProps({
  selectedImage: {
    type: Object,
    required: true
  }
})

defineEmits(['close'])

// Image loading state
const imageElement = ref(null)
const isLoading = ref(true)
const loadError = ref(false)
const useThumbnail = ref(false)

// Tab management
const tabs = [
  { id: 'patterns', label: 'Patterns' },
  { id: 'colors', label: 'Colors' },
  { id: 'details', label: 'Details' }
]
const activeTab = ref('patterns')

// Data availability checks
const hasSecondaryPatterns = computed(() => {
  return props.selectedImage.patterns?.secondary_patterns && 
    (Array.isArray(props.selectedImage.patterns.secondary_patterns) || 
    typeof props.selectedImage.patterns.secondary_patterns === 'object')
})

// Core data functions
const getImageUrl = (path) => {
  if (useThumbnail.value || !path) {
    return getThumbnailUrl(props.selectedImage.thumbnail_path)
  }
  const filename = path.split('/').pop()
  return `http://localhost:8000/api/images/${filename}`
}

const getThumbnailUrl = (path) => {
  if (!path) return ''
  const filename = path.split('/').pop()
  return `http://localhost:8000/api/thumbnails/${filename}`
}

const getImageName = (path) => {
  if (!path) return 'Unknown Image'
  return path.split('/').pop()
}

const getPromptText = (prompt) => {
  if (!prompt) return 'No description available'
  return typeof prompt === 'string' ? prompt : (prompt.final_prompt || 'No description available')
}

// Pattern handling
const getSecondaryPatterns = () => {
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

// Color handling
const getDominantColor = () => {
  if (props.selectedImage.colors?.dominant_color) {
    return props.selectedImage.colors.dominant_color
  }
  
  if (props.selectedImage.colors?.palette && props.selectedImage.colors.palette.length > 0) {
    return props.selectedImage.colors.palette[0]
  }
  
  if (props.selectedImage.patterns?.colors?.dominant_color) {
    return props.selectedImage.patterns.colors.dominant_color
  }
  
  return null
}

const getColorPalette = () => {
  if (props.selectedImage.colors?.palette && props.selectedImage.colors.palette.length > 0) {
    return props.selectedImage.colors.palette
  }
  
  if (props.selectedImage.colors?.dominant_colors && props.selectedImage.colors.dominant_colors.length > 0) {
    return props.selectedImage.colors.dominant_colors
  }
  
  if (props.selectedImage.patterns?.colors?.palette && props.selectedImage.patterns.colors.palette.length > 0) {
    return props.selectedImage.patterns.colors.palette
  }
  
  return []
}

const extractHexColor = (color) => {
  if (!color) return '#cccccc'
  
  if (typeof color === 'string') return color
  
  if (typeof color === 'object' && color.hex) return color.hex
  
  try {
    const colorStr = JSON.stringify(color)
    const hexMatch = colorStr.match(/"hex":"(#[0-9a-fA-F]{6})"/i)
    if (hexMatch && hexMatch[1]) return hexMatch[1]
  } catch (e) {}
  
  return '#cccccc'
}

const extractColorName = (color) => {
  if (!color) return ''
  
  if (typeof color === 'object' && color.name) return color.name
  
  try {
    const colorStr = JSON.stringify(color)
    const nameMatch = colorStr.match(/"name":"([^"]+)"/i)
    if (nameMatch && nameMatch[1]) return nameMatch[1]
  } catch (e) {}
  
  return ''
}

const extractRGB = (color) => {
  if (!color) return null
  
  if (typeof color === 'object' && Array.isArray(color.rgb)) {
    return color.rgb.join(', ')
  }
  
  try {
    const colorStr = JSON.stringify(color)
    const rgbMatch = colorStr.match(/"rgb":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/i)
    if (rgbMatch && rgbMatch.length >= 4) {
      return `${rgbMatch[1]}, ${rgbMatch[2]}, ${rgbMatch[3]}`
    }
  } catch (e) {}
  
  return null
}

const extractProportion = (color) => {
  if (!color) return null
  
  if (typeof color === 'object' && typeof color.proportion === 'number') {
    return color.proportion
  }
  
  try {
    const colorStr = JSON.stringify(color)
    const proportionMatch = colorStr.match(/"proportion":([0-9.]+)/i)
    if (proportionMatch && proportionMatch[1]) {
      return parseFloat(proportionMatch[1])
    }
  } catch (e) {}
  
  return null
}

const formatProportion = (proportion) => {
  if (proportion === null || proportion === undefined) return ''
  return `${Math.round(proportion * 100)}%`
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
  isLoading.value = false
  loadError.value = false
}

const handleImageError = () => {
  if (!useThumbnail.value) {
    tryLoadThumbnail()
  } else {
    isLoading.value = false
    loadError.value = true
  }
}

const tryLoadThumbnail = () => {
  useThumbnail.value = true
  isLoading.value = true
  loadError.value = false
  
  if (imageElement.value) {
    const thumbnailUrl = getThumbnailUrl(props.selectedImage.thumbnail_path)
    if (thumbnailUrl) {
      setTimeout(() => {
        imageElement.value.src = thumbnailUrl
      }, 10)
    } else {
      isLoading.value = false
      loadError.value = true
    }
  }
}

onMounted(() => {
  isLoading.value = true
  loadError.value = false
  useThumbnail.value = false
})
</script>

<style scoped>
.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(5px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-4);
}

.modal-content {
  width: 95%;
  max-width: 1200px;
  max-height: 90vh;
  border-radius: var(--radius-lg);
  overflow: hidden;
  position: relative;
  background-color: rgba(244, 244, 248, 0.95);
  box-shadow: var(--shadow-xl);
  color: #333;
}

.close-button {
  position: absolute;
  top: var(--space-3);
  right: var(--space-3);
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

.image-container {
  flex: 1;
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-md);
  background-color: white;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  margin-bottom: var(--space-3);
}

.main-image {
  max-width: 100%;
  max-height: 70vh;
  object-fit: contain;
  transition: opacity 0.3s;
}

.main-image.hidden {
  opacity: 0;
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
}

.try-thumbnail-btn:hover {
  opacity: 0.9;
  transform: translateY(-1px);
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

.prompt-text {
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
}

.color-display {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-top: var(--space-2);
  padding: var(--space-3);
  background-color: #f9f9fb;
  border-radius: var(--radius-md);
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.color-preview {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.color-info {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.color-value {
  font-family: monospace;
  font-size: 0.95rem;
  font-weight: 600;
  color: #333;
}

.color-name {
  font-size: 0.85rem;
  color: #555;
}

.color-rgb {
  font-family: monospace;
  font-size: 0.85rem;
  color: #666;
}

.color-proportion {
  font-size: 0.8rem;
  color: #777;
}

.color-palette-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: var(--space-3);
  margin-top: var(--space-3);
}

.palette-color {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2);
  background-color: #f9f9fb;
  border-radius: var(--radius-md);
  border: 1px solid rgba(0, 0, 0, 0.05);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.palette-color:hover {
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
</style> 