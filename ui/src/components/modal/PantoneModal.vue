<template>
  <section class="analysis-section">
    <h4>Pantone Matches</h4>
    
    <div class="pantone-controls">
      <label for="pantone-catalog">Select Catalog:</label>
      <select 
        id="pantone-catalog" 
        v-model="selectedCatalog"
        @change="updatePantoneColors"
      >
        <option value="">All Catalogs</option>
        <option v-for="catalog in pantoneCatalogs" :key="catalog" :value="catalog">
          {{ catalog }}
        </option>
      </select>
      
      <div v-if="selectedCatalog && getSelectedCatalogInfo()" class="catalog-info">
        <span class="catalog-colors-count">{{ getSelectedCatalogInfo().colors_count }} colors</span>
      </div>
    </div>
    
    <div v-if="loadingPantone" class="pantone-loading">
      <div class="loading-spinner"></div>
      <p>Converting colors...</p>
    </div>
    
    <div v-else-if="!pantoneColors.length" class="pantone-empty">
      <p v-if="pantoneCatalogs.length === 0">
        No Pantone catalogs available in the catalogs folder.
      </p>
      <p v-else-if="!selectedImage">
        Please select an image first to convert its colors to Pantone.
      </p>
      <p v-else>
        Click "Convert Colors" to find Pantone matches using the selected catalog.
      </p>
      <button 
        class="convert-colors-btn"
        @click="convertToPantone"
        :disabled="loadingPantone || !selectedImage"
      >
        Convert Colors
      </button>
    </div>
    
    <div v-else class="pantone-grid">
      <div 
        v-for="(color, index) in pantoneColors" 
        :key="index"
        class="pantone-item"
      >
        <div class="pantone-color-preview">
          <div class="color-original" :style="{ backgroundColor: color.hex }"></div>
          <div class="color-arrow">→</div>
          <div class="color-pantone" :style="{ backgroundColor: color.pantone?.pantone_hex }"></div>
        </div>
        
        <div class="pantone-info">
          <div class="pantone-name">
            {{ color.pantone?.pantone_name || 'No match' }}
          </div>
          <div class="pantone-match-quality">
            Match: {{ Math.round(color.pantone?.match_quality || 0) }}%
          </div>
          <div class="pantone-catalog">
            Catalog: {{ color.pantone?.source_catalog || 'N/A' }}
          </div>
          <div class="pantone-rgb">
            RGB: {{ color.pantone?.pantone_rgb?.join(', ') || 'N/A' }}
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'

const props = defineProps({
  selectedImage: {
    type: Object,
    default: null
  }
})

// Pantone-related state
const pantoneCatalogs = ref([])
const catalogInfo = ref([])
const selectedCatalog = ref('')
const pantoneColors = ref([])
const loadingPantone = ref(false)

// Watch for changes in the selected image
watch(() => props.selectedImage, (newImage) => {
  if (newImage) {
    // Clear Pantone colors when a new image is selected
    pantoneColors.value = []
  }
})

onMounted(() => {
  fetchPantoneCatalogs()
})

// Pantone-related functions
const fetchPantoneCatalogs = async () => {
  try {
    console.log('Fetching available Pantone catalogs...')
    const response = await fetch('http://localhost:8000/api/pantone/catalogs/info')
    
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    if (data.status === 'success') {
      // Store the detailed catalog info
      catalogInfo.value = data.catalog_info || []
      console.log(`Found ${catalogInfo.value.length} Pantone catalogs:`, catalogInfo.value)
      
      // Extract just the catalog names for the dropdown
      pantoneCatalogs.value = catalogInfo.value.map(catalog => catalog.name)
      
      // If we have catalogs but none selected, select the first one
      if (pantoneCatalogs.value.length > 0 && !selectedCatalog.value) {
        selectedCatalog.value = pantoneCatalogs.value[0]
        console.log(`Auto-selected catalog: ${selectedCatalog.value}`)
      }
    } else {
      console.error('Error fetching Pantone catalogs:', data.error)
    }
  } catch (error) {
    console.error('Error fetching Pantone catalogs:', error)
  }
}

const convertToPantone = async () => {
  if (!props.selectedImage || !props.selectedImage.colors?.dominant_colors) {
    console.error('No image selected or no color data available')
    alert('No image selected or no color data available')
    return
  }
  
  if (!props.selectedImage.original_path) {
    console.error('Image path is undefined')
    
    // Try to fix the path if possible
    if (props.selectedImage.path) {
      console.log('Using image.path instead:', props.selectedImage.path)
      props.selectedImage.original_path = props.selectedImage.path
    } else if (props.selectedImage.thumbnail_path) {
      // Derive original path from thumbnail path
      const filename = props.selectedImage.thumbnail_path.split('/').pop()
      props.selectedImage.original_path = `uploads/${filename}`
      console.log('Derived path from thumbnail:', props.selectedImage.original_path)
    } else {
      alert('Image path is undefined. Please select a valid image.')
      return
    }
  }
  
  if (pantoneCatalogs.value.length === 0) {
    alert('No Pantone catalogs available. Please add .cat files to the catalogs folder.')
    return
  }
  
  loadingPantone.value = true
  pantoneColors.value = []
  
  try {
    console.log('Converting colors to Pantone for image:', props.selectedImage.original_path)
    console.log('Using catalog:', selectedCatalog.value || 'All catalogs')
    
    // Extract just the filename from the path
    const imagePath = props.selectedImage.original_path
    const filename = imagePath.split('/').pop()
    
    console.log('Image filename:', filename)
    
    // Build the URL with the catalog parameter if one is selected
    let url = `http://localhost:8000/api/pantone/analyze-image/${encodeURIComponent(filename)}`
    if (selectedCatalog.value) {
      url += `?catalog_name=${encodeURIComponent(selectedCatalog.value)}`
    }
    
    console.log('Sending request to:', url)
    
    const response = await fetch(url)
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error('Server error response:', errorData)
      throw new Error(errorData.message || `Server returned ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()
    console.log('Server response:', data)
    
    if (data.status === 'success') {
      console.log('Pantone conversion successful:', data.pantone_colors.length, 'colors converted')
      console.log('Using catalog:', data.catalog_used)
      pantoneColors.value = data.pantone_colors || []
      
      if (pantoneColors.value.length === 0) {
        alert('No Pantone matches found for the selected colors.')
      }
    } else {
      console.error('Error converting to Pantone:', data.error || data.message)
      pantoneColors.value = []
      alert(`Error converting colors: ${data.error || data.message || 'Unknown error'}`)
    }
  } catch (error) {
    console.error('Error converting to Pantone:', error)
    pantoneColors.value = []
    alert(`Error converting colors: ${error.message || 'Unknown error'}`)
  } finally {
    loadingPantone.value = false
  }
}

const updatePantoneColors = () => {
  // Clear current pantone colors when changing catalog
  pantoneColors.value = []
  console.log(`Selected catalog changed to: ${selectedCatalog.value || 'All catalogs'}`)
  
  // Only convert colors if an image is selected
  if (props.selectedImage) {
    // Convert colors with the new catalog
    convertToPantone()
  }
}

const getSelectedCatalogInfo = () => {
  return catalogInfo.value.find(catalog => catalog.name === selectedCatalog.value) || {}
}
</script>

<style scoped>
.pantone-analysis {
  padding: 1.5rem;
}

.pantone-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
}

.pantone-controls label {
  font-weight: 500;
  color: #666;
  min-width: 120px;
}

.pantone-controls select {
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  min-width: 250px;
  flex-grow: 1;
  font-size: 1rem;
  background-color: white;
}

.catalog-info {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 8px;
  text-align: center;
}

.catalog-colors-count {
  font-weight: 600;
  color: #2c3e50;
}

.pantone-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  color: #666;
}

.pantone-loading .loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  border-top-color: #4CAF50;
  animation: spin 1s ease-in-out infinite;
  margin-bottom: 1rem;
}

.pantone-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  color: #666;
  text-align: center;
}

.convert-colors-btn {
  padding: 0.5rem 1rem;
  border: none;
  background: #4CAF50;
  color: white;
  cursor: pointer;
  border-radius: 4px;
  margin-top: 1rem;
  font-size: 0.9rem;
}

.convert-colors-btn:hover {
  background: #45a049;
}

.convert-colors-btn:disabled {
  background: #cccccc;
  cursor: not-allowed;
}

.pantone-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.pantone-item {
  display: flex;
  flex-direction: column;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.pantone-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.pantone-color-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}

.color-original, .color-pantone {
  width: 100px;
  height: 100px;
  border-radius: 8px;
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.color-arrow {
  font-size: 2rem;
  color: #666;
}

.pantone-info {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.pantone-name {
  font-weight: 600;
  font-size: 1.2rem;
  color: #333;
}

.pantone-match-quality {
  color: #666;
  font-size: 1rem;
  display: flex;
  align-items: center;
}

.pantone-match-quality::before {
  content: '';
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
  background-color: #4CAF50;
}

.pantone-catalog {
  color: #666;
  font-size: 1rem;
}

.pantone-rgb {
  color: #666;
  font-size: 1rem;
  font-family: monospace;
  background: rgba(0,0,0,0.05);
  padding: 0.5rem;
  border-radius: 4px;
}

.analysis-section {
  margin-bottom: 2rem;
}

.analysis-section h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style> 