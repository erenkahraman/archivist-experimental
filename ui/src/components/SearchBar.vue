<template>
  <div 
    class="search-container" 
    :class="{ 'is-dragover': isDragover }"
    @dragenter.prevent="handleDragEnter"
    @dragover.prevent="handleDragOver"
    @dragleave.prevent="handleDragLeave"
    @drop.prevent="handleDrop"
  >
    <div class="search-form">
      <div class="search-input-wrapper">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <input 
          v-model="searchQuery" 
          type="text" 
          class="search-input"
          placeholder="Search for patterns, colors (e.g. 'paisley, red, flower')"
          @keyup.enter="handleSearch"
          title="Use commas to separate search terms for more precise results (e.g., 'paisley, red, flower')"
        >
        <button 
          v-if="searchQuery" 
          class="clear-search-button"
          @click="clearSearch"
          title="Clear search"
        >
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <button 
        class="search-button"
        @click="handleSearch"
        :disabled="!searchQuery"
        title="Search the image collection using the provided terms"
      >
        Search
      </button>
    </div>

    <div class="search-options">
      <div class="search-filters">
        <div class="filter-group">
          <label class="filter-label">Results:</label>
          <select v-model="resultCount" class="result-count-select">
            <option value="10">10</option>
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </div>
        
        <div class="filter-group">
          <label class="filter-label">Min Score:</label>
          <select v-model="minSimilarity" class="min-score-select">
            <option value="0.01">Very low (0.01)</option>
            <option value="0.1">Low (0.1)</option>
            <option value="0.3">Medium (0.3)</option>
            <option value="0.5">High (0.5)</option>
            <option value="0.7">Very high (0.7)</option>
          </select>
        </div>
      </div>
      
      <div class="search-status" v-if="imageStore.searching">
        <div class="search-spinner"></div>
        <span>Searching...</span>
      </div>
      
      <div class="search-results-info" v-if="imageStore.hasSearchResults">
        <span class="result-count">
          <strong>{{ imageStore.searchResults.length }}</strong> results for 
          <span class="search-terms">{{ imageStore.searchQuery }}</span>
        </span>
        <button class="reset-button" @click="resetSearch">Reset</button>
      </div>
    </div>
    
    <!-- Quick search examples -->
    <div class="search-examples">
      <span class="examples-label">Try:</span>
      <div class="example-buttons">
        <button 
          v-for="example in searchExamples" 
          :key="example"
          class="example-button"
          @click="runExampleSearch(example)"
        >
          {{ example }}
        </button>
      </div>
    </div>

    <!-- Add an info hint about advanced search -->
    <div class="search-hint">
      <span class="hint-icon">💡</span>
      <span class="hint-text">Use commas to separate search terms for more specific results.</span>
    </div>
    
    <!-- Add a drag-and-drop hint -->
    <div class="search-hint drag-hint">
      <span class="hint-icon">🔍</span>
      <span class="hint-text">Drag any image from the gallery here to find similar patterns.</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useImageStore } from '../stores/imageStore'

const imageStore = useImageStore()

const searchQuery = ref('')
const searchType = ref('all')
const resultCount = ref(20)
const minSimilarity = ref(0.1)
const isDragover = ref(false)

// Drag and drop handlers
const handleDragEnter = (event) => {
  isDragover.value = true
}

const handleDragOver = (event) => {
  event.dataTransfer.dropEffect = 'copy'
  isDragover.value = true
}

const handleDragLeave = (event) => {
  // Only set to false if we're leaving the container (not entering a child element)
  if (!event.currentTarget.contains(event.relatedTarget)) {
    isDragover.value = false
  }
}

const handleDrop = async (event) => {
  isDragover.value = false
  
  try {
    // Get the image ID or filename from the dataTransfer
    const imageId = event.dataTransfer.getData('text/plain')
    
    if (!imageId) {
      console.error('No image ID received in drop')
      return
    }
    
    console.log('Image dropped with ID:', imageId)
    
    // Call the store method for similarity search
    await imageStore.searchSimilarById(imageId)
  } catch (error) {
    console.error('Error handling image drop:', error)
  }
}

// Watch for external search reset
watch(() => imageStore.searchQuery, (newVal) => {
  if (!newVal) {
    searchQuery.value = ''
  }
})

const handleSearch = async () => {
  if (!searchQuery.value.trim()) return
  
  console.log("Starting search for:", searchQuery.value);
  
  const results = await imageStore.search({
    query: searchQuery.value,
    type: searchType.value,
    k: parseInt(resultCount.value),
    minSimilarity: parseFloat(minSimilarity.value)
  });
  
  console.log("Search returned:", results.length, "results");
  console.log("Raw results:", results);
  console.log("Displayed images:", imageStore.getValidImages().length);
}

const clearSearch = () => {
  searchQuery.value = ''
  // Don't reset search results until the user explicitly clicks Search or Reset
}

const resetSearch = async () => {
  searchQuery.value = ''
  await imageStore.clearSearch()
}

// Examples of common searches
const searchExamples = [
  "paisley", 
  "floral",
  "red, paisley",
  "paisley, flower",
  "blue, floral", 
  "geometric, green"
]

const runExampleSearch = (example) => {
  searchQuery.value = example
  handleSearch()
}
</script>

<style scoped>
.search-container {
  margin-bottom: var(--space-4);
}

.search-container.is-dragover {
  background-color: rgba(79, 70, 229, 0.1);
  border: 2px dashed var(--color-primary);
  border-radius: var(--radius-lg);
  padding: var(--space-3);
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
  100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
}

.search-form {
  display: flex;
  gap: var(--space-2);
  margin-bottom: var(--space-4);
}

.search-input-wrapper {
  position: relative;
  flex: 1;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: var(--space-3);
  width: 20px;
  height: 20px;
  color: var(--color-text-light);
  pointer-events: none;
}

.search-input {
  flex: 1;
  padding: var(--space-3) var(--space-3) var(--space-3) calc(var(--space-3) + 24px);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: 1rem;
  transition: all var(--transition-fast);
  background-color: var(--color-surface);
}

.search-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
  outline: none;
}

.clear-search-button {
  position: absolute;
  right: var(--space-3);
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  color: var(--color-text-light);
}

.clear-search-button:hover {
  color: var(--color-text);
  background: none;
}

.clear-search-button svg {
  width: 16px;
  height: 16px;
}

.search-button {
  padding: var(--space-3) var(--space-6);
  background-color: var(--color-primary);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-fast);
  white-space: nowrap;
}

.search-button:hover:not(:disabled) {
  background-color: var(--color-primary-dark);
}

.search-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.search-options {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-2);
}

.search-filters {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-4);
}

.filter-group {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.filter-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-light);
}

.filter-buttons {
  display: flex;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--color-border);
}

.filter-button {
  padding: var(--space-2) var(--space-3);
  background-color: var(--color-surface);
  border: none;
  color: var(--color-text-light);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-fast);
  border-right: 1px solid var(--color-border);
}

.filter-button:last-child {
  border-right: none;
}

.filter-button.active {
  background-color: var(--color-primary);
  color: white;
}

.filter-button:hover:not(.active) {
  background-color: var(--color-background);
  transform: none;
}

.result-count-select {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background-color: var(--color-surface);
  font-size: 0.875rem;
  color: var(--color-text);
}

.search-status {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text-light);
  font-size: 0.875rem;
}

.search-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(79, 70, 229, 0.2);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s infinite linear;
}

.search-results-info {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.result-count {
  font-size: 0.875rem;
  color: var(--color-text-light);
}

.reset-button {
  padding: var(--space-1) var(--space-3);
  background-color: transparent;
  border: 1px solid var(--color-border);
  color: var(--color-text);
  font-size: 0.875rem;
  border-radius: var(--radius-md);
}

.reset-button:hover {
  background-color: var(--color-background);
  color: var(--color-primary);
  border-color: var(--color-primary-light);
  transform: none;
}

.search-examples {
  margin-top: var(--space-3);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.examples-label {
  font-size: 0.875rem;
  color: var(--color-text-light);
}

.example-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.example-button {
  font-size: 0.75rem;
  padding: var(--space-1) var(--space-2);
  background-color: var(--color-surface-lighter);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-text);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.example-button:hover {
  background-color: var(--color-primary-lighter);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.min-score-select {
  padding: var(--space-1) var(--space-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background-color: var(--color-surface);
  font-size: 0.875rem;
  color: var(--color-text);
}

.search-hint {
  margin-top: var(--space-2);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text-light);
  font-size: 0.85rem;
  padding: var(--space-2);
  background-color: rgba(79, 70, 229, 0.05);
  border-radius: var(--radius-md);
  border: 1px solid rgba(79, 70, 229, 0.1);
}

.hint-icon {
  font-size: 1rem;
  color: var(--color-primary);
}

.hint-text {
  font-size: 0.875rem;
}

.search-terms {
  font-weight: 600;
  color: var(--color-primary);
  background-color: rgba(79, 70, 229, 0.1);
  padding: 2px 6px;
  border-radius: var(--radius-md);
}

.search-hint.drag-hint {
  margin-top: var(--space-2);
  background-color: rgba(79, 70, 229, 0.05);
  border: 1px dashed rgba(79, 70, 229, 0.3);
}

.search-hint.drag-hint .hint-icon {
  color: var(--color-primary-dark);
}

@media (max-width: 768px) {
  .search-form {
    flex-direction: column;
  }
  
  .search-options {
    flex-direction: column;
    gap: var(--space-3);
  }
  
  .search-filters {
    flex-direction: column;
    gap: var(--space-3);
  }
}
</style> 