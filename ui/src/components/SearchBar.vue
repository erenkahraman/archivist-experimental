<template>
  <div 
    class="sidebar-search-container" 
    :class="{ 'is-dragover': isDragover }"
    @dragenter.prevent="handleDragEnter"
    @dragover.prevent="handleDragOver"
    @dragleave.prevent="handleDragLeave"
    @drop.prevent="handleDrop"
  >
    <!-- Logo at the top of sidebar -->
    <div class="sidebar-logo">
      <img :src="aiLogo" alt="AI Tools Logo">
    </div>
    
    <div class="ai-search-header">
      <div class="ai-badge">
        <span class="ai-pulse"></span>
        AI-Powered
      </div>
      <h2 class="sidebar-title">Smart Search</h2>
    </div>
    
    <div class="search-form">
      <div class="search-input-wrapper">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <input 
          v-model="searchQuery" 
          type="text" 
          class="search-input"
          placeholder="Describe images or patterns..."
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
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Search</span>
      </button>
    </div>

    <div class="search-settings-section">
      <h3 class="settings-title">Settings</h3>
      <div class="settings-grid">
        <div class="setting-item">
          <label class="setting-label">Results</label>
          <select v-model="resultCount" class="setting-select">
            <option value="10">10</option>
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </div>
        
        <div class="setting-item">
          <label class="setting-label">Min Score</label>
          <select v-model="minSimilarity" class="setting-select">
            <option value="0.01">Very low (0.01)</option>
            <option value="0.1">Low (0.1)</option>
            <option value="0.3">Medium (0.3)</option>
            <option value="0.5">High (0.5)</option>
            <option value="0.7">Very high (0.7)</option>
          </select>
        </div>
      </div>
    </div>
    
    <div class="search-status-section" v-if="imageStore.searching || imageStore.hasSearchResults">
      <div class="search-status-container" v-if="imageStore.searching">
        <div class="search-spinner"></div>
        <span>AI processing your search...</span>
      </div>
      
      <div class="search-results-info" v-if="imageStore.hasSearchResults">
        <div class="result-count">
          <strong>{{ imageStore.searchResults.length }}</strong> results for 
          <span class="search-terms">{{ imageStore.searchQuery }}</span>
        </div>
        <button class="reset-button" @click="resetSearch">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 4V9H4.58152M19.9381 11C19.446 7.05369 16.0796 4 12 4C8.64262 4 5.76829 6.06817 4.58152 9M4.58152 9H9M20 20V15H19.4185M19.4185 15C18.2317 17.9318 15.3574 20 12 20C7.92038 20 4.55399 16.9463 4.06189 13M19.4185 15H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <span>Reset</span>
        </button>
      </div>
    </div>
    
    <!-- Quick search examples -->
    <div class="search-examples-section">
      <h3 class="examples-title">Try Natural Language Queries</h3>
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
      <span class="hint-text">Use descriptive terms like "vibrant floral with blue background"</span>
    </div>
    
    <!-- Add a drag-and-drop hint -->
    <div class="search-hint drag-hint">
      <span class="hint-icon">🔍</span>
      <span class="hint-text">Drag any image here to find visually similar patterns</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useImageStore } from '../stores/imageStore'
import aiLogo from '../assets/aitlogo.png'

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
  "vibrant paisley pattern", 
  "blue floral designs",
  "traditional red patterns",
  "geometric patterns with green",
  "modern abstract designs", 
  "natural leaf motifs"
]

const runExampleSearch = (example) => {
  searchQuery.value = example
  handleSearch()
}
</script>

<style scoped>
.sidebar-search-container {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  padding: var(--space-5);
  padding-top: 0; /* No top padding to allow logo to use all space */
  padding-bottom: var(--space-8); /* Extra padding at bottom */
  transition: all 0.3s ease;
  height: 100%;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(99, 102, 241, 0.3) transparent;
}

/* For WebKit browsers */
.sidebar-search-container::-webkit-scrollbar {
  width: 4px;
}

.sidebar-search-container::-webkit-scrollbar-track {
  background: transparent;
}

.sidebar-search-container::-webkit-scrollbar-thumb {
  background-color: rgba(99, 102, 241, 0.3);
  border-radius: var(--radius-full);
}

.sidebar-search-container::-webkit-scrollbar-thumb:hover {
  background-color: rgba(99, 102, 241, 0.5);
}

.sidebar-search-container.is-dragover {
  background-color: rgba(79, 70, 229, 0.1);
  border: 2px dashed var(--color-primary);
  border-radius: var(--radius-lg);
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
  100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
}

/* AI Search Header */
.ai-search-header {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-1);
  margin-bottom: var(--space-2);
}

.ai-badge {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-3);
  background: linear-gradient(90deg, rgba(99, 102, 241, 0.1), rgba(34, 211, 238, 0.1));
  border-radius: var(--radius-full);
  color: var(--color-accent);
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  border: 1px solid rgba(34, 211, 238, 0.3);
  box-shadow: 0 0 10px rgba(34, 211, 238, 0.1);
}

.ai-pulse {
  width: 8px;
  height: 8px;
  background-color: var(--color-accent);
  border-radius: 50%;
  position: relative;
  display: inline-block;
}

.ai-pulse::before {
  content: '';
  position: absolute;
  inset: -2px;
  background-color: rgba(34, 211, 238, 0.5);
  border-radius: 50%;
  animation: pulse-animation 2s infinite;
}

@keyframes pulse-animation {
  0% { transform: scale(1); opacity: 1; }
  70% { transform: scale(1.5); opacity: 0; }
  100% { transform: scale(1.5); opacity: 0; }
}

.sidebar-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  background: linear-gradient(135deg, #6366f1, #22d3ee);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
}

.search-form {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  margin-bottom: var(--space-2);
}

.search-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: var(--space-3);
  width: 18px;
  height: 18px;
  color: var(--color-primary-light);
  pointer-events: none;
}

.search-input {
  width: 100%;
  padding: var(--space-3) var(--space-3) var(--space-3) calc(var(--space-3) + 24px);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: var(--radius-md);
  font-size: 0.95rem;
  background-color: rgba(255, 255, 255, 0.03);
  color: var(--color-text);
  transition: all 0.3s ease;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.search-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15), 0 0 10px rgba(99, 102, 241, 0.1);
  outline: none;
  background-color: rgba(255, 255, 255, 0.05);
}

.clear-search-button {
  position: absolute;
  right: var(--space-3);
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  color: var(--color-text-light);
  opacity: 0.6;
  transition: all 0.2s ease;
}

.clear-search-button:hover {
  color: var(--color-text);
  opacity: 1;
}

.clear-search-button svg {
  width: 16px;
  height: 16px;
}

.search-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-4);
  background: linear-gradient(135deg, #6366f1, #22d3ee);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.95rem;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
}

.search-button svg {
  width: 18px;
  height: 18px;
}

.search-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #4f46e5, #0ea5e9);
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
}

.search-button:active:not(:disabled) {
  transform: translateY(1px);
  box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
}

.search-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Settings section */
.search-settings-section {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: var(--space-4);
}

.settings-title {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 var(--space-3) 0;
  color: var(--color-text-light);
}

.settings-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.setting-label {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--color-text-light);
  margin-left: var(--space-2);
}

.setting-select {
  padding: var(--space-2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  background-color: rgba(255, 255, 255, 0.05);
  color: var(--color-text);
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.setting-select:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
  outline: none;
}

/* Search status section */
.search-status-section {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: var(--space-4);
}

.search-status-container {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text-light);
  font-size: 0.9rem;
}

.search-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(34, 211, 238, 0.2);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 1s infinite linear;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.search-results-info {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.result-count {
  font-size: 0.9rem;
  color: var(--color-text-light);
}

.search-terms {
  font-style: italic;
  color: var(--color-primary-light);
}

.reset-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background-color: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--color-text);
  font-size: 0.85rem;
  border-radius: var(--radius-md);
  transition: all 0.2s ease;
  cursor: pointer;
}

.reset-button svg {
  width: 16px;
  height: 16px;
}

.reset-button:hover {
  background-color: rgba(255, 255, 255, 0.1);
  color: var(--color-primary-light);
  border-color: var(--color-primary-light);
}

/* Search examples section */
.search-examples-section {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: var(--space-4);
}

.examples-title {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 var(--space-3) 0;
  color: var(--color-text-light);
}

.example-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.example-button {
  font-size: 0.8rem;
  padding: var(--space-2) var(--space-3);
  background-color: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.2);
  color: var(--color-primary-light);
  border-radius: var(--radius-full);
  cursor: pointer;
  transition: all 0.2s ease;
}

.example-button:hover {
  background-color: rgba(99, 102, 241, 0.2);
  border-color: var(--color-primary);
  color: var(--color-primary-light);
  box-shadow: 0 0 10px rgba(99, 102, 241, 0.15);
}

/* Hints */
.search-hint {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: 0.8rem;
  color: var(--color-text-light);
  padding: var(--space-3);
  background-color: rgba(255, 255, 255, 0.03);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--color-primary);
  margin-top: var(--space-2);
}

.drag-hint {
  border-left-color: var(--color-accent);
}

.hint-icon {
  font-size: 1rem;
}

.hint-text {
  line-height: 1.4;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .sidebar-search-container {
    padding-top: 0;
    height: calc(100% - 60px);
  }

  .sidebar-logo {
    margin: 0 0 1.5rem 0;
    padding: 1rem;
  }
  
  .sidebar-logo img {
    width: 160px;
  }

  .settings-grid {
    grid-template-columns: 1fr;
  }
  
  .search-form {
    flex-direction: column;
  }
  
  .search-button {
    margin-top: var(--space-2);
  }

  .ai-search-header {
    margin-top: 0;
  }
}

/* Logo container */
.sidebar-logo {
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0 0 2rem 0;
  padding: 1.5rem 1rem;
  background-color: transparent;
  box-shadow: none;
}

.sidebar-logo img {
  width: 200px;
  height: auto;
  object-fit: contain;
  filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.5));
}
</style> 