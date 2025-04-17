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
          placeholder="Search for patterns or designs..."
          @keyup.enter="handleSearch"
          title="Enter a search term to find matching images"
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
        title="Search the image collection"
      >
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Search</span>
      </button>
    </div>

    <div class="search-status-section" v-if="imageStore.searching || imageStore.hasSearchResults">
      <div class="search-status-container" v-if="imageStore.searching">
        <div class="search-spinner"></div>
        <span>AI processing your search...</span>
      </div>
      
      <div class="search-results-info" v-if="imageStore.hasSearchResults">
        <!-- Search results summary and reset button -->
        <div class="search-header">
          <div class="results-summary">
            <div class="result-count-badge">
              <strong>{{ imageStore.searchResults.length }}</strong>
            </div>
            <div class="results-description">
              results for 
              <span 
                class="search-terms" 
                :class="{ 'image-search': isImageSearch }"
              >
                <svg v-if="isImageSearch" class="image-search-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <circle cx="14" cy="9" r="1" fill="currentColor"/>
                  <path d="M5 3h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2z" stroke="currentColor" stroke-width="2"/>
                </svg>
                {{ imageStore.searchQuery }}
              </span>
            </div>
          </div>
          
          <button class="reset-search-button" @click="resetSearch">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 4V9H4.58152M19.9381 11C19.446 7.05369 16.0796 4 12 4C8.64262 4 5.76829 6.06817 4.58152 9M4.58152 9H9M20 20V15H19.4185M19.4185 15C18.2317 17.9318 15.3574 20 12 20C7.92038 20 4.55399 16.9463 4.06189 13M19.4185 15H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Reset Search</span>
          </button>
        </div>
      </div>
    </div>
    
    <!-- Search hint -->
    <div class="search-hint">
      <span class="hint-icon">💡</span>
      <span class="hint-text">Enter exact phrases like "hibiscus flower" to find matching images</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useImageStore } from '../stores/imageStore'
import aiLogo from '../assets/aitlogo.png'

const imageStore = useImageStore()
const searchQuery = ref('')
const isDragover = ref(false)

// Add a computed property to detect if this is an image search
const isImageSearch = computed(() => {
  return imageStore.searchQuery && 
    (imageStore.searchQuery.startsWith('Similar to:') || 
     imageStore.searchQuery.startsWith('Similar to image:'))
})

// Max results to fetch - set to a very high number to get all images
const MAX_RESULTS = 1000

// Drag and drop handlers (only for dropped images from within the app, not uploads)
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

// Handle internal app drag and drop - for searching, not uploading
const handleDrop = (event) => {
  event.preventDefault()
  isDragover.value = false
  console.log('Image dropped for similarity search')

  const items = event.dataTransfer.items
  if (!items || items.length === 0) {
    console.warn('No items in drop event')
    return
  }

  // Get the imageId from the dragged image
  const imageId = event.dataTransfer.getData('text/plain')
  console.log('Image ID:', imageId)

  // Try to extract image data if available
  let imageData = null
  try {
    const jsonData = event.dataTransfer.getData('application/json')
    if (jsonData) {
      imageData = JSON.parse(jsonData)
      console.log('Image data:', imageData)
    }
  } catch (error) {
    console.error('Error parsing image data:', error)
  }

  // Clear existing text search if any
  searchQuery.value = ''
  
  // Perform image similarity search
  imageStore.searchSimilarById(imageId, MAX_RESULTS)
    .then(() => {
      // Update search query for display purposes
      if (imageData && imageData.pattern) {
        imageStore.searchQuery = `Similar to: ${imageData.pattern}`
      } else {
        imageStore.searchQuery = `Similar to image: ${imageId.substring(0, 8)}...`
      }
      // Setting searchStatus is handled by imageStore
    })
    .catch(error => {
      console.error('Error in similarity search:', error)
    })
}

// Watch for external search reset
watch(() => imageStore.searchQuery, (newVal) => {
  if (!newVal) {
    searchQuery.value = ''
  }
})

// Handle text search
const handleSearch = async () => {
  if (!searchQuery.value.trim()) return
  
  console.log("Starting search for:", searchQuery.value)
  
  // Use our text search function
  await imageStore.searchByText(searchQuery.value, MAX_RESULTS)
  
  console.log("Text search completed")
  console.log("Displayed images:", imageStore.searchResults.length)
}

// Clear the search
const clearSearch = () => {
  searchQuery.value = ''
  imageStore.clearSearch()
}

// Reset the search
const resetSearch = () => {
  searchQuery.value = ''
  imageStore.clearSearch()
  imageStore.fetchImages(MAX_RESULTS)
}

// Get thumbnail URL helper
const getThumbnailUrl = (path) => {
  if (!path) return ''
  return path
}
</script>

<style scoped>
.sidebar-search-container {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: var(--space-3);
  padding-top: 0; /* No top padding to allow logo to use all space */
  padding-bottom: var(--space-3); /* Reduced bottom padding */
  transition: all 0.3s ease;
  height: 100%;
  overflow-y: hidden; /* Prevent scrolling */
  min-height: 0;
}

/* For WebKit browsers */
.sidebar-search-container::-webkit-scrollbar {
  width: 4px;
}

.sidebar-search-container::-webkit-scrollbar-track {
  background: transparent;
}

.sidebar-search-container::-webkit-scrollbar-thumb {
  background-color: rgba(58, 58, 58, 0.5);
  border-radius: var(--radius-full);
}

.sidebar-search-container::-webkit-scrollbar-thumb:hover {
  background-color: rgba(58, 58, 58, 0.7);
}

.sidebar-search-container.is-dragover {
  background-color: rgba(40, 40, 40, 0.2);
  border: 2px dashed #3a3a3a;
  border-radius: var(--radius-lg);
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(58, 58, 58, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(58, 58, 58, 0); }
  100% { box-shadow: 0 0 0 0 rgba(58, 58, 58, 0); }
}

/* AI Search Header */
.ai-search-header {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-1);
  margin-bottom: var(--space-1); /* Reduced margin */
}

.ai-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px; /* Reduced gap */
  background: rgba(37, 37, 37, 0.5);
  border: 1px solid var(--color-primary-dark);
  color: var(--color-secondary);
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 6px; /* Reduced padding */
  border-radius: var(--radius-full);
  margin-bottom: 0; /* Removed margin */
}

.ai-pulse {
  width: 6px;
  height: 6px;
  background-color: var(--color-primary-light);
  border-radius: 50%;
  animation: pulse 2s infinite;
  flex-shrink: 0;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(119, 119, 119, 0.7);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(119, 119, 119, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(119, 119, 119, 0);
  }
}

.sidebar-title {
  font-size: 1.25rem; /* Smaller font */
  font-weight: 700;
  margin: 0 0 var(--space-2) 0; /* Reduced margin */
  color: var(--color-secondary);
}

.search-form {
  display: flex;
  flex-direction: column;
  gap: var(--space-2); /* Reduced gap */
  margin-bottom: var(--space-1); /* Reduced margin */
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
  padding: var(--space-2) var(--space-2) var(--space-2) calc(var(--space-2) + 24px); /* Reduced padding */
  border: 1px solid rgba(58, 58, 58, 0.3);
  border-radius: var(--radius-md);
  font-size: 0.9rem; /* Smaller font */
  background-color: rgba(255, 255, 255, 0.03);
  color: var(--color-text);
  transition: all 0.3s ease;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.search-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(58, 58, 58, 0.15), 0 0 10px rgba(58, 58, 58, 0.1);
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
  background: var(--color-primary);
  color: var(--color-text);
  border: 1px solid var(--color-primary-dark);
  border-radius: var(--radius-md);
  padding: var(--space-1) var(--space-2); /* Reduced padding */
  display: flex;
  align-items: center;
  gap: var(--space-1); /* Reduced gap */
  font-weight: 600;
  font-size: 0.9rem; /* Smaller font */
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.search-button:hover {
  background: var(--color-primary-dark);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  transform: translateY(-1px);
}

.search-button:active:not(:disabled) {
  transform: translateY(1px);
  box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
}

.search-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.search-button svg {
  width: 16px;
  height: 16px;
  color: var(--color-text);
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
  border: 2px solid rgba(58, 58, 58, 0.2);
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
  align-items: stretch;
  gap: var(--space-4);
  width: 100%;
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-lg);
  border: 1px solid rgba(255, 255, 255, 0.06);
  padding: var(--space-4);
  margin: var(--space-3) 0;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

.search-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: var(--space-3);
  flex-wrap: wrap;
  gap: var(--space-3);
}

.results-summary {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.result-count-badge {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--color-text);
  min-width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 var(--space-2);
  background: var(--color-primary);
  border-radius: var(--radius-full);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.results-description {
  font-size: 0.9rem;
  color: var(--color-text-light);
  padding: var(--space-1) var(--space-2);
}

.search-terms {
  color: var(--color-secondary);
  font-weight: 600;
}

.search-terms.image-search {
  display: inline-flex;
  align-items: center;
  background-color: rgba(79, 70, 229, 0.1);
  color: #4F46E5;
  border-radius: var(--radius-md);
  padding: 0.25rem 0.5rem;
}

.image-search-icon {
  width: 14px;
  height: 14px;
  margin-right: 4px;
}

.reset-search-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: #505050;
  border: 1px solid #3a3a3a;
  color: white;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: var(--radius-md);
  transition: all 0.2s ease;
  cursor: pointer;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.reset-search-button svg {
  width: 18px;
  height: 18px;
}

.reset-search-button:hover {
  background: #3a3a3a;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

/* Search hint */
.search-hint {
  display: flex;
  align-items: center;
  gap: var(--space-1); /* Reduced gap */
  font-size: 0.7rem; /* Smaller font */
  color: var(--color-text-light);
  padding: var(--space-1) var(--space-2); /* Reduced padding */
  background-color: rgba(58, 58, 58, 0.1);
  border-radius: var(--radius-md);
  border-left: 2px solid var(--color-primary); /* Thinner border */
  margin-top: var(--space-1); /* Reduced margin */
}

.hint-icon {
  font-size: 0.8rem; /* Smaller icon */
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
  margin: 0 0 1rem 0; /* Reduced margin */
  padding: 1rem 0.5rem; /* Reduced padding */
  background-color: transparent;
  box-shadow: none;
}

.sidebar-logo img {
  width: 160px; /* Reduced logo size */
  height: auto;
  object-fit: contain;
  filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.5));
}
</style> 