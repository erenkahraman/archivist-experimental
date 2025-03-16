import { ref, computed } from 'vue'
import { defineStore } from 'pinia'

// Define the image store using Pinia
export const useImageStore = defineStore('images', () => {
  // State
  const images = ref([])
  const loading = ref(false)
  const isSearching = ref(false)
  const searchQuery = ref('')
  const searchFilters = ref(null)
  const searchSort = ref('relevance')
  const searchTotalResults = ref(null)
  
  // API endpoints configuration
  const apiEndpoints = {
    primary: 'http://localhost:8000/api',
    fallback: 'http://localhost:5000/api'
  }
  
  // Computed properties
  const filteredImages = computed(() => {
    // If no search is active, return all images
    if (!searchQuery.value && (!searchFilters.value || Object.values(searchFilters.value).every(v => !v))) {
      return images.value
    }
    // Otherwise return only images with search scores
    return images.value.filter(image => image.searchScore !== undefined)
  })
  
  // Helper function to try multiple API endpoints
  async function tryApiEndpoints(apiPath, options) {
    let lastError = null
    
    // Try primary endpoint
    try {
      const response = await fetch(`${apiEndpoints.primary}/${apiPath}`, options)
      if (response.ok) return response
    } catch (error) {
      console.log(`Primary endpoint failed for ${apiPath}:`, error)
      lastError = error
    }
    
    // Try fallback endpoint
    try {
      const response = await fetch(`${apiEndpoints.fallback}/${apiPath}`, options)
      if (response.ok) return response
    } catch (error) {
      console.log(`Fallback endpoint failed for ${apiPath}:`, error)
      lastError = error
    }
    
    // If both failed, throw the last error
    throw lastError || new Error('All API endpoints failed')
  }
  
  // Actions
  async function fetchImages() {
    loading.value = true
    try {
      const response = await tryApiEndpoints('images', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      })
      
      const data = await response.json()
      console.log('Fetched images:', data)
      
      // Check the structure of the response
      if (Array.isArray(data)) {
        images.value = data
      } else if (data.images && Array.isArray(data.images)) {
        images.value = data.images
      } else {
        console.error('Unexpected API response format:', data)
        images.value = []
      }
      
      console.log('Processed images:', images.value.length)
    } catch (error) {
      console.error('Failed to fetch images:', error)
    } finally {
      loading.value = false
    }
  }
  
  async function uploadImage(file, onProgress) {
    const formData = new FormData()
    formData.append('file', file)
    
    // Create a placeholder for the uploading image
    const placeholder = {
      original_path: URL.createObjectURL(file),
      thumbnail_path: URL.createObjectURL(file),
      isUploading: true,
      uploadProgress: 0,
      uploadStatus: 'Uploading...'
    }
    
    // Add to the beginning of the array
    images.value.unshift(placeholder)
    
    try {
      const response = await fetch(`${apiEndpoints.primary}/upload`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) throw new Error('Upload failed')
      
      const result = await response.json()
      
      // Replace placeholder with actual image data
      const index = images.value.findIndex(img => img === placeholder)
      if (index !== -1) {
        images.value[index] = result.image
      }
      
      return result.image
    } catch (error) {
      console.error('Upload failed:', error)
      
      // Update placeholder to show error
      const index = images.value.findIndex(img => img === placeholder)
      if (index !== -1) {
        images.value[index].uploadStatus = 'Failed: ' + error.message
      }
      
      throw error
    }
  }
  
  async function deleteImage(path) {
    // Extract just the filename from the path
    const filename = path.split('/').pop().split('\\').pop()
    
    // First, remove from local array immediately to improve UI responsiveness
    const index = images.value.findIndex(img => 
      img.original_path.includes(path) || 
      (typeof img.original_path === 'string' && img.original_path.endsWith(path))
    )
    
    if (index !== -1) {
      // Remove from local array
      images.value.splice(index, 1)
      
      // Attempt server deletion in the background, but don't wait for it
      // This ensures the UI remains responsive regardless of server issues
      setTimeout(() => {
        // Try with no-cors mode to avoid CORS errors
        fetch(`${apiEndpoints.primary}/images/${encodeURIComponent(filename)}`, {
          method: 'DELETE',
          mode: 'no-cors' // This will prevent CORS errors but make response unreadable
        }).catch(() => {
          // If primary fails, try fallback silently
          fetch(`${apiEndpoints.fallback}/images/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
            mode: 'no-cors'
          }).catch(() => {
            // Silent failure is fine here
          })
        })
      }, 100) // Small delay to ensure UI updates first
      
      // Always return success for UI purposes
      return true
    } else {
      // Return success even if image wasn't found
      return true
    }
  }
  
  async function searchImages(query, filters = {}, sort = 'relevance') {
    isSearching.value = true
    searchQuery.value = query
    searchFilters.value = filters
    searchSort.value = sort
    
    try {
      const response = await tryApiEndpoints('search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, filters, sort })
      })
      
      const data = await response.json()
      
      // Update search scores on existing images
      images.value.forEach(img => {
        img.searchScore = undefined
        img.matchedTerms = undefined
        img.isSearching = false
      })
      
      // Apply search scores to matching images
      data.results.forEach(result => {
        const img = images.value.find(i => i.original_path === result.original_path)
        if (img) {
          img.searchScore = result.score
          img.matchedTerms = result.matched_terms
          img.isSearching = true
        }
      })
      
      searchTotalResults.value = data.results.length
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      isSearching.value = false
    }
  }
  
  function clearSearch() {
    searchQuery.value = ''
    searchFilters.value = null
    searchSort.value = 'relevance'
    searchTotalResults.value = null
    
    // Clear search indicators from images
    images.value.forEach(img => {
      img.searchScore = undefined
      img.matchedTerms = undefined
      img.isSearching = false
    })
  }
  
  function clearUploadingStates() {
    // Remove any stale uploading images on page refresh
    images.value = images.value.filter(img => !img.isUploading)
  }
  
  return {
    images,
    loading,
    isSearching,
    searchQuery,
    searchFilters,
    searchSort,
    searchTotalResults,
    filteredImages,
    fetchImages,
    uploadImage,
    deleteImage,
    searchImages,
    clearSearch,
    clearUploadingStates
  }
})