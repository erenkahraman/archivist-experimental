import { ref, computed } from 'vue'
import { defineStore } from 'pinia'

// Define the image store using Pinia
export const useImageStore = defineStore('images', () => {
  const images = ref([])
  const loading = ref(false)
  const isSearching = ref(false)
  const searchQuery = ref('')
  const searchFilters = ref(null)
  const searchSort = ref('relevance')
  const searchTotalResults = ref(null)
  
  // Computed properties
  const filteredImages = computed(() => {
    // If no search is active, return all images
    if (!searchQuery.value && (!searchFilters.value || Object.values(searchFilters.value).every(v => !v))) {
      return images.value
    }
    // Otherwise return only images with search scores
    return images.value.filter(image => image.searchScore !== undefined)
  })
  
  // Actions
  async function fetchImages() {
    loading.value = true
    try {
      // Try both endpoints to see which one works
      let response;
      try {
        response = await fetch('http://localhost:8000/api/images')
      } catch (e) {
        // If first endpoint fails, try the alternative
        console.log('Trying alternative endpoint...')
        response = await fetch('http://localhost:5000/api/images')
      }
      
      const data = await response.json()
      console.log('Fetched images:', data)
      
      // Check the structure of the response
      if (Array.isArray(data)) {
        // If the API returns an array directly
        images.value = data
      } else if (data.images && Array.isArray(data.images)) {
        // If the API returns an object with an images property
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
      const response = await fetch('http://localhost:8000/api/upload', {
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
    try {
      await fetch(`http://localhost:8000/api/images/${encodeURIComponent(path)}`, {
        method: 'DELETE'
      })
      
      // Remove from local array
      const index = images.value.findIndex(img => img.original_path === path)
      if (index !== -1) {
        images.value.splice(index, 1)
      }
    } catch (error) {
      console.error('Delete failed:', error)
      throw error
    }
  }
  
  async function searchImages(query, filters = {}, sort = 'relevance') {
    isSearching.value = true
    searchQuery.value = query
    searchFilters.value = filters
    searchSort.value = sort
    
    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          filters,
          sort
        })
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