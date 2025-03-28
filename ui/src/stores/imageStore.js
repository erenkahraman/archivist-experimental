import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Helper to control log level
const isDev = process.env.NODE_ENV === 'development'
const log = (message, ...args) => {
  if (isDev) {
    console.log(message, ...args)
  }
}

export const useImageStore = defineStore('images', () => {
  // Core state
  const images = ref([])
  const loading = ref(false)
  const error = ref(null)
  const searchResults = ref([])
  const searchQuery = ref('')
  const searchTotalResults = ref(0)
  const isSearching = ref(false)
  const API_BASE_URL = 'http://localhost:8000/api'

  // Computed properties
  const hasError = computed(() => error.value !== null)
  const isLoading = computed(() => loading.value)
  const searching = computed(() => isSearching.value)
  const hasSearchResults = computed(() => searchResults.value.length > 0)

  // Basic validation
  const isValidImage = (img) => {
    if (!img) return false
    if (img.isUploading) return true
    return img.thumbnail_path && img.original_path
  }

  // Clean filename helper
  const getFileName = (path) => {
    if (!path) return ''
    return path.split('/').pop()
  }

  // Fetch all images from the server
  const fetchImages = async (limit = 20) => {
    try {
      loading.value = true
      error.value = null
      
      const response = await fetch(`${API_BASE_URL}/images?limit=${limit}`)
      
      if (!response.ok) {
        throw new Error(`Failed to fetch images: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      if (!Array.isArray(data)) {
        throw new Error('Invalid server response')
      }
      
      // Keep only uploading images from the current state
      const uploadingImages = images.value.filter(img => img.isUploading)
      
      // Set images with fresh data from server
      images.value = [...uploadingImages, ...data]
      
      return data
    } catch (err) {
      error.value = err.message
      return []
    } finally {
      loading.value = false
    }
  }

  // Delete an image
  const deleteImage = async (filepath) => {
    try {
      error.value = null
      
      if (!filepath) {
        throw new Error('Invalid filepath')
      }
      
      // Get just the filename for the API call
      const filename = getFileName(filepath)
      
      // Call the delete API
      const response = await fetch(`${API_BASE_URL}/delete/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      })
      
      if (!response.ok) {
        throw new Error('Delete failed')
      }
      
      // Remove the deleted image from the local state
      images.value = images.value.filter(img => {
        const imgFilename = getFileName(img.original_path || img.thumbnail_path || '')
        return imgFilename !== filename
      })
      
      return true
    } catch (err) {
      error.value = err.message
      throw err
    }
  }

  // Search for images
  const searchImages = async (params) => {
    const { query = '', k = 20, minSimilarity = 0.1 } = params || {}
    
    // Clean the query
    const cleanedQuery = query.trim()
    
    // If query is empty, clear the search
    if (!cleanedQuery) {
      clearSearch()
      return []
    }

    // Don't search for very short queries (unless comma-separated)
    if (cleanedQuery.length < 2 && !cleanedQuery.includes(',')) {
      return []
    }

    // Set searching state
    isSearching.value = true
    searchQuery.value = cleanedQuery
    error.value = null

    try {
      // Make sure we have images loaded
      if (images.value.filter(img => !img.isUploading).length === 0) {
        await fetchImages()
      }
      
      // Call the search API
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ 
          query: cleanedQuery,
          limit: k,
          min_similarity: minSimilarity
        }),
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      // Process the search results
      const searchData = await response.json()
      
      if (searchData.results && Array.isArray(searchData.results)) {
        searchTotalResults.value = searchData.result_count || 0
        
        // Keep uploading images
        const uploadingImages = images.value.filter(img => img.isUploading)
        
        // Set images to search results plus uploading images
        images.value = [...uploadingImages, ...searchData.results]
        searchResults.value = searchData.results
        
        return searchResults.value
      } else {
        return []
      }
    } catch (err) {
      error.value = err.message
      return []
    } finally {
      isSearching.value = false
    }
  }

  // Convenience alias for searchImages
  const search = async (params) => {
    return searchImages(params)
  }

  // Clear search and restore all images
  const clearSearch = () => {
    searchResults.value = []
    searchQuery.value = ''
    searchTotalResults.value = 0
    isSearching.value = false
    
    // Fetch all images
    fetchImages()
  }

  // Clear all images from the store
  const clearAllImages = () => {
    images.value = []
    searchResults.value = []
    searchQuery.value = ''
    searchTotalResults.value = 0
    error.value = null
  }

  // Reset store to fresh state
  const resetStore = async () => {
    // Clear all state
    clearAllImages()
    
    // Fetch fresh data
    await fetchImages()
  }

  // Clean current gallery of invalid images
  const cleanGallery = () => {
    // Filter out any images without valid paths
    const validImages = images.value.filter(isValidImage)
    
    // If we removed any, update the state
    if (validImages.length !== images.value.length) {
      images.value = validImages
    }
  }

  // Methods for getting valid images
  const getValidImages = () => {
    return images.value.filter(isValidImage)
  }

  // Nuclear option: completely reset backend storage too
  const nukeEverything = async () => {
    try {
      loading.value = true
      
      // Clear all frontend state first
      images.value = []
      searchResults.value = []
      searchQuery.value = ''
      searchTotalResults.value = 0
      error.value = null
      
      // Call backend to purge everything
      const response = await fetch(`${API_BASE_URL}/purge-all`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      if (!response.ok) {
        throw new Error('Failed to purge images')
      }
      
      await response.json()
      return true
    } catch (err) {
      error.value = err.message
      return false
    } finally {
      loading.value = false
    }
  }

  // Initially load images
  fetchImages()

  return {
    // State
    images,
    loading,
    error,
    searchResults,
    searchQuery,
    searchTotalResults,
    isSearching,
    API_BASE_URL,
    
    // Methods
    fetchImages,
    deleteImage,
    searchImages,
    search,
    clearSearch,
    clearAllImages,
    resetStore,
    cleanGallery,
    getValidImages,
    isValidImage,
    getFileName,
    nukeEverything,
    
    // Computed
    hasError,
    isLoading,
    searching,
    hasSearchResults
  }
}) 