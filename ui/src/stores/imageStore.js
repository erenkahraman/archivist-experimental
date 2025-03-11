import { defineStore } from 'pinia'
import { ref, onMounted } from 'vue'
import { useStorage } from '@vueuse/core'

export const useImageStore = defineStore('images', () => {
  // Persist in local storage using useStorage
  const images = useStorage('gallery-images', [])
  const loading = ref(false)
  const error = ref(null)
  const searchResults = ref([])
  const searchQuery = ref('')
  const searchFilters = ref({})
  const searchSort = ref('relevance')
  const searchTotalResults = ref(0)
  const isSearching = ref(false)
  const API_BASE_URL = 'http://localhost:8000/api'

  // Method to clear uploading states
  const clearUploadingStates = () => {
    images.value = images.value.filter(img => !img.isUploading)
  }

  const fetchImages = async () => {
    try {
      loading.value = true
      error.value = null
      
      // Clear any uploading states
      clearUploadingStates()
      
      // Fetch all images from the API
      const response = await fetch(`${API_BASE_URL}/images`)
      
      if (!response.ok) {
        throw new Error(`Failed to fetch images: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      // Update images with the fetched data
      if (Array.isArray(data)) {
        // Keep any uploading images
        const uploadingImages = images.value.filter(img => img.isUploading)
        images.value = [...uploadingImages, ...data]
      } else {
        console.error('Invalid image data format:', data)
        images.value = []
      }
    } catch (err) {
      console.error('Fetch error:', err)
      error.value = err.message
    } finally {
      loading.value = false
    }
  }

  const deleteImage = async (filepath) => {
    try {
      error.value = null
      
      // Extract just the filename from the path
      const filename = filepath.split('/').pop()
      const response = await fetch(`${API_BASE_URL}/delete/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || 'Delete failed')
      }
      
      // Remove the deleted image from the store
      images.value = images.value.filter(img => img.original_path !== filepath)
      
      return true
    } catch (err) {
      console.error('Delete error:', err)
      error.value = err.message
      throw err
    }
  }

  const searchImages = async (query, filters = {}, sort = 'relevance', limit = 50) => {
    // If query is empty, just clear the search
    if (!query.trim() && Object.keys(filters).length === 0) {
      clearSearch()
      return []
    }

    // Don't search for very short queries unless filters are provided
    if (query.trim().length < 2 && Object.keys(filters).length === 0) {
      return []
    }

    // Set searching state without affecting loading state
    isSearching.value = true
    searchQuery.value = query
    searchFilters.value = filters
    searchSort.value = sort
    error.value = null

    try {
      console.log(`Searching for: "${query}" with filters:`, filters, `sort: ${sort}`)
      
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ 
          query,
          filters,
          sort,
          limit
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Search failed: ${response.statusText}`)
      }

      // Get the search results from the response
      const searchData = await response.json()
      
      // Update the search metadata
      searchTotalResults.value = searchData.total_results || 0
      
      // Keep any uploading images
      const uploadingImages = images.value.filter(img => img.isUploading)
      
      // Process the search results
      if (searchData.results && Array.isArray(searchData.results)) {
        console.log(`Found ${searchData.total_results} results for "${query}"`)
        
        // Add similarity score information to display in UI
        const processedResults = searchData.results.map(result => ({
          ...result,
          searchScore: result.similarity || 0,
          matchedTerms: result.matched_terms || 0
        }))
        
        // Combine uploading images with search results
        images.value = [...uploadingImages, ...processedResults]
        searchResults.value = processedResults
      } else {
        console.error('Invalid search results format:', searchData)
        images.value = uploadingImages
        searchResults.value = []
      }
      
      return searchResults.value
    } catch (err) {
      console.error('Search error:', err)
      error.value = err.message
      
      // Keep uploading images on error
      const uploadingImages = images.value.filter(img => img.isUploading)
      images.value = uploadingImages
      searchResults.value = []
      
      throw err
    } finally {
      isSearching.value = false
    }
  }

  const clearSearch = () => {
    searchResults.value = []
    searchQuery.value = ''
    searchFilters.value = {}
    searchSort.value = 'relevance'
    searchTotalResults.value = 0
    isSearching.value = false
    
    // Fetch all images and reset search-related properties
    fetchImages().then(() => {
      // Clear any search-related properties from the images
      images.value = images.value.map(img => {
        const newImg = { ...img }
        // Remove search-specific properties
        delete newImg.searchScore
        delete newImg.matchedTerms
        delete newImg.similarity
        delete newImg.matched_terms
        return newImg
      })
    })
  }

  const clearAllImages = () => {
    images.value = []
    searchResults.value = []
    searchQuery.value = ''
    searchFilters.value = {}
    searchSort.value = 'relevance'
    searchTotalResults.value = 0
    error.value = null
  }

  // Initialize store
  clearUploadingStates() // Clear uploading states when store is created

  return {
    images,
    loading,
    error,
    searchResults,
    searchQuery,
    searchFilters,
    searchSort,
    searchTotalResults,
    isSearching,
    fetchImages,
    deleteImage,
    searchImages,
    clearSearch,
    clearAllImages,
    clearUploadingStates
  }
}) 