import { defineStore } from 'pinia'
import { ref, onMounted, computed } from 'vue'
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

  // Add state management
  const state = ref({
    loading: false,
    error: null,
    images: [],
    searchResults: [],
    searchQuery: '',
    searchFilters: {},
    searchSort: 'relevance',
    searchTotalResults: 0,
    isSearching: false
  })

  // Add computed properties
  const hasError = computed(() => state.value.error !== null)
  const isLoading = computed(() => state.value.loading)

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

  const searchImages = async (params) => {
    // Extract parameters with defaults
    const { 
      query = '', 
      type = 'all', 
      k = 20, 
      minSimilarity = 0.1 
    } = params || {};
    
    // Clean the query - trim but preserve commas
    const cleanedQuery = query.trim();
    
    // If query is empty, just clear the search
    if (!cleanedQuery) {
      clearSearch()
      return []
    }

    // Don't search for very short queries (but allow if it contains a comma)
    if (cleanedQuery.length < 2 && !cleanedQuery.includes(',')) {
      return []
    }

    // Set searching state
    isSearching.value = true
    searchQuery.value = cleanedQuery
    error.value = null

    try {
      console.log(`Searching for: "${cleanedQuery}" with limit: ${k}, min similarity: ${minSimilarity}`)
      
      // Make sure we have gallery images loaded first (for template structure)
      if (images.value.filter(img => !img.isUploading).length === 0) {
        await fetchImages();
      }
      
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
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Search failed: ${response.statusText}`)
      }

      // Get the search results from the response
      const searchData = await response.json()
      
      // Keep any uploading images
      const uploadingImages = images.value.filter(img => img.isUploading)
      
      // Process the search results
      if (searchData.results && Array.isArray(searchData.results)) {
        console.log(`Found ${searchData.result_count} results for "${cleanedQuery}"`)
        searchTotalResults.value = searchData.result_count || 0
        
        // Normalize search results to match gallery data structure
        const processedResults = searchData.results.map(result => {
          // Get a template image from the gallery if available
          const galleryImages = images.value.filter(img => !img.isUploading);
          const templateImage = galleryImages.length > 0 ? galleryImages[0] : null;
          
          // Process colors to ensure they're always in the expected format
          let processedColors = result.colors;
          
          // If we have a template, use its structure, otherwise just use the result
          if (templateImage) {
            // Make sure thumbnail_path is properly formed
            let thumbnailPath = result.thumbnail_path;
            
            // Ensure the thumbnail_path doesn't include the full API URL
            if (thumbnailPath && thumbnailPath.includes('/api/thumbnails/')) {
              thumbnailPath = thumbnailPath.split('/api/thumbnails/').pop();
            }
            
            return {
              ...JSON.parse(JSON.stringify(templateImage)), // Deep copy all properties from template
              ...result,        // Override with search result data
              thumbnail_path: thumbnailPath, // Use the corrected thumbnail path
              // Keep track of the similarity score for displaying in the search UI
              similarity: result.similarity || 0
            };
          } else {
            // No gallery images available, just use the result as is
            return result;
          }
        });
        
        // Store the results
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

  // Add new function for searching conveniently
  const search = async (params) => {
    return searchImages(params);
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
    search,
    clearSearch,
    clearAllImages,
    clearUploadingStates,
    state,
    hasError,
    isLoading,
    
    // Computed properties for convenience
    searching: computed(() => isSearching.value),
    hasSearchResults: computed(() => searchResults.value.length > 0)
  }
}) 