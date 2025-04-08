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
  const searchQueryReferenceImage = ref(null)
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
    return img.thumbnail_path || img.path || img.original_path
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
    // Clear search data and reset to all images
    searchResults.value = []
    searchQuery.value = ''
    searchTotalResults.value = 0
    isSearching.value = false
    searchQueryReferenceImage.value = null
    
    // Fetch all images
    fetchImages()
  }

  // Search for similar images by ID
  const searchSimilarById = async (imageId, options = {}, imageData = null) => {
    try {
      if (!imageId) {
        throw new Error('Image ID is required for similarity search')
      }
      
      // Set searching state
      isSearching.value = true
      error.value = null
      
      // Extract options with defaults
      const k = options.limit || 20
      const minSimilarity = options.minSimilarity || 0.1
      const sortBySimilarity = options.sortBySimilarity === undefined ? true : options.sortBySimilarity
      
      // Log the search attempt
      if (isDev) {
        console.log(`Searching for images similar to: ${imageId} with min_similarity: ${minSimilarity}, limit: ${k}`)
      }
      
      // Call the API endpoint for similarity search with explicit sort parameter
      const apiUrl = `${API_BASE_URL}/similar/${encodeURIComponent(imageId)}?sort=${sortBySimilarity}&min_similarity=${minSimilarity}&limit=${k}`
      
      if (isDev) {
        console.log(`Calling API: ${apiUrl}`)
      }
      
      const response = await fetch(apiUrl, {
        method: 'GET'
      })
      
      if (!response.ok) {
        throw new Error(`Similarity search failed: ${response.statusText}`)
      }
      
      // Process the search results
      const searchData = await response.json()
      
      if (isDev) {
        console.log('API response:', searchData)
      }
      
      // Check if there are results directly in the response or in a results property
      let results = [];
      if (Array.isArray(searchData)) {
        // Results are directly in the response as an array
        results = searchData;
        if (isDev) console.log('Results found in array format', results.length)
      } else if (searchData.results && Array.isArray(searchData.results)) {
        // Results are in a 'results' property
        results = searchData.results;
        if (isDev) console.log('Results found in results property', results.length)
      } else {
        // No valid results structure found
        if (isDev) {
          console.warn('No valid results structure found in API response:', searchData);
        }
        results = [];
      }
      
      // If no results and similarity threshold is > 0.01, try again with a lower threshold
      if (results.length === 0 && minSimilarity > 0.01) {
        if (isDev) {
          console.log(`No results found with similarity ${minSimilarity}, retrying with 0.01`)
        }
        
        // Set a lower threshold and retry
        const lowerMinSimilarity = 0.01
        const retryResponse = await fetch(
          `${API_BASE_URL}/similar/${encodeURIComponent(imageId)}?sort=${sortBySimilarity}&min_similarity=${lowerMinSimilarity}&limit=${k}`,
          { method: 'GET' }
        )
        
        if (retryResponse.ok) {
          const retryData = await retryResponse.json()
          if (isDev) {
            console.log('Retry API response:', retryData)
          }
          
          if (retryData.results && Array.isArray(retryData.results)) {
            results = retryData.results
            if (isDev) console.log(`Found ${results.length} results with lower threshold ${lowerMinSimilarity}`)
          }
        }
      }
      
      // Transform results to ensure better scoring distribution if needed
      results = results.map(result => {
        // If needed, transform similarity scores here for better UI experience
        return {
          ...result,
          // Ensure score is properly formatted
          similarity: parseFloat(result.similarity || 0).toFixed(4)
        }
      });
      
      // Ensure results are sorted by similarity if requested
      if (sortBySimilarity) {
        results.sort((a, b) => parseFloat(b.similarity) - parseFloat(a.similarity))
      }
      
      // Store the results with a friendlier display message
      searchQuery.value = `Similar patterns`;
      searchTotalResults.value = results.length;
      searchResults.value = results;
      
      // Keep uploading images
      const uploadingImages = images.value.filter(img => img.isUploading)
      
      // Find the source image if it exists in current images
      const sourceImage = images.value.find(img => 
        img.id === imageId || 
        getFileName(img.thumbnail_path || '') === imageId || 
        getFileName(img.original_path || '') === imageId
      )
      
      // Set the query reference image for display
      if (sourceImage) {
        // Use the actual image object if we found it
        searchQueryReferenceImage.value = sourceImage
      } else if (imageData && imageData.thumbnailPath) {
        // Create a temporary image object from the drag data
        searchQueryReferenceImage.value = {
          thumbnail_path: imageData.thumbnailPath,
          original_path: imageData.originalPath,
          patterns: {
            primary_pattern: imageData.patternName || 'Reference image'
          }
        }
      }
      
      // Set images to search results plus uploading images
      images.value = [...uploadingImages, ...results]
      
      if (isDev && results.length === 0) {
        console.warn('Similar search returned 0 results for image ID:', imageId);
        console.warn('This could mean:');
        console.warn('1. No images meet the similarity threshold');
        console.warn('2. The reference image embedding is not being found');
        console.warn('3. There might be an issue with the similarity calculation');
      }
      
      return searchResults.value
    } catch (err) {
      error.value = err.message
      if (isDev) {
        console.error('Similarity search error:', err)
      }
      return []
    } finally {
      isSearching.value = false
    }
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
    searchQueryReferenceImage,
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
    searchSimilarById,
    
    // Computed
    hasError,
    isLoading,
    searching,
    hasSearchResults
  }
}) 