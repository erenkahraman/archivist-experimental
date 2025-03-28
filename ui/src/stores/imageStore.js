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
        
        // Create a map of valid images by their filenames
        const validImageMap = new Map()
        data.forEach(img => {
          if (img.original_path) {
            const filename = img.original_path.split('/').pop()
            validImageMap.set(filename, true)
          }
          if (img.thumbnail_path) {
            const filename = img.thumbnail_path.split('/').pop()
            validImageMap.set(filename, true)
          }
        })
        
        // Clean up any stale images in local storage
        // This will clear any deleted images that might still be in localStorage
        const storedImages = JSON.parse(localStorage.getItem('gallery-images') || '[]')
        if (Array.isArray(storedImages) && storedImages.length > 0) {
          const cleanedStoredImages = storedImages.filter(img => {
            // Keep uploading images
            if (img.isUploading) return true
            
            const originalFilename = (img.original_path || '').split('/').pop()
            const thumbnailFilename = (img.thumbnail_path || '').split('/').pop()
            
            return validImageMap.has(originalFilename) || validImageMap.has(thumbnailFilename) 
          })
          
          if (cleanedStoredImages.length !== storedImages.length) {
            localStorage.setItem('gallery-images', JSON.stringify(cleanedStoredImages))
            console.log(`Cleaned up ${storedImages.length - cleanedStoredImages.length} stale images from local storage`)
          }
        }
        
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
      
      // Check if filepath is undefined or null
      if (!filepath) {
        throw new Error('Invalid filepath: filepath is undefined or null')
      }
      
      // Extract just the filename from the path
      // This is the key part - we need the actual filename that was uploaded
      let filename = filepath.split('/').pop()
      console.log("Deleting file with filename:", filename)
      
      // Find the image first to ensure we're using the correct original filename
      const imageToDelete = images.value.find(img => {
        const imgOriginalFilename = (img.original_path || '').split('/').pop()
        const imgThumbnailFilename = (img.thumbnail_path || '').split('/').pop()
        const imgPathFilename = (img.path || '').split('/').pop()
        
        return filename === imgOriginalFilename || 
               filename === imgThumbnailFilename ||
               filename === imgPathFilename
      })
      
      // If we found the image in our list, use the original filename for deletion
      if (imageToDelete) {
        const originalFilename = (imageToDelete.original_path || imageToDelete.path || filepath).split('/').pop()
        filename = originalFilename
        console.log("Using original filename for deletion:", filename)
      }
      
      const response = await fetch(`${API_BASE_URL}/delete/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      })
      
      if (!response.ok) {
        throw new Error('Delete failed')
      }
      
      // Remove the deleted image from the store
      images.value = images.value.filter(img => {
        // Check against both original file and thumbnail paths
        const imgOriginalFilename = (img.original_path || '').split('/').pop()
        const imgThumbnailFilename = (img.thumbnail_path || '').split('/').pop()
        const imgPathFilename = (img.path || '').split('/').pop()
        
        // Keep if none of the filenames match
        return filename !== imgOriginalFilename && 
               filename !== imgThumbnailFilename &&
               filename !== imgPathFilename
      })
      
      // After successfully deleting on the server, clear local storage cache
      // and re-fetch clean data from the server
      localStorage.removeItem('gallery-images')
      
      // Fetch fresh data from the server to ensure the UI is in sync
      await fetchImages()
      
      return true
    } catch (err) {
      console.error('Delete error:', err)
      error.value = err.message
      throw err
    }
  }

  // Clear cached data and reload from server
  const clearCache = async () => {
    localStorage.removeItem('gallery-images')
    await fetchImages()
  }

  // Check that our local storage is in sync with server data
  const validateLocalCache = async () => {
    try {
      // This will check for any local images that might be deleted on the server
      // or any corrupted image records
      const response = await fetch(`${API_BASE_URL}/images`)
      
      if (!response.ok) {
        console.error("Failed to validate cache - server error")
        return
      }
      
      const serverImages = await response.json()
      
      if (!Array.isArray(serverImages)) {
        console.error("Invalid server response for images validation")
        return
      }
      
      // Map by image identifiers (filename)
      const serverImageMap = new Map()
      serverImages.forEach(img => {
        if (img.original_path) {
          const filename = img.original_path.split('/').pop()
          serverImageMap.set(filename, true)
        }
        if (img.thumbnail_path) {
          const filename = img.thumbnail_path.split('/').pop()
          serverImageMap.set(filename, true)
        }
      })
      
      // Filter local images that don't exist on server
      const filteredImages = images.value.filter(img => {
        // Skip uploading images
        if (img.isUploading) return true
        
        const originalFilename = (img.original_path || '').split('/').pop()
        const thumbnailFilename = (img.thumbnail_path || '').split('/').pop()
        
        return serverImageMap.has(originalFilename) || serverImageMap.has(thumbnailFilename)
      })
      
      // If we filtered out some images, update the local store
      if (filteredImages.length !== images.value.length) {
        console.log(`Removed ${images.value.length - filteredImages.length} stale images from local cache`)
        images.value = filteredImages
      }
    } catch (err) {
      console.error("Failed to validate image cache:", err)
    }
  }

  // Ensure our local cache matches server data on startup
  onMounted(() => {
    validateLocalCache()
  })

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

  // Add a function to purge stale/deleted images that triggers on startup
  const purgeDeletedImages = async () => {
    try {
      console.log("Purging deleted images from cache...")
      
      // IMMEDIATELY clear localStorage to prevent stale data
      localStorage.removeItem('gallery-images')
      
      // Fetch valid images from server
      const response = await fetch(`${API_BASE_URL}/images`)
      
      if (!response.ok) {
        console.error("Failed to fetch images for purging")
        images.value = [] // Force empty state if server error
        return
      }
      
      const serverImages = await response.json()
      
      if (!Array.isArray(serverImages)) {
        console.error("Invalid server response format for purging")
        images.value = [] // Force empty state if invalid data
        return
      }
      
      // Filter out images that don't have both original_path and thumbnail_path
      const validImages = serverImages.filter(img => 
        img && 
        img.original_path && 
        img.thumbnail_path && 
        typeof img.original_path === 'string' &&
        typeof img.thumbnail_path === 'string'
      );
      
      // Update images with valid images
      images.value = validImages
      
      // Update localStorage with clean data
      localStorage.setItem('gallery-images', JSON.stringify(validImages))
      
      console.log(`Gallery updated with ${validImages.length} valid images`)
      
    } catch (err) {
      console.error("Error purging deleted images:", err)
      // Clear images on error to prevent stale data
      images.value = []
      localStorage.removeItem('gallery-images')
    }
  }

  // Run purge on startup
  purgeDeletedImages()

  // Completely reset the store to a fresh state
  const resetStore = async () => {
    // Clear local storage
    localStorage.removeItem('gallery-images')
    
    // Clear reactive state
    images.value = []
    searchResults.value = []
    searchQuery.value = ''
    searchFilters.value = {}
    searchSort.value = 'relevance'
    searchTotalResults.value = 0
    error.value = null
    
    // Force fetch fresh data from server
    await fetchImages()
    
    console.log("Image store fully reset and reloaded")
  }

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
    purgeDeletedImages,
    resetStore,
    
    // Computed properties for convenience
    searching: computed(() => isSearching.value),
    hasSearchResults: computed(() => searchResults.value.length > 0),
    clearCache,
    validateLocalCache
  }
}) 