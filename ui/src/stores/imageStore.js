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
  const API_BASE_URL = 'http://localhost:8080/api'

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

  // Fetch all images from the server with cache busting
  const fetchImages = async (limit = 20) => {
    try {
      loading.value = true
      error.value = null
      
      // Add cache busting parameter
      const cacheBuster = Date.now()
      const response = await fetch(`${API_BASE_URL}/images?limit=${limit}&_=${cacheBuster}`)
      
      if (!response.ok) {
        throw new Error(`Failed to fetch images: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      if (!Array.isArray(data)) {
        throw new Error('Invalid server response')
      }
      
      // Keep only uploading images from the current state
      const uploadingImages = images.value.filter(img => img.isUploading)
      
      // Filter out any obviously invalid images
      const validImages = data.filter(img => img && (img.thumbnail_path || img.path || img.original_path))
      
      // Set images with fresh data from server
      images.value = [...uploadingImages, ...validImages]
      
      return validImages
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
      
      // Remove the deleted image from all states
      const removeFromStates = (filename) => {
        // Remove from main images array
        images.value = images.value.filter(img => {
          const imgFilename = getFileName(img.original_path || img.thumbnail_path || '')
          return imgFilename !== filename
        })
        
        // Also remove from search results if present
        if (searchResults.value.length > 0) {
          searchResults.value = searchResults.value.filter(img => {
            const imgFilename = getFileName(img.original_path || img.thumbnail_path || '')
            return imgFilename !== filename
          })
        }
      }
      
      // Remove from our state
      removeFromStates(filename)
      
      // Clear browser cache for this image
      await clearImageCache(filename)
      
      return true
    } catch (err) {
      error.value = err.message
      throw err
    }
  }

  // Clear browser cache for specific image or all images
  const clearImageCache = async (filename) => {
    if (!('caches' in window)) return
    
    try {
      // If we have a specific filename, notify the backend to clean it up too
      if (filename) {
        try {
          // Try to tell backend this file is missing (best effort, don't wait)
          fetch(`${API_BASE_URL}/mark-missing/${encodeURIComponent(filename)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          }).catch(() => {}) // Ignore if endpoint doesn't exist
          
          // Also try direct delete as fallback
          fetch(`${API_BASE_URL}/delete/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
          }).catch(() => {}) // Ignore errors
        } catch (err) {
          // Ignore backend errors, focus on clearing browser cache
        }
      }
      
      const cacheNames = await window.caches.keys()
      
      for (const cacheName of cacheNames) {
        const cache = await window.caches.open(cacheName)
        
        if (filename) {
          // Delete specific image URLs
          const urlsToDelete = [
            `${API_BASE_URL}/thumbnails/${filename}`,
            `${API_BASE_URL}/images/${filename}`
          ]
          
          for (const url of urlsToDelete) {
            await cache.delete(url)
          }
        } else {
          // Delete all image cache entries
          const requests = await cache.keys()
          for (const request of requests) {
            if (request.url.includes('/thumbnails/') || request.url.includes('/images/')) {
              await cache.delete(request)
            }
          }
        }
      }
    } catch (err) {
      console.warn('Error clearing image cache:', err)
    }
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
      const minSimilarity = 0.0001  // Always use the lowest threshold to get results
      const sortBySimilarity = options.sortBySimilarity === undefined ? true : options.sortBySimilarity
      
      // Find the source image if it exists in current images
      const sourceImage = images.value.find(img => 
        img.id === imageId || 
        getFileName(img.thumbnail_path || '') === imageId || 
        getFileName(img.original_path || '') === imageId
      )
      
      // Use cache busting to ensure fresh results
      const cacheBuster = Date.now()
      
      // First check if API exists
      try {
        const pingResponse = await fetch(`${API_BASE_URL}/ping`, { method: 'GET' })
        if (!pingResponse.ok) {
          throw new Error('API server is not responding')
        }
      } catch (err) {
        console.error('API server error:', err)
        throw new Error('Cannot connect to API server')
      }
      
      // Then try to purge known nonexistent files from backend before search
      await fetch(`${API_BASE_URL}/purge-missing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }).catch(() => {}) // Ignore errors if endpoint doesn't exist
      
      // Call similarity search API
      const apiUrl = `${API_BASE_URL}/similar/${encodeURIComponent(imageId)}?sort=${sortBySimilarity}&min_similarity=${minSimilarity}&limit=${k}&_=${cacheBuster}`
      
      if (isDev) {
        console.log(`Calling API: ${apiUrl}`)
      }
      
      const response = await fetch(apiUrl)
      
      if (!response.ok) {
        throw new Error(`Similarity search failed: ${response.statusText}`)
      }
      
      // Process the search results
      const searchData = await response.json()
      
      // Extract results array
      let results = []
      if (Array.isArray(searchData)) {
        results = searchData
      } else if (searchData.results && Array.isArray(searchData.results)) {
        results = searchData.results
      }
      
      // Skip validation if no results
      if (results.length === 0) {
        // Store empty results
        searchQuery.value = `Similar patterns`
        searchTotalResults.value = 0
        searchResults.value = []
        
        // Keep only uploading images
        const uploadingImages = images.value.filter(img => img.isUploading)
        images.value = [...uploadingImages]
        
        // Set reference image if available
        if (sourceImage) {
          searchQueryReferenceImage.value = sourceImage
        } else if (imageData && imageData.thumbnailPath) {
          searchQueryReferenceImage.value = {
            thumbnail_path: imageData.thumbnailPath,
            original_path: imageData.originalPath,
            patterns: {
              primary_pattern: imageData.patternName || 'Reference image'
            }
          }
        }
        
        return []
      }
      
      // Batch validation of thumbnails for better performance
      const validResults = []
      const imagesToDelete = []
      
      // Prepare for batch HEAD requests
      const checkPromises = results.map(async result => {
        // Skip results with no paths
        if (!result || (!result.thumbnail_path && !result.path && !result.original_path)) {
          return null
        }
        
        const filename = getFileName(result.thumbnail_path || result.path || result.original_path)
        
        try {
          const checkUrl = `${API_BASE_URL}/thumbnails/${filename}`
          const checkResponse = await fetch(checkUrl, { 
            method: 'HEAD',
            // Add cache busting to avoid browser cache
            headers: { 'Cache-Control': 'no-cache, no-store' }
          })
          
          if (checkResponse.ok) {
            return {
              ...result,
              similarity: parseFloat(result.similarity || 0).toFixed(4)
            }
          } else {
            // Add to delete list
            imagesToDelete.push(filename)
            return null
          }
        } catch (err) {
          // Add to delete list on error
          imagesToDelete.push(filename)
          return null
        }
      })
      
      // Wait for all checks to complete
      const checkedResults = await Promise.all(checkPromises)
      
      // Filter out null values
      const filteredResults = checkedResults.filter(Boolean)
      
      if (isDev) {
        const filteredCount = results.length - filteredResults.length
        if (filteredCount > 0) {
          console.log(`Filtered out ${filteredCount} missing images from search results`)
        }
      }
      
      // Delete all missing images in batch (in background)
      if (imagesToDelete.length > 0) {
        // We don't await this - let it run in background
        Promise.all(imagesToDelete.map(filename => 
          fetch(`${API_BASE_URL}/delete/${encodeURIComponent(filename)}`, { 
            method: 'DELETE' 
          }).catch(() => {})
        )).then(() => {
          if (isDev) console.log(`Deleted ${imagesToDelete.length} missing images from backend`)
        })
      }
      
      // Store the results
      searchQuery.value = `Similar patterns`
      searchTotalResults.value = filteredResults.length
      searchResults.value = filteredResults
      
      // Keep uploading images and add search results
      const uploadingImages = images.value.filter(img => img.isUploading)
      images.value = [...uploadingImages, ...filteredResults]
      
      // Set the query reference image for display
      if (sourceImage) {
        searchQueryReferenceImage.value = sourceImage
      } else if (imageData && imageData.thumbnailPath) {
        searchQueryReferenceImage.value = {
          thumbnail_path: imageData.thumbnailPath,
          original_path: imageData.originalPath,
          patterns: {
            primary_pattern: imageData.patternName || 'Reference image'
          }
        }
      }
      
      return filteredResults
    } catch (err) {
      error.value = err.message
      return []
    } finally {
      isSearching.value = false
    }
  }

  // Clear search and restore all images
  const clearSearch = () => {
    searchResults.value = []
    searchQuery.value = ''
    searchTotalResults.value = 0
    isSearching.value = false
    searchQueryReferenceImage.value = null
    
    // Fetch fresh images
    fetchImages()
  }

  // Reset everything
  const nukeEverything = async () => {
    try {
      loading.value = true
      
      // Clear frontend state
      images.value = []
      searchResults.value = []
      searchQuery.value = ''
      searchTotalResults.value = 0
      searchQueryReferenceImage.value = null
      error.value = null
      
      // Call backend to purge everything
      const response = await fetch(`${API_BASE_URL}/purge-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!response.ok) {
        throw new Error('Failed to purge images')
      }
      
      // Clear elasticsearch
      await fetch(`${API_BASE_URL}/cleanup-elasticsearch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }).catch(() => {})
      
      // Clear all image caches
      await clearImageCache()
      
      return true
    } catch (err) {
      error.value = err.message
      return false
    } finally {
      loading.value = false
    }
  }

  // Clean gallery of invalid images
  const cleanGallery = async () => {
    try {
      // Just fetch fresh images and clear cache
      await clearImageCache()
      return await fetchImages(100)
    } catch (err) {
      console.error('Error cleaning gallery:', err)
      return false
    }
  }

  // Get valid images 
  const getValidImages = () => {
    return images.value.filter(isValidImage)
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
    searchSimilarById,
    clearSearch,
    nukeEverything,
    cleanGallery,
    getValidImages,
    isValidImage,
    getFileName,
    clearImageCache,
    
    // Computed
    hasError,
    isLoading,
    searching,
    hasSearchResults
  }
}) 