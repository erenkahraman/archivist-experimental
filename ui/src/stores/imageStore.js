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

    // Search logging - start
    console.group('🔍 TEXT SEARCH REQUEST')
    console.log(`Query: "${cleanedQuery}"`)
    console.log(`Parameters: limit=${k}, min_similarity=${minSimilarity}`)
    console.groupEnd()

    try {
      // Make sure we have images loaded
      if (images.value.filter(img => !img.isUploading).length === 0) {
        await fetchImages()
      }
      
      // Add a cache busting parameter to prevent cached results
      const cacheBuster = Date.now()
      
      // Call the search API
      const response = await fetch(`${API_BASE_URL}/search?_=${cacheBuster}`, {
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
        
        // Search logging - results
        console.group('🔍 TEXT SEARCH RESULTS')
        console.log(`Found ${searchData.results.length} results for query "${cleanedQuery}"`)
        console.table(searchData.results.map(result => ({
          id: result.id || getFileName(result.path || result.thumbnail_path),
          pattern: result.patterns?.primary_pattern || 'Unknown',
          similarity: (parseFloat(result.similarity) * 100).toFixed(2) + '%',
          raw_score: result.raw_score?.toFixed(4) || 'N/A',
          theme: result.patterns?.main_theme || 'N/A',
          styleKeywords: Array.isArray(result.patterns?.style_keywords) ? 
            result.patterns.style_keywords.join(', ').substring(0, 50) : 'None'
        })))
        
        // Log detailed pattern information for deeper analysis
        console.group('📊 Detailed Pattern Analysis')
        searchData.results.forEach((result, index) => {
          const id = result.id || getFileName(result.path || result.thumbnail_path)
          console.group(`Result #${index + 1}: ${id} (${(parseFloat(result.similarity) * 100).toFixed(1)}%)`)
          
          if (result.patterns) {
            console.log('Main theme:', result.patterns.main_theme)
            console.log('Primary pattern:', result.patterns.primary_pattern)
            console.log('Confidence:', result.patterns.pattern_confidence)
            
            if (result.patterns.secondary_patterns && result.patterns.secondary_patterns.length > 0) {
              console.group('Secondary patterns:')
              result.patterns.secondary_patterns.forEach(pattern => {
                console.log(`- ${pattern.name} (${(pattern.confidence * 100).toFixed(1)}%)`)
              })
              console.groupEnd()
            }
            
            if (result.patterns.content_details && result.patterns.content_details.length > 0) {
              console.group('Content details:')
              result.patterns.content_details.forEach(detail => {
                console.log(`- ${detail.name} (${(detail.confidence * 100).toFixed(1)}%)`)
              })
              console.groupEnd()
            }
          }
          
          if (result.colors?.dominant_colors) {
            console.group('Colors:')
            result.colors.dominant_colors.slice(0, 3).forEach(color => {
              console.log(`- ${color.name} (${(color.proportion * 100).toFixed(1)}%)`)
            })
            console.groupEnd()
          }
          
          console.groupEnd() // Result
        })
        console.groupEnd() // Detailed Pattern Analysis
        console.groupEnd() // Text Search Results
        
        return searchResults.value
      } else {
        console.log('🔍 No results found for query:', cleanedQuery)
        return []
      }
    } catch (err) {
      error.value = err.message
      console.error('🔍 Search error:', err)
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
      const minSimilarity = 0.0001  // Always use the lowest threshold to get results
      const sortBySimilarity = options.sortBySimilarity === undefined ? true : options.sortBySimilarity
      
      // Find the source image if it exists in current images
      const sourceImage = images.value.find(img => 
        img.id === imageId || 
        getFileName(img.thumbnail_path || '') === imageId || 
        getFileName(img.original_path || '') === imageId
      )
      
      // Search logging - start
      console.group('🔍 SIMILARITY SEARCH REQUEST')
      console.log(`Reference image ID: ${imageId}`)
      console.log(`Parameters: limit=${k}, min_similarity=${minSimilarity}, sortBySimilarity=${sortBySimilarity}`)
      
      if (sourceImage) {
        console.group('Reference Image Details:')
        console.log('ID:', sourceImage.id || getFileName(sourceImage.path || sourceImage.thumbnail_path))
        if (sourceImage.patterns) {
          console.log('Pattern:', sourceImage.patterns.primary_pattern || 'Unknown')
          console.log('Main theme:', sourceImage.patterns.main_theme || 'N/A')
          console.log('Keywords:', Array.isArray(sourceImage.patterns.style_keywords) ? 
            sourceImage.patterns.style_keywords.join(', ').substring(0, 100) : 'None')
          
          if (sourceImage.patterns.secondary_patterns && sourceImage.patterns.secondary_patterns.length > 0) {
            console.group('Secondary patterns:')
            sourceImage.patterns.secondary_patterns.forEach(pattern => {
              console.log(`- ${pattern.name} (${(pattern.confidence * 100).toFixed(1)}%)`)
            })
            console.groupEnd()
          }
        }
        
        if (sourceImage.colors?.dominant_colors) {
          console.group('Colors:')
          sourceImage.colors.dominant_colors.slice(0, 3).forEach(color => {
            console.log(`- ${color.name} (${(color.proportion * 100).toFixed(1)}%)`)
          })
          console.groupEnd()
        }
        
        console.groupEnd() // Reference Image Details
      } else if (imageData) {
        console.log('Using dragged image reference with path:', imageData.thumbnailPath)
      }
      
      console.groupEnd() // Similarity Search Request
      
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
      
      // Search logging - results
      console.group('🔍 SIMILARITY SEARCH RESULTS')
      console.log(`Found ${results.length} similar images to ${imageId}`)
      
      console.table(results.map(result => ({
        id: result.id || getFileName(result.path || result.thumbnail_path),
        pattern: result.patterns?.primary_pattern || 'Unknown',
        similarity: (parseFloat(result.similarity) * 100).toFixed(2) + '%',
        text_score: result.text_score ? (parseFloat(result.text_score) * 100).toFixed(2) + '%' : 'N/A',
        vector_score: result.vector_score ? (parseFloat(result.vector_score) * 100).toFixed(2) + '%' : 'N/A',
        raw_score: result.raw_score?.toFixed(4) || 'N/A',
        theme: result.patterns?.main_theme || 'N/A',
        styleKeywords: Array.isArray(result.patterns?.style_keywords) ? 
          result.patterns.style_keywords.join(', ').substring(0, 50) : 'None'
      })))
      
      // Log detailed similarity comparison
      console.group('🔍 Detailed Similarity Analysis')
      
      if (sourceImage) {
        console.log('Reference pattern:', sourceImage.patterns?.primary_pattern || 'Unknown')
        console.log('Reference main theme:', sourceImage.patterns?.main_theme || 'N/A')
        
        // Extract reference features for comparison
        const refKeywords = sourceImage.patterns?.style_keywords || [];
        const refSecondaryPatterns = sourceImage.patterns?.secondary_patterns?.map(p => p.name) || [];
        const refColors = sourceImage.colors?.dominant_colors?.map(c => c.name) || [];
        
        // Log matches for each result
        results.forEach((result, index) => {
          const id = result.id || getFileName(result.path || result.thumbnail_path)
          
          // Include text & vector scores if available (for hybrid search)
          let scoreDetails = `${(parseFloat(result.similarity) * 100).toFixed(2)}%`;
          if (result.text_score !== undefined && result.vector_score !== undefined) {
            scoreDetails += ` (text: ${(parseFloat(result.text_score) * 100).toFixed(2)}%, vector: ${(parseFloat(result.vector_score) * 100).toFixed(2)}%)`;
          }
          
          console.group(`Result #${index + 1}: ${id} (${scoreDetails})`)
          
          // Calculate keyword matches
          const resultKeywords = result.patterns?.style_keywords || [];
          const keywordMatches = refKeywords.filter(kw => resultKeywords.includes(kw));
          
          // Calculate pattern matches
          const resultSecondaryPatterns = result.patterns?.secondary_patterns?.map(p => p.name) || [];
          const patternMatches = refSecondaryPatterns.filter(p => resultSecondaryPatterns.includes(p));
          
          // Calculate color matches
          const resultColors = result.colors?.dominant_colors?.map(c => c.name) || [];
          const colorMatches = refColors.filter(c => resultColors.includes(c));
          
          console.log('Match components:')
          console.log(`- Primary pattern: ${result.patterns?.primary_pattern === sourceImage.patterns?.primary_pattern ? '✅' : '❌'} (${result.patterns?.primary_pattern || 'Unknown'})`)
          console.log(`- Main theme: ${result.patterns?.main_theme === sourceImage.patterns?.main_theme ? '✅' : '❌'} (${result.patterns?.main_theme || 'Unknown'})`)
          console.log(`- Matching keywords: ${keywordMatches.length}/${refKeywords.length} (${keywordMatches.join(', ')})`)
          console.log(`- Matching secondary patterns: ${patternMatches.length}/${refSecondaryPatterns.length} (${patternMatches.join(', ')})`)
          console.log(`- Matching colors: ${colorMatches.length}/${Math.min(3, refColors.length)} (${colorMatches.join(', ')})`)
          
          console.groupEnd() // Result
        });
      }
      
      console.groupEnd() // Detailed Similarity Analysis
      console.groupEnd() // Similarity Search Results
      
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
    // Track if the active search results need cleaning
    const wasSearchActive = searchResults.value.length > 0;
    
    // Function to check if an image is valid by checking if it can be retrieved
    const checkValidImageFile = async (img) => {
      if (!img || !img.thumbnail_path) return false;
      
      try {
        // Try to fetch the thumbnail to see if it exists on the server
        const filename = getFileName(img.thumbnail_path);
        const response = await fetch(`${API_BASE_URL}/thumbnails/${filename}`, { method: 'HEAD' });
        return response.ok;
      } catch (e) {
        return false;
      }
    };
    
    // Filter out any images without valid paths
    const validImages = images.value.filter(isValidImage);
    
    // If we removed any, update the state
    if (validImages.length !== images.value.length) {
      // Store the list of invalid image IDs/paths so we can also filter them from search results
      const invalidIds = images.value
        .filter(img => !isValidImage(img))
        .map(img => img.id || img.path || img.thumbnail_path);
      
      images.value = validImages;
      
      // Also clean up search results if any
      if (wasSearchActive && searchResults.value.length > 0) {
        searchResults.value = searchResults.value.filter(result => {
          const resultId = result.id || result.path || result.thumbnail_path;
          return !invalidIds.includes(resultId);
        });
      }
      
      // If search reference image is invalid, clear search
      if (searchQueryReferenceImage.value && 
          invalidIds.includes(searchQueryReferenceImage.value.id || 
                              searchQueryReferenceImage.value.path || 
                              searchQueryReferenceImage.value.thumbnail_path)) {
        clearSearch();
      }
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
      searchQueryReferenceImage.value = null
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
      
      // Get the response and wait for it
      await response.json()
      
      // Additional call to ensure Elasticsearch is also cleared
      const esCleanupResponse = await fetch(`${API_BASE_URL}/cleanup-elasticsearch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      if (!esCleanupResponse.ok) {
        console.warn('Failed to cleanup Elasticsearch, but images were purged')
      }
      
      // Clear browser cache for /thumbnails/ requests to prevent stale references
      if ('caches' in window) {
        try {
          const cacheNames = await window.caches.keys()
          for (const cacheName of cacheNames) {
            await window.caches.delete(cacheName)
          }
        } catch (err) {
          console.warn('Failed to clear browser cache:', err)
        }
      }
      
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