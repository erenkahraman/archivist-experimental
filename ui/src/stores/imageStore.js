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
    
    // Check if the image has at least one valid path
    const hasPath = img.thumbnail_path || img.path || img.original_path
    
    if (!hasPath) return false
    
    // If the image has invalid_thumbnail flag, it's not valid
    if (img.invalid_thumbnail || img.failed_to_load) return false
    
    return true
  }

  // Clean filename helper
  const getFileName = (path) => {
    if (!path) return ''
    return path.split('/').pop()
  }

  // TEXT SEARCH FUNCTION
  // Search for images using text-based filtering
  const searchByText = async (query, limit = 100) => {
    try {
      // Set searching state and update query
      isSearching.value = true
      error.value = null
      searchQuery.value = query
      
      console.log(`Starting text search for: "${query}"`);
      
      // First, ensure we have the latest images (fetch ALL available images)
      await fetchImages(0) // Fetch all available images
      
      console.log(`Total images loaded: ${images.value.length}`);
      
      if (!query || query.trim() === '') {
        // If no query, return all images without limiting
        searchResults.value = images.value
        searchTotalResults.value = images.value.length
        console.log(`Empty query, returning all ${searchResults.value.length} images`);
        return searchResults.value
      }
      
      // Clean and normalize the search query
      const normalizedQuery = query.trim().toLowerCase()
      
      // Track the number of images that had debugInfo printed
      let debuggedImages = 0;
      
      // Filter images that have matching text in their metadata
      const filtered = images.value.filter((img, index) => {
        // Create a comprehensive searchable text from ALL metadata
        let allMetadata = '';
        let matchFound = false;
        let matchDetails = '';
        
        // Recursively extract ALL text from the entire object
        const getAllTextFromObject = (obj, prefix = '') => {
          if (!obj || typeof obj !== 'object') return '';
          
          let extractedText = '';
          
          // Handle objects and arrays
          if (Array.isArray(obj)) {
            // For arrays, check each element
            obj.forEach((item, i) => {
              // Handle primitive array items
              if (typeof item === 'string') {
                extractedText += ' ' + item.toLowerCase();
              } else if (typeof item === 'number') {
                extractedText += ' ' + item.toString();
              } else if (typeof item === 'object' && item !== null) {
                // Process nested objects in arrays
                extractedText += getAllTextFromObject(item, `${prefix}[${i}]`);
              }
            });
          } else {
            // Process object properties
            for (const [key, value] of Object.entries(obj)) {
              // Skip non-descriptive fields and circular references
              if (['id', 'thumbnail_path', 'original_path', 'path', 'file', 'isUploading'].includes(key)) {
                continue;
              }
              
              const currentPath = prefix ? `${prefix}.${key}` : key;
              
              // Extract text based on value type
              if (typeof value === 'string') {
                extractedText += ' ' + value.toLowerCase();
                
                // Check if this specific field contains a match (for debugging)
                if (value.toLowerCase().includes(normalizedQuery)) {
                  matchFound = true;
                  matchDetails += `Match in field: ${currentPath} = "${value}"\n`;
                }
              } else if (typeof value === 'number') {
                extractedText += ' ' + value.toString();
              } else if (Array.isArray(value)) {
                // Process array values
                value.forEach((item, i) => {
                  if (typeof item === 'string') {
                    extractedText += ' ' + item.toLowerCase();
                    
                    // Check for match in array item
                    if (item.toLowerCase().includes(normalizedQuery)) {
                      matchFound = true;
                      matchDetails += `Match in array: ${currentPath}[${i}] = "${item}"\n`;
                    }
                  } else if (typeof item === 'object' && item !== null) {
                    extractedText += getAllTextFromObject(item, `${currentPath}[${i}]`);
                  }
                });
              } else if (typeof value === 'object' && value !== null) {
                // Recursively process nested objects
                extractedText += getAllTextFromObject(value, currentPath);
              }
            }
          }
          
          return extractedText;
        };
        
        // Process the entire image object to get ALL text
        allMetadata = getAllTextFromObject(img);
        
        // Perform the search with the complete metadata
        const isMatch = allMetadata.includes(normalizedQuery);
        
        // Debug a sample of images - both matches and non-matches
        if ((isMatch && debuggedImages < 5) || 
            (!isMatch && index < 3 && debuggedImages < 5)) {
          debuggedImages++;
          
          // Get a sample of the metadata for logging
          let metadataSample = allMetadata.substring(0, 150) + '...';
          
          // For non-matches, do deeper analysis
          if (!isMatch) {
            console.log(`Non-matching image ${index}: ${img.id || getFileName(img.thumbnail_path || '')}`);
            console.log(`- Metadata sample: ${metadataSample}`);
            
            // Check if there are partial matches (e.g., "roses" when searching for "rose")
            if (allMetadata.includes(normalizedQuery.slice(0, -1)) || 
                allMetadata.includes(normalizedQuery + 's') ||
                allMetadata.includes(normalizedQuery + 'es')) {
              console.log(`- Contains partial match: ${normalizedQuery.slice(0, -1)} or ${normalizedQuery}s`);
            }
          } else {
            console.log(`Match found! Image ${index}: ${img.id || getFileName(img.thumbnail_path || '')}`);
            console.log(`- Match details: ${matchDetails}`);
            console.log(`- Metadata sample: ${metadataSample}`);
          }
        }
        
        return isMatch;
      });
      
      console.log(`Search complete. Found ${filtered.length} matches for "${normalizedQuery}"`);
      
      // Sample a few results for inspection
      if (filtered.length > 0) {
        const sample = filtered.slice(0, Math.min(5, filtered.length));
        console.log('Sample of matching images:', sample.map(img => img.id || getFileName(img.thumbnail_path || '')));
      } else {
        // Log a sample of the images and their metadata structure to help debug
        console.log('Search returned no results. Here is a sample image structure:');
        if (images.value.length > 0) {
          const sampleImage = images.value[0];
          console.log('Image object keys (top level):', Object.keys(sampleImage));
          
          // Log all fields recursively to understand the structure
          const describeObject = (obj, prefix = '', depth = 0) => {
            if (!obj || typeof obj !== 'object' || depth > 3) return;
            
            for (const [key, value] of Object.entries(obj)) {
              const path = prefix ? `${prefix}.${key}` : key;
              
              if (typeof value === 'string') {
                console.log(`${path} = "${value.substring(0, 50)}${value.length > 50 ? '...' : ''}"`);
              } else if (Array.isArray(value)) {
                console.log(`${path} = Array(${value.length})`);
                if (value.length > 0 && depth < 2) {
                  // Show a sample array item
                  const item = value[0];
                  if (typeof item === 'object' && item !== null) {
                    describeObject(item, `${path}[0]`, depth + 1);
                  } else if (typeof item === 'string') {
                    console.log(`${path}[0] = "${item.substring(0, 50)}${item.length > 50 ? '...' : ''}"`);
                  }
                }
              } else if (typeof value === 'object' && value !== null) {
                console.log(`${path} = Object`);
                describeObject(value, path, depth + 1);
              } else {
                console.log(`${path} = ${value}`);
              }
            }
          };
          
          describeObject(sampleImage);
        }
      }
      
      // Update the results - don't limit the results anymore
      searchResults.value = filtered;
      searchTotalResults.value = filtered.length;
      
      return searchResults.value;
    } catch (err) {
      console.error(`Search error: ${err.message}`);
      error.value = err.message
      return []
    } finally {
      isSearching.value = false
    }
  }

  // Fetch all images from the server with cache busting
  const fetchImages = async (limit = 20) => {
    try {
      loading.value = true
      error.value = null
      
      // Add cache busting parameter
      const cacheBuster = Date.now()
      
      // Always attempt to get all images by setting a very high limit
      // This ensures we don't hit arbitrary server limits
      const effectiveLimit = limit === 0 ? 1000 : limit
      const limitParam = `limit=${effectiveLimit}`
      const url = `${API_BASE_URL}/images/?${limitParam}&_=${cacheBuster}`
      
      console.log(`Fetching images with URL: ${url}`);
      
      const response = await fetch(url)
      
      if (!response.ok) {
        throw new Error(`Failed to fetch images: ${response.statusText}`)
      }
      
      let data
      try {
        data = await response.json()
      } catch (jsonError) {
        // Handle JSON parsing errors - server might return empty content or invalid JSON
        console.warn("Server returned invalid JSON response, using empty array")
        data = []
      }
      
      if (!Array.isArray(data)) {
        console.warn("Server returned non-array response, using empty array")
        data = []
      }
      
      console.log(`Fetched ${data.length} images from server`);
      
      // Keep only uploading images from the current state
      const uploadingImages = images.value.filter(img => img.isUploading)
      
      // Process and enhance image data with proper paths
      const validImages = data
        .filter(img => img && (img.thumbnail_path || img.path || img.original_path || img.file))
        .map(img => {
          // Make sure all images have paths that point to the right endpoints
          const filename = img.file || getFileName(img.original_path || img.path || '');
          if (filename) {
            return {
              ...img,
              // Ensure paths are properly formed for thumbnails and originals
              thumbnail_path: `${API_BASE_URL}/images/thumbnails/${filename}`,
              original_path: `${API_BASE_URL}/images/${filename}`
            };
          }
          return img;
        });
      
      console.log(`Processed ${validImages.length} valid images`);
      
      // Merge with existing images that have been marked as having failed thumbnails
      // This prevents the UI from constantly trying to load invalid thumbnails
      const existingImageMap = new Map();
      images.value.forEach(img => {
        if (img.failed_to_load || img.invalid_thumbnail) {
          const filename = getFileName(img.thumbnail_path || img.path || img.original_path || '');
          if (filename) {
            existingImageMap.set(filename, img);
          }
        }
      });
      
      // Merge existing invalid images with new valid ones
      const mergedImages = validImages.map(img => {
        const filename = getFileName(img.thumbnail_path || img.path || img.original_path || '');
        const existingImg = existingImageMap.get(filename);
        
        if (existingImg) {
          // Keep the invalid_thumbnail flag if it exists
          return {
            ...img,
            invalid_thumbnail: existingImg.invalid_thumbnail || false,
            failed_to_load: existingImg.failed_to_load || false
          };
        }
        
        return img;
      });
      
      // Set images with fresh data from server
      images.value = [...uploadingImages, ...mergedImages]
      
      return mergedImages
    } catch (err) {
      console.error(`Error fetching images: ${err.message}`);
      error.value = err.message
      
      // Don't clear the images array on error if we already have images
      // This prevents emptying the gallery on temporary network errors
      if (images.value.length === 0) {
        // Only set empty array if we don't have any images already
        images.value = []
      }
      
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
      const response = await fetch(`${API_BASE_URL}/images/delete/${encodeURIComponent(filename)}`, {
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
          // Check if endpoints exist first by checking response headers (don't parse body)
          const checkEndpoint = async (endpoint) => {
            try {
              const response = await fetch(`${API_BASE_URL}/${endpoint}/${encodeURIComponent(filename)}`, {
                method: 'HEAD'
              });
              return response.status !== 404; // Consider anything not 404 as potentially valid
            } catch (err) {
              return false;
            }
          };
          
          // Try to tell backend this file is missing using the endpoint if it exists
          const markMissingEndpointExists = await checkEndpoint('images/mark-missing');
          if (markMissingEndpointExists) {
            await fetch(`${API_BASE_URL}/images/mark-missing/${encodeURIComponent(filename)}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' }
            }).catch(() => {}); // Still catch errors
          }
          
          // Also try direct delete as fallback
          const deleteEndpointExists = await checkEndpoint('images/delete');
          if (deleteEndpointExists) {
            await fetch(`${API_BASE_URL}/images/delete/${encodeURIComponent(filename)}`, {
              method: 'DELETE'
            }).catch(() => {}); // Ignore errors
          }
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
            `${API_BASE_URL}/images/thumbnails/${filename}`,
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
      
      return true
    } catch (err) {
      console.warn('Error clearing image cache:', err)
      return false
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
      
      // Cache busting for API calls
      const cacheBuster = Date.now()
      
      // Try to use the API for similarity search
      let apiAvailable = false
      try {
        const pingResponse = await fetch(`${API_BASE_URL}/ping`, { 
          method: 'GET',
          // Add timeout to prevent long waits
          signal: AbortSignal.timeout(3000)
        })
        apiAvailable = pingResponse.ok
      } catch (err) {
        console.warn('API server not available, falling back to local search:', err)
        apiAvailable = false
      }
      
      let results = []
      
      // API-based search
      if (apiAvailable) {
        try {
          // Try to purge known nonexistent files from backend before search
          await fetch(`${API_BASE_URL}/purge-missing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          }).catch(() => {}) // Ignore errors if endpoint doesn't exist
          
          // Call similarity search API
          const apiUrl = `${API_BASE_URL}/similar/${encodeURIComponent(imageId)}?sort=${sortBySimilarity}&min_similarity=${minSimilarity}&limit=${k}&_=${cacheBuster}`
          
          console.log(`Calling API for similarity search: ${apiUrl}`)
          
          const response = await fetch(apiUrl)
          
          if (response.ok) {
            // Process the search results
            const searchData = await response.json()
            
            // Extract results array
            if (Array.isArray(searchData)) {
              results = searchData
            } else if (searchData.results && Array.isArray(searchData.results)) {
              results = searchData.results
            }
          } else {
            throw new Error(`API similarity search failed: ${response.statusText}`)
          }
        } catch (apiError) {
          console.warn('API search failed, falling back to local search:', apiError)
          apiAvailable = false
        }
      }
      
      // Fallback: Local metadata-based similarity search
      if (!apiAvailable || results.length === 0) {
        console.log('Using local metadata comparison for similarity search')
        
        // Get source image's metadata
        let sourceMetadata = {}
        let exactPattern = ''
        
        if (sourceImage) {
          // Get all metadata from source image
          sourceMetadata = { ...sourceImage }
          
          // Save the primary pattern for exact matching
          if (sourceImage.patterns && sourceImage.patterns.primary_pattern) {
            exactPattern = sourceImage.patterns.primary_pattern
          }
          
          console.log('Source image metadata:', {
            id: sourceImage.id,
            patterns: sourceImage.patterns,
            exactPattern
          })
        } else if (imageData) {
          // For dropped images without full metadata
          exactPattern = imageData.patternName || ''
          sourceMetadata = { 
            patterns: {
              primary_pattern: imageData.patternName,
            }
          }
          console.log('Using image data for search:', { patternName: imageData.patternName })
        }
        
        // Helper function to calculate similarity score between two metadata objects
        const calculateSimilarity = (sourceData, targetData) => {
          // Initialize score components for hierarchical ranking
          let primaryPatternScore = 0
          let keywordScore = 0 
          let promptScore = 0
          let otherScore = 0
          let scoreDetails = []
          
          // If no metadata to compare, return zero score
          if (!sourceData || !targetData) return { totalScore: 0, components: {} }
          
          // Helper for string similarity
          const getStringSimilarity = (str1, str2) => {
            if (!str1 || !str2) return 0
            if (typeof str1 !== 'string' || typeof str2 !== 'string') return 0
            
            // Convert strings to lowercase and trim
            const s1 = str1.toLowerCase().trim()
            const s2 = str2.toLowerCase().trim()
            
            // Exact match is highest value
            if (s1 === s2) return 1
            
            // Check if one contains the other completely
            if (s1.includes(s2) || s2.includes(s1)) {
              const coverage = Math.min(s1.length, s2.length) / Math.max(s1.length, s2.length)
              return 0.8 * coverage
            }
            
            // Check for word-level matches (more precise than character level)
            const words1 = s1.split(/\s+/).filter(w => w.length > 2)
            const words2 = s2.split(/\s+/).filter(w => w.length > 2)
            
            if (words1.length && words2.length) {
              let matchedWords = 0
              for (const w1 of words1) {
                if (words2.some(w2 => w1 === w2)) {
                  matchedWords++ // Exact word match
                } else if (words2.some(w2 => w1.includes(w2) || w2.includes(w1))) {
                  matchedWords += 0.5 // Partial word match
                }
              }
              
              if (matchedWords > 0) {
                const wordOverlapScore = matchedWords / Math.max(words1.length, words2.length)
                return 0.6 * wordOverlapScore
              }
            }
            
            return 0
          }
          
          // ===== TIER 1: PRIMARY PATTERN MATCHING (highest priority) =====
          const sourcePrimaryPattern = sourceData.patterns?.primary_pattern || exactPattern || ''
          const targetPrimaryPattern = targetData.patterns?.primary_pattern || ''
          
          if (sourcePrimaryPattern && targetPrimaryPattern && 
              typeof sourcePrimaryPattern === 'string' && 
              typeof targetPrimaryPattern === 'string') {
            
            // Exact primary pattern match (case insensitive)
            if (sourcePrimaryPattern.toLowerCase() === targetPrimaryPattern.toLowerCase()) {
              primaryPatternScore = 10.0 // Highest possible score
              scoreDetails.push(`Exact primary pattern match: +10.0 (${sourcePrimaryPattern})`)
            } 
            // High similarity in primary pattern
            else {
              const patternSimilarity = getStringSimilarity(sourcePrimaryPattern, targetPrimaryPattern)
              if (patternSimilarity > 0.7) { // Stricter threshold for primary pattern
                primaryPatternScore = 8.0 * patternSimilarity
                scoreDetails.push(`Similar primary pattern: +${primaryPatternScore.toFixed(2)} (${sourcePrimaryPattern} vs ${targetPrimaryPattern})`)
              }
              else if (patternSimilarity > 0.5) {
                primaryPatternScore = 6.0 * patternSimilarity
                scoreDetails.push(`Partial primary pattern match: +${primaryPatternScore.toFixed(2)} (${sourcePrimaryPattern} vs ${targetPrimaryPattern})`)
              }
            }
          }
          
          // ===== TIER 2: KEYWORDS & SECONDARY PATTERNS =====
          // Check secondary patterns (treated as keywords)
          const sourceSecondaryPatterns = sourceData.patterns?.secondary_patterns || []
          const targetSecondaryPatterns = targetData.patterns?.secondary_patterns || []
          
          // Process patterns as arrays
          const sourcePatterns = Array.isArray(sourceSecondaryPatterns) ? 
            sourceSecondaryPatterns : [sourceSecondaryPatterns].filter(Boolean)
          const targetPatterns = Array.isArray(targetSecondaryPatterns) ? 
            targetSecondaryPatterns : [targetSecondaryPatterns].filter(Boolean)
          
          // Get valid string patterns only
          const validSourcePatterns = sourcePatterns.filter(p => typeof p === 'string')
          const validTargetPatterns = targetPatterns.filter(p => typeof p === 'string')
          
          if (validSourcePatterns.length && validTargetPatterns.length) {
            let exactMatches = 0
            let partialMatches = 0
            
            // Check for exact secondary pattern matches first
            for (const sPattern of validSourcePatterns) {
              if (validTargetPatterns.some(t => t.toLowerCase() === sPattern.toLowerCase())) {
                exactMatches++
                scoreDetails.push(`Exact keyword match: +1.0 (${sPattern})`)
              } else {
                // Look for partial matches
                let bestMatchScore = 0
                let bestMatchPattern = ''
                
                for (const tPattern of validTargetPatterns) {
                  const similarity = getStringSimilarity(sPattern, tPattern)
                  if (similarity > 0.7 && similarity > bestMatchScore) {
                    bestMatchScore = similarity
                    bestMatchPattern = tPattern
                  }
                }
                
                if (bestMatchScore > 0) {
                  partialMatches += bestMatchScore
                  scoreDetails.push(`Similar keyword: +${(bestMatchScore).toFixed(2)} (${sPattern} vs ${bestMatchPattern})`)
                }
              }
            }
            
            // Calculate keyword score based on matches
            if (exactMatches > 0) {
              keywordScore += Math.min(5.0, exactMatches * 1.0) // Cap at 5.0
            }
            
            if (partialMatches > 0) {
              keywordScore += Math.min(2.5, partialMatches * 0.5) // Cap partial matches at 2.5
            }
          }
          
          // Check for tags/keywords in other fields
          const sourceTags = sourceData.tags || sourceData.patterns?.tags || []
          const targetTags = targetData.tags || targetData.patterns?.tags || []
          
          // Convert to arrays and filter valid strings
          const validSourceTags = (Array.isArray(sourceTags) ? sourceTags : [sourceTags])
            .filter(t => typeof t === 'string')
          const validTargetTags = (Array.isArray(targetTags) ? targetTags : [targetTags])
            .filter(t => typeof t === 'string')
          
          if (validSourceTags.length && validTargetTags.length) {
            let tagMatches = 0
            
            for (const sTag of validSourceTags) {
              if (validTargetTags.some(t => t.toLowerCase() === sTag.toLowerCase())) {
                tagMatches++
                scoreDetails.push(`Tag match: +0.75 (${sTag})`)
              }
            }
            
            if (tagMatches > 0) {
              keywordScore += Math.min(3.0, tagMatches * 0.75) // Cap at 3.0
            }
          }
          
          // ===== TIER 3: PROMPT TEXT COMPARISON =====
          // Compare prompts or descriptions if available
          const sourcePrompt = sourceData.prompt || sourceData.patterns?.prompt || sourceData.description || ''
          const targetPrompt = targetData.prompt || targetData.patterns?.prompt || targetData.description || ''
          
          if (typeof sourcePrompt === 'string' && typeof targetPrompt === 'string' &&
              sourcePrompt.length > 10 && targetPrompt.length > 10) {
            
            const promptSimilarity = getStringSimilarity(sourcePrompt, targetPrompt)
            
            if (promptSimilarity > 0.5) {
              promptScore = 3.0 * promptSimilarity
              scoreDetails.push(`Prompt similarity: +${promptScore.toFixed(2)}`)
            }
          }
          
          // ===== TIER 4: SUPPLEMENTARY ATTRIBUTES =====
          // Only consider these if we have some match from higher tiers
          if (primaryPatternScore > 0 || keywordScore > 0 || promptScore > 0) {
            // Category match
            const sourceCategory = sourceData.category || sourceData.patterns?.category || ''
            const targetCategory = targetData.category || targetData.patterns?.category || ''
            
            if (sourceCategory && targetCategory && 
                typeof sourceCategory === 'string' && typeof targetCategory === 'string') {
              if (sourceCategory.toLowerCase() === targetCategory.toLowerCase()) {
                otherScore += 1.0
                scoreDetails.push(`Category match: +1.0 (${sourceCategory})`)
              }
            }
            
            // Style match
            const sourceStyle = sourceData.style || sourceData.patterns?.style || ''
            const targetStyle = targetData.style || targetData.patterns?.style || ''
            
            if (sourceStyle && targetStyle && 
                typeof sourceStyle === 'string' && typeof targetStyle === 'string') {
              const styleSimilarity = getStringSimilarity(sourceStyle, targetStyle)
              if (styleSimilarity > 0.6) {
                const stylePoints = 1.0 * styleSimilarity
                otherScore += stylePoints
                scoreDetails.push(`Style match: +${stylePoints.toFixed(2)} (${sourceStyle} vs ${targetStyle})`)
              }
            }
            
            // Color matching (only if we have other matches)
            try {
              const sourceColors = sourceData.colors || sourceData.patterns?.colors || []
              const targetColors = targetData.colors || targetData.patterns?.colors || []
              
              // Convert to arrays
              const sColors = Array.isArray(sourceColors) ? sourceColors : [sourceColors].filter(Boolean)
              const tColors = Array.isArray(targetColors) ? targetColors : [targetColors].filter(Boolean)
              
              if (sColors.length && tColors.length) {
                let colorMatches = 0
                
                // Compare color names
                for (const sColor of sColors) {
                  if (!sColor || typeof sColor !== 'object') continue
                  const sColorName = sColor.name || ''
                  
                  if (!sColorName || typeof sColorName !== 'string') continue
                  
                  for (const tColor of tColors) {
                    if (!tColor || typeof tColor !== 'object') continue
                    const tColorName = tColor.name || ''
                    
                    if (!tColorName || typeof tColorName !== 'string') continue
                    
                    // Exact color name match
                    if (sColorName.toLowerCase() === tColorName.toLowerCase()) {
                      colorMatches += 1
                      break
                    }
                  }
                }
                
                if (colorMatches > 0) {
                  const colorPoints = Math.min(1.0, 0.2 * colorMatches)
                  otherScore += colorPoints
                  scoreDetails.push(`Color matches: +${colorPoints.toFixed(2)} (${colorMatches} colors)`)
                }
              }
            } catch (e) {
              // Skip color comparison if error
            }
          }
          
          // ===== FINAL SCORING =====
          // Calculate total score with hierarchical weighting
          // Formula ensures primary_pattern > keywords > prompt > others
          const totalScore = primaryPatternScore + 
                            (primaryPatternScore > 0 ? keywordScore : keywordScore * 0.5) + 
                            (keywordScore > 0 ? promptScore : promptScore * 0.3) +
                            (promptScore > 0 ? otherScore : otherScore * 0.2)
          
          // Add tiny random variation to prevent identical scores
          const finalScore = totalScore > 0 ? 
              totalScore + (Math.random() * 0.0001) : 0
          
          // Store score components for result visualization
          const scoreComponents = {
            primaryPattern: primaryPatternScore,
            keywords: keywordScore,
            prompt: promptScore,
            other: otherScore,
            total: totalScore,
            final: finalScore
          }
          
          // Log detailed scoring for debugging
          if (finalScore > 3.0 && Math.random() < 0.05) {
            console.log(`High score ${finalScore.toFixed(4)}, components:`, scoreComponents)
            console.log(`Details:`, scoreDetails)
          }
          
          return finalScore
        }
        
        // Higher minimum threshold for similarity
        const localMinSimilarity = 0.5 // Much higher threshold for meaningful results
        
        // Compute similarity scores for all images
        results = images.value
          .filter(img => {
            // Skip source image and uploading images
            if (img.isUploading) return false
            if (sourceImage && img.id === sourceImage.id) return false
            
            // Only include valid images
            return isValidImage(img)
          })
          .map(img => {
            // Calculate similarity score
            const similarityScore = calculateSimilarity(sourceMetadata, img)
            
            return {
              ...img,
              similarity: similarityScore.toFixed(4),
              similarityRaw: similarityScore
            }
          })
          .filter(result => parseFloat(result.similarity) >= localMinSimilarity) // Higher threshold
          .sort((a, b) => b.similarityRaw - a.similarityRaw) // Sort by raw similarity score
          .slice(0, k) // Limit results
        
        console.log(`Local search found ${results.length} similar images`)
      }
      
      // Skip further processing if no results
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
      
      // If we used the API, validate results with thumbnail checks
      if (apiAvailable) {
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
              headers: { 'Cache-Control': 'no-cache, no-store' },
              // Add timeout
              signal: AbortSignal.timeout(2000)
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
        results = checkedResults.filter(Boolean)
        
        // Delete all missing images in batch (in background)
        if (imagesToDelete.length > 0) {
          // We don't await this - let it run in background
          Promise.all(imagesToDelete.map(filename => 
            fetch(`${API_BASE_URL}/delete/${encodeURIComponent(filename)}`, { 
              method: 'DELETE' 
            }).catch(() => {})
          )).then(() => {
            console.log(`Deleted ${imagesToDelete.length} missing images from backend`)
          })
        }
      }
      
      // Store the results
      searchQuery.value = `Similar patterns`
      searchTotalResults.value = results.length
      searchResults.value = results
      
      // Keep uploading images and add search results
      const uploadingImages = images.value.filter(img => img.isUploading)
      images.value = [...uploadingImages, ...results]
      
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
      
      return results
    } catch (err) {
      console.error('Error in similarity search:', err)
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
    
    // Fetch fresh images - use 0 to get all images
    fetchImages(0)
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
      // Remove images with invalid thumbnails first
      images.value = images.value.filter(img => {
        // Keep uploading images
        if (img.isUploading) return true;
        
        // Remove images with invalid thumbnails
        if (img.invalid_thumbnail || img.failed_to_load) return false;
        
        // Keep all other images
        return true;
      });
      
      // Clear cache
      await clearImageCache();
      
      try {
        // Fetch fresh images, but don't fail the entire operation if this fails
        const freshImages = await fetchImages(100);
        return freshImages;
      } catch (fetchError) {
        console.warn("Could not refresh images during cleanup:", fetchError);
        return images.value || []; // Return current images as fallback
      }
    } catch (err) {
      console.error('Error cleaning gallery:', err);
      return images.value || []; // Return current images as fallback
    }
  }

  // Get valid images 
  const getValidImages = () => {
    return images.value.filter(isValidImage)
  }

  // Reset store to initial state
  const resetStore = async () => {
    // Clear all state
    images.value = []
    loading.value = false
    error.value = null
    searchResults.value = []
    searchQuery.value = ''
    searchTotalResults.value = 0
    isSearching.value = false
    searchQueryReferenceImage.value = null
    
    // Clear image cache
    await clearImageCache()
    
    // Reload images
    return await fetchImages()
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
    resetStore,
    searchByText,
    
    // Computed
    hasError,
    isLoading,
    searching,
    hasSearchResults
  }
}) 