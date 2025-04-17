<template>
  <div class="gallery-container">
    <!-- Reset and Cleanup Buttons -->
    <div class="action-controls">
      <button 
        class="cleanup-button" 
        @click="forceCleanup"
        title="Find and clean invalid images"
      >
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Refresh Gallery
      </button>
      
      <button 
        class="purge-button" 
        @click="purgeInvalidImages"
        title="Remove all images with invalid thumbnails"
      >
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Clean Invalid Images
      </button>
      
      <button 
        class="reset-button" 
        @click="confirmReset"
        title="WARNING: This will delete ALL images"
      >
        Reset All Images
      </button>
    </div>
    
    <!-- Empty state -->
    <div v-if="loading" class="gallery-placeholder animate__animated animate__pulse">
      <div class="loading-container">
        <div class="loading-spinner"></div>
        <p class="loading-text">Discovering images...</p>
      </div>
    </div>
    
    <!-- Empty state -->
    <div v-else-if="images.length === 0" class="gallery-empty animate__animated animate__fadeIn">
      <div class="empty-illustration">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M4 16L8.58579 11.4142C9.36684 10.6332 10.6332 10.6332 11.4142 11.4142L16 16M14 14L15.5858 12.4142C16.3668 11.6332 17.6332 11.6332 18.4142 12.4142L20 14M14 8H14.01M6 20H18C19.1046 20 20 19.1046 20 18V6C20 4.89543 19.1046 4 18 4H6C4.89543 4 4 4.89543 4 6V18C4 19.1046 4.89543 20 6 20Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <h3 v-if="searchActive" class="empty-title">No images found</h3>
      <h3 v-else class="empty-title">Your gallery is empty</h3>
      <p v-if="searchActive" class="empty-description">
        No similar images were found for your search. Make sure the reference image appears in the sidebar.
        <br>If you're using drag & drop, try refreshing the page and trying again.
      </p>
      
      <!-- Show reference image if we have one -->
      <div v-if="searchActive && imageStore.searchQueryReferenceImage" class="reference-image-container">
        <h4>Reference Image:</h4>
        <div class="reference-image">
          <img 
            :src="getThumbnailUrl(imageStore.searchQueryReferenceImage.thumbnail_path)" 
            :alt="getPatternName(imageStore.searchQueryReferenceImage)"
          >
          <div class="reference-info">
            <p>{{ getPatternName(imageStore.searchQueryReferenceImage) }}</p>
            <p class="ref-help">Try adjusting the minimum similarity threshold in the sidebar settings.</p>
          </div>
        </div>
      </div>
      
      <p v-else class="empty-description">Upload images to begin your collection</p>
    </div>
    
    <!-- Image grid -->
    <div v-else class="gallery-grid animate__animated animate__fadeIn">
      <div 
        v-for="image in images" 
        :key="image.id || image.timestamp || (image.thumbnail_path ? image.thumbnail_path : Math.random())"
        class="gallery-item"
        :class="{
          'is-uploading': image.isUploading,
          'is-searching': image.isSearching,
          'invalid-thumbnail': image.invalid_thumbnail || image.failed_to_load
        }"
        draggable="true"
        @dragstart="handleDragStart(image, $event)"
      >
        <!-- Uploading state -->
        <template v-if="image.isUploading">
          <div class="upload-placeholder">
            <div class="upload-spinner"></div>
            <div class="upload-progress">
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: `${image.uploadProgress || 0}%` }"></div>
              </div>
              <p class="progress-text">{{ image.uploadProgress || 0 }}%</p>
              <p class="upload-status">{{ image.uploadStatus || 'Uploading...' }}</p>
            </div>
          </div>
        </template>
        
        <!-- Regular image display -->
        <template v-else>
          <div class="image-actions">
            <button 
              class="delete-button"
              @click.stop="confirmDelete(image)"
              title="Delete image"
              aria-label="Delete image"
            >
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 6H5H21" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
          
          <div class="image-container" @click="handleImageClick(image)">
            <img 
              :src="getThumbnailUrl(image)" 
              :alt="getImageName(image)"
              class="gallery-image"
              loading="lazy"
              @error="handleImageError(image)"
            >
            
            <!-- Hover overlay with quick info -->
            <div class="image-overlay">
              <div class="overlay-content">
                <p class="overlay-title">
                  {{ getPatternName(image) }}
                  <span v-if="getPatternConfidence(image)" class="confidence-badge">
                    {{ formatConfidence(getPatternConfidence(image)) }}
                  </span>
                </p>
                
                <!-- Color chips -->
                <div class="color-chips" v-if="image.colors && image.colors.length > 0">
                  <div 
                    v-for="(color, idx) in image.colors.slice(0, 5)" 
                    :key="idx"
                    class="color-chip"
                    :style="{ backgroundColor: color.hex }"
                    :title="`${color.name} (${Math.round(color.proportion * 100)}%)`"
                  ></div>
                </div>
                
                <!-- Search score when in search mode -->
                <div v-if="searchActive && image.similarity !== undefined" class="search-score-info">
                  <span class="score-label">Match score: {{ (image.similarity * 100).toFixed(0) }}%</span>
                </div>
              </div>
            </div>
          </div>
          
          <div class="image-metadata">
            <div class="pattern-type">
              <span class="type-label">Pattern:</span>
              <span class="type-value">{{ getPatternName(image) }}</span>
            </div>
            
            <!-- Secondary patterns if available -->
            <div class="secondary-patterns" v-if="hasSecondaryPatterns(image)">
              <span class="sec-pattern-label">Also:</span>
              <div class="sec-pattern-tags">
                <span 
                  v-for="(pattern, idx) in getSecondaryPatterns(image)" 
                  :key="idx"
                  class="sec-pattern-tag"
                >
                  {{ pattern }}
                </span>
              </div>
            </div>
            
            <!-- Style keywords -->
            <div class="style-keywords" v-if="hasStyleKeywords(image)">
              <div class="keyword-tags">
                <span 
                  v-for="(keyword, idx) in getStyleKeywords(image)" 
                  :key="idx"
                  class="keyword-tag"
                >
                  {{ keyword }}
                </span>
              </div>
            </div>
            
            <!-- Display search score when in search mode -->
            <div v-if="searchActive && image.similarity !== undefined" class="search-score">
              <div class="score-bar" :style="{ width: `${image.similarity * 100}%` }"></div>
              <span class="score-label">Match: {{ (image.similarity * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Use the modular ImageModal component -->
    <ImageModal 
      v-if="selectedImage" 
      :selected-image="selectedImage" 
      @close="selectedImage = null"
    />

    <!-- Delete Confirmation Modal -->
    <div v-if="showDeleteConfirm" class="delete-modal" @click="showDeleteConfirm = false">
      <div class="delete-modal-content" @click.stop>
        <h3>Delete Image</h3>
        <p>Are you sure you want to delete this image?</p>
        <div class="delete-modal-actions">
          <button class="cancel-button" @click="showDeleteConfirm = false">Cancel</button>
          <button class="confirm-button" @click="handleDelete">Delete</button>
        </div>
      </div>
    </div>

    <!-- Add the reset confirmation modal -->
    <div v-if="showResetConfirm" class="reset-modal" @click="showResetConfirm = false">
      <div class="reset-modal-content" @click.stop>
        <h3>⚠️ WARNING: Delete ALL Images</h3>
        <p>This will permanently delete ALL images from the server.</p>
        <p>This action <strong>cannot be undone</strong>.</p>
        <div class="reset-modal-actions">
          <button class="cancel-button" @click="showResetConfirm = false">Cancel</button>
          <button class="confirm-button" @click="handleReset">Delete Everything</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useImageStore } from '../stores/imageStore'
import ImageModal from './modal/ImageModal.vue'

// Helper to control log level
const isDev = process.env.NODE_ENV === 'development'
const log = (message, ...args) => {
  if (isDev) {
    console.log(message, ...args)
  }
}

const imageStore = useImageStore()
const selectedImage = ref(null)
const showDeleteConfirm = ref(false)
const imageToDelete = ref(null)
const showResetConfirm = ref(false)
const isResetting = ref(false)

const images = computed(() => {
  // Check if a search (text or similarity) is active
  const searchIsActive = imageStore.searchQuery !== '' || imageStore.searchQueryReferenceImage !== null;

  if (searchIsActive) {
    // If search is active, use searchResults instead of all images
    // This is where the bug was - it was returning all valid images instead of filtered search results
    console.log(`Gallery loading ${imageStore.searchResults.length} search results`);
    return imageStore.searchResults;
  } else {
    // Default view: sort by timestamp (newest first)
    // Get valid images and sort them
    const validImages = imageStore.getValidImages() || [];
    console.log(`Gallery loading ${validImages.length} images (default view)`);
    return [...validImages].sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
  }
})
const loading = computed(() => imageStore.loading)
const searchActive = computed(() => {
  return imageStore.searchQuery !== '' || imageStore.searchQueryReferenceImage !== null;
})

onMounted(async () => {
  try {
    // Clear any stale cache
    await imageStore.clearImageCache();
    
    // Fetch fresh images - use 0 to get all images
    await imageStore.fetchImages(0);
    console.log(`Initially loaded ${imageStore.images.length} images`);
  } catch (error) {
    // Only log errors in development
    if (isDev) {
      console.error("Failed to initialize gallery:", error)
    }
  }
})

// Helper functions for pattern information
const getPatternName = (image) => {
  if (image.pattern && image.pattern.primary) {
    return image.pattern.primary;
  }
  if (image.patterns && image.patterns.main_theme) {
    return image.patterns.main_theme;
  }
  if (image.metadata && image.metadata.patterns && image.metadata.patterns.main_theme) {
    return image.metadata.patterns.main_theme;
  }
  return 'Unknown pattern';
}

const getPatternConfidence = (image) => {
  if (image.pattern && image.pattern.confidence) {
    return image.pattern.confidence;
  }
  if (image.patterns && image.patterns.pattern_confidence) {
    return image.patterns.pattern_confidence;
  }
  return null;
}

const formatConfidence = (confidence) => {
  const percent = Math.round(confidence * 100);
  return `${percent}%`;
}

const hasSecondaryPatterns = (image) => {
  if (image.pattern && image.pattern.secondary && image.pattern.secondary.length > 0) {
    return true;
  }
  if (image.patterns && image.patterns.secondary_patterns && image.patterns.secondary_patterns.length > 0) {
    return true;
  }
  return false;
}

const getSecondaryPatterns = (image) => {
  if (image.pattern && image.pattern.secondary) {
    return image.pattern.secondary;
  }
  if (image.patterns && image.patterns.secondary_patterns) {
    return image.patterns.secondary_patterns.map(p => p.name);
  }
  return [];
}

const hasStyleKeywords = (image) => {
  if (image.style_keywords && image.style_keywords.length > 0) {
    return true;
  }
  if (image.patterns && image.patterns.style_keywords && image.patterns.style_keywords.length > 0) {
    return true;
  }
  return false;
}

const getStyleKeywords = (image) => {
  if (image.style_keywords) {
    return image.style_keywords;
  }
  if (image.patterns && image.patterns.style_keywords) {
    return image.patterns.style_keywords;
  }
  return [];
}

// Handle image error
const handleImageError = (image) => {
  // Check for valid image
  if (!image) return;
  
  // Log the error for debugging
  const imagePath = image.thumbnail_path || image.path || image.original_path || 'unknown';
  console.warn('Thumbnail failed to load:', imagePath);
  
  // Get the filename for processing
  const filename = getFileName(imagePath);
  
  try {
    // Mark this image as failed
    if (image) {
      // Set flags on the image object to avoid re-loading
      image.invalid_thumbnail = true;
      image.failed_to_load = true;
    }
    
    // Find the image in the main store and mark it
    const storeImage = imageStore.images.value.find(img => {
      const imgPath = img.thumbnail_path || img.path || img.original_path || '';
      return getFileName(imgPath) === filename;
    });
    
    if (storeImage) {
      storeImage.invalid_thumbnail = true;
      storeImage.failed_to_load = true;
    }
    
    // Clear cache for this image
    imageStore.clearImageCache(filename);
    
    // Don't remove the image from UI immediately - just mark it as invalid
    // This prevents flickering and lets the user still see metadata
    
    // Only check if this was the reference image for search
    if (imageStore.searchQueryReferenceImage) {
      const refPath = imageStore.searchQueryReferenceImage.thumbnail_path || 
                     imageStore.searchQueryReferenceImage.original_path || '';
      
      if (getFileName(refPath) === filename) {
        imageStore.clearSearch();
      }
    }
  } catch (err) {
    console.error('Error handling failed image:', err);
  }
}

const getImageName = (image) => {
  // Check various possible path properties
  const path = image.file_path || image.image_path || image.original_path || 
              image.path || image.thumbnail_path || '';
  
  if (!path) return 'Unknown image';
  
  return path.split('/').pop();
}

// Handle image click to open modal
const handleImageClick = (image) => {
  // Just directly set the selected image without modifications
  selectedImage.value = image;
}

const confirmDelete = (image) => {
  if (!image) return
  imageToDelete.value = image
  showDeleteConfirm.value = true
}

const handleDelete = async () => {
  if (!imageToDelete.value) {
    if (isDev) {
      console.error("Cannot delete: Image is undefined")
    }
    showDeleteConfirm.value = false
    return
  }

  // Try to find a valid path using the same logic as getImageName
  const image = imageToDelete.value
  const path = image.original_path || image.file_path || image.image_path || 
              image.path || image.thumbnail_path

  if (path) {
    try {
      await imageStore.deleteImage(path)
      showDeleteConfirm.value = false
      imageToDelete.value = null
      if (selectedImage.value === imageToDelete.value) {
        selectedImage.value = null
      }
    } catch (error) {
      if (isDev) {
        console.error("Failed to delete image:", error)
      }
    }
  } else {
    if (isDev) {
      console.error("Cannot delete: Image path is undefined")
    }
    showDeleteConfirm.value = false
    imageToDelete.value = null
  }
}

const truncatePrompt = (prompt) => {
  return getPromptText(prompt, true);
}

const getPromptText = (prompt, truncate = false) => {
  if (!prompt) return 'No description available';
  
  // Handle both string and object formats
  const promptText = typeof prompt === 'string' 
    ? prompt 
    : (prompt.final_prompt || 'No description available');
    
  if (truncate && promptText.length > 100) {
    return promptText.substring(0, 100) + '...';
  }
  
  return promptText;
}

// Replace getThumbnailUrl function
const getThumbnailUrl = (image) => {
  if (!image) return '';
  
  // Skip if we already know the thumbnail is invalid
  if (image.invalid_thumbnail || image.failed_to_load) {
    // Return a placeholder image instead
    return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzMzMzMzMyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjIwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5OTk5IiBkeT0iLjNlbSI+SW1hZ2UgTm90IEZvdW5kPC90ZXh0Pjwvc3ZnPg==';
  }
  
  // If the image already has a full thumbnail URL, use it
  if (image.thumbnail_path && image.thumbnail_path.includes('http')) {
    return image.thumbnail_path;
  }
  
  // For images that have a complete thumbnail_path
  if (image.thumbnail_path) {
    // If it's not a complete URL but has the API path structure
    if (image.thumbnail_path.includes('/api/')) {
      return image.thumbnail_path;
    }
    
    // Otherwise, just use the filename
    const filename = imageStore.getFileName(image.thumbnail_path);
    return `${imageStore.API_BASE_URL}/images/thumbnails/${filename}`;
  }
  
  // If there's a file property (from recent uploads)
  if (image.file) {
    return `${imageStore.API_BASE_URL}/images/thumbnails/${image.file}`;
  }
  
  // Last resort: try to get filename from original_path
  if (image.original_path) {
    const filename = imageStore.getFileName(image.original_path);
    return `${imageStore.API_BASE_URL}/images/thumbnails/${filename}`;
  }
  
  // If no valid path can be found
  return '';
}

// Add these functions for the reset functionality
const confirmReset = () => {
  showResetConfirm.value = true
}

const handleReset = async () => {
  try {
    isResetting.value = true
    
    // Clear search state first
    imageStore.clearSearch()
    
    // Clear any selected image
    selectedImage.value = null
    
    // Use our improved deleteAllImages function
    const result = await deleteAllImages()
    console.log(`Reset complete: ${result.deleted} images deleted, ${result.failed} failed`)
    
    showResetConfirm.value = false
    
    // No need to fetch images again - deleteAllImages already does this
    
  } catch (error) {
    if (isDev) {
      console.error("Failed to reset images:", error)
    }
  } finally {
    isResetting.value = false
  }
}

// Custom function to delete all images one by one
const deleteAllImages = async () => {
  console.log("Deleting all images one by one...")
  
  // Get all valid images
  const allImages = imageStore.getValidImages()
  console.log(`Found ${allImages.length} images to delete`)
  
  // Set a counter for status tracking
  let deletedCount = 0
  let failedCount = 0
  let processedCount = 0
  
  // First, clear any active search to ensure we have the full image set
  imageStore.clearSearch()
  
  // Try to fetch fresh image data before deleting to ensure we have all records
  try {
    await imageStore.fetchImages(0)
    // Use a safe access pattern to avoid TypeError when accessing images.value.length
    const imagesCount = imageStore.images && imageStore.images.value ? imageStore.images.value.length : 0
    console.log(`Refreshed image list, now have ${imagesCount} images`)
  } catch (error) {
    console.warn("Could not refresh image list:", error)
  }
  
  // Get updated list
  const refreshedImages = imageStore.getValidImages()
  const imagesToProcess = refreshedImages.length > allImages.length ? refreshedImages : allImages
  console.log(`Proceeding to delete ${imagesToProcess.length} images`)
  
  // Create a more robust path resolution function
  const getImagePath = (image) => {
    if (!image) return null
    
    // Try all possible path properties in order of preference
    const pathOptions = [
      image.original_path,
      image.file_path,
      image.image_path, 
      image.path,
      image.thumbnail_path
    ]
    
    // Return the first non-empty path
    for (const path of pathOptions) {
      if (path && typeof path === 'string' && path.trim() !== '') {
        return path
      }
    }
    
    // If we have an id or filename, try to construct a path
    if (image.id) {
      return `${imageStore.API_BASE_URL}/images/${image.id}`
    }
    
    if (image.file) {
      return `${imageStore.API_BASE_URL}/images/${image.file}`
    }
    
    return null
  }
  
  // Process images in batches to avoid overwhelming the API
  const BATCH_SIZE = 5
  const totalBatches = Math.ceil(imagesToProcess.length / BATCH_SIZE)
  
  for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
    const startIdx = batchIndex * BATCH_SIZE
    const endIdx = Math.min(startIdx + BATCH_SIZE, imagesToProcess.length)
    const batch = imagesToProcess.slice(startIdx, endIdx)
    
    console.log(`Processing batch ${batchIndex + 1}/${totalBatches} (${batch.length} images)`)
    
    // Process each image in the batch
    const batchPromises = batch.map(async (image) => {
      try {
        // Get a valid path for the image
        const path = getImagePath(image)
        
        if (!path) {
          console.warn("Skipping image with no path", image)
          return { success: false, error: "No valid path" }
        }
        
        // Delete the image
        await imageStore.deleteImage(path)
        return { success: true }
      } catch (error) {
        console.error("Failed to delete image:", error)
        return { success: false, error: error.message }
      }
    })
    
    // Wait for all batch operations to complete
    const results = await Promise.allSettled(batchPromises)
    
    // Count results
    results.forEach(result => {
      processedCount++
      
      if (result.status === 'fulfilled') {
        if (result.value.success) {
          deletedCount++
        } else {
          failedCount++
        }
      } else {
        failedCount++
        console.error("Promise rejected:", result.reason)
      }
    })
    
    // Update progress
    console.log(`Progress: ${processedCount}/${imagesToProcess.length} (${deletedCount} deleted, ${failedCount} failed)`)
    
    // Small delay between batches to reduce server load
    if (batchIndex < totalBatches - 1) {
      await new Promise(resolve => setTimeout(resolve, 300))
    }
  }
  
  console.log(`Deletion complete. Deleted: ${deletedCount}, Failed: ${failedCount}`)
  
  // Final operations to ensure clean state
  try {
    // Clear browser cache
    await imageStore.clearImageCache()
    
    // Clean gallery to ensure UI is updated
    try {
      await imageStore.cleanGallery()
    } catch (err) {
      // Ignore cleanGallery errors - the server might be empty
      console.log("Gallery cleanup completed with possible warnings (this is normal after deletion)")
    }
    
    // Try to fix CORS issues with direct server endpoints
    try {
      // Check if API is available first
      const checkApi = async () => {
        try {
          const pingResponse = await fetch(`${imageStore.API_BASE_URL}/ping`, { 
            method: 'GET',
            mode: 'no-cors' 
          })
          return pingResponse.ok
        } catch (err) {
          return false
        }
      }
      
      const apiAvailable = await checkApi()
      
      if (apiAvailable) {
        // Use the correct API endpoint from the store
        const endpoint = `${imageStore.API_BASE_URL}/purge-missing`
        const headers = {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          // Add CORS headers that might help
          'Access-Control-Allow-Origin': '*'
        }
        
        // Use mode: 'no-cors' to bypass CORS errors
        await fetch(endpoint, { 
          method: 'POST', 
          headers,
          mode: 'no-cors'
        })
        console.log("Purge attempt made (best effort)")
      }
    } catch (err) {
      // Ignore server errors
      console.log("Server cleanup skipped - this is normal if server is unavailable")
    }
    
    // Fetch fresh images with cache busting
    try {
      await imageStore.fetchImages(0)
    } catch (err) {
      // The server might return an error if there are no images left, this is expected
      console.log("Image refresh completed - if empty, this is expected behavior")
    }
  } catch (error) {
    console.warn("Error during cleanup phase:", error.message)
  }
  
  return { total: processedCount, deleted: deletedCount, failed: failedCount }
}

// Add draggable attribute and event handler to the gallery item
const handleDragStart = (image, event) => {
  // Store the image ID (or path) in the drag event
  const imageId = image.id || getFileName(image.thumbnail_path || image.path || image.original_path || '');
  event.dataTransfer.setData('text/plain', imageId);
  
  // Store thumbnail information for display in search results
  const imageData = {
    thumbnailPath: image.thumbnail_path,
    originalPath: image.original_path || image.path,
    patternName: getPatternName(image)
  };
  event.dataTransfer.setData('application/json', JSON.stringify(imageData));
  
  event.dataTransfer.effectAllowed = 'copy';
  
  // Add a CSS class for visual feedback
  event.target.closest('.gallery-item').classList.add('is-dragging');
  
  // Add a dragend event listener to remove the class when dragging ends
  event.target.closest('.gallery-item').addEventListener('dragend', () => {
    event.target.closest('.gallery-item').classList.remove('is-dragging');
  }, { once: true });
}

// Helper to get filename from path
const getFileName = (path) => {
  if (!path) return '';
  return path.split('/').pop();
}

// Add a method to force cleanup
const forceCleanup = async () => {
  try {
    // Set a loading indicator
    imageStore.loading = true;
    
    // Clear search if active
    if (searchActive.value) {
      imageStore.clearSearch();
    }
    
    // Try to clear server caches and handle CORS issues
    const callServerEndpoint = async (endpoint, method = 'POST') => {
      try {
        const headers = {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        };
        
        // Use timeout promise to prevent hanging on server issues
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout')), 3000)
        );
        
        // Use no-cors mode to avoid CORS errors and add timeout
        const fetchPromise = fetch(`${imageStore.API_BASE_URL}/${endpoint}`, {
          method,
          headers,
          mode: 'no-cors'
        });
        
        // Race between fetch and timeout
        await Promise.race([fetchPromise, timeoutPromise]);
        console.log(`Called ${endpoint} endpoint (best effort)`);
      } catch (err) {
        // Silently ignore errors - this is a best effort
        console.log(`Attempted to call ${endpoint} - ignoring errors`);
      }
    };
    
    // Try server-side cache clearing in a way that handles CORS errors
    await callServerEndpoint('clear-cache');
    await callServerEndpoint('purge-missing');
    
    // Clear browser cache - this is more reliable
    await imageStore.clearImageCache();
    
    // Fetch fresh images with cache busting - use 0 to get all images
    try {
      await imageStore.fetchImages(0);
      const imagesCount = imageStore.images && imageStore.images.value ? imageStore.images.value.length : 0;
      console.log(`Cleanup complete, loaded ${imagesCount} images`);
    } catch (err) {
      console.log("Error refreshing images, using current state");
    }
  } catch (error) {
    if (isDev) {
      console.error("Failed to clean gallery:", error);
    }
  } finally {
    imageStore.loading = false;
  }
}

// Add a method to purge invalid images
const purgeInvalidImages = async () => {
  try {
    // Find all images with invalid thumbnails
    const invalidImages = imageStore.images.value.filter(img => 
      img.invalid_thumbnail || img.failed_to_load
    );
    
    if (invalidImages.length === 0) {
      console.log("No invalid images to purge");
      return;
    }
    
    console.log(`Found ${invalidImages.length} invalid images to purge`);
    
    // Set a counter for status tracking
    let deletedCount = 0;
    let failedCount = 0;
    
    // Process each invalid image
    for (const image of invalidImages) {
      try {
        // Get a valid path for deletion
        const path = image.original_path || image.thumbnail_path || image.path;
        
        if (!path) {
          console.warn("Skipping image with no path");
          failedCount++;
          continue;
        }
        
        // Delete the image
        await imageStore.deleteImage(path);
        deletedCount++;
        
        if (deletedCount % 5 === 0) {
          console.log(`Deleted ${deletedCount}/${invalidImages.length} invalid images...`);
        }
      } catch (error) {
        console.error("Failed to delete invalid image:", error);
        failedCount++;
      }
    }
    
    console.log(`Purge complete. Deleted: ${deletedCount}, Failed: ${failedCount}`);
    
    // Refresh the gallery
    await imageStore.fetchImages(0);
  } catch (error) {
    console.error("Failed to purge invalid images:", error);
  }
};
</script>

<style scoped>
.gallery-container {
  width: 100%;
  margin: 0 var(--space-1);
}

.action-controls {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.cleanup-button, 
.purge-button {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  background-color: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: var(--radius-md);
  padding: var(--space-2) var(--space-4);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cleanup-button:hover,
.purge-button:hover {
  background-color: rgba(59, 130, 246, 0.15);
  color: #2563eb;
}

.cleanup-button svg,
.purge-button svg {
  width: 16px;
  height: 16px;
}

.purge-button {
  background-color: rgba(234, 88, 12, 0.1);
  color: #ea580c;
  border: 1px solid rgba(234, 88, 12, 0.2);
}

.purge-button:hover {
  background-color: rgba(234, 88, 12, 0.15);
  color: #c2410c;
}

.reset-container {
  display: flex;
  justify-content: flex-end;
  margin-bottom: var(--space-4);
}

.reset-button {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  background-color: rgba(251, 113, 133, 0.1);
  color: #fb7185;
  border: 1px solid rgba(251, 113, 133, 0.2);
  border-radius: var(--radius-md);
  padding: var(--space-2) var(--space-4);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.reset-button:hover {
  background-color: rgba(251, 113, 133, 0.15);
  color: #f43f5e;
}

.gallery-placeholder, 
.gallery-empty {
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  padding: var(--space-8);
  text-align: center;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(79, 70, 229, 0.1);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-text {
  font-size: 1.1rem;
  color: var(--color-text-light);
  margin: 0;
}

.empty-illustration {
  width: 80px;
  height: 80px;
  color: var(--color-text-lighter);
  margin-bottom: var(--space-4);
  opacity: 0.6;
}

.empty-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0 0 var(--space-2) 0;
  color: var(--color-text);
}

.empty-description {
  font-size: 1rem;
  color: var(--color-text-light);
  margin: 0;
  max-width: 400px;
}

.reference-image-container {
  margin-top: var(--space-4);
  width: 100%;
  max-width: 400px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.reference-image-container h4 {
  font-size: 0.9rem;
  color: var(--color-text-light);
  margin-top: 0;
  margin-bottom: var(--space-2);
}

.reference-image {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.reference-image img {
  width: 100px;
  height: 100px;
  object-fit: cover;
  border-radius: var(--radius-sm);
}

.reference-info p {
  margin: 0;
  font-size: 0.9rem;
}

.reference-info .ref-help {
  color: var(--color-text-lighter);
  font-size: 0.85rem;
  margin-top: var(--space-2);
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: var(--space-4);
  width: 100%;
  justify-content: center;
  margin: 0 auto;
}

.gallery-item {
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
}

.gallery-item.invalid-thumbnail {
  border-color: rgba(239, 68, 68, 0.3);
  background: rgba(239, 68, 68, 0.05);
}

.gallery-item.invalid-thumbnail .image-container {
  opacity: 0.7;
}

.gallery-item.invalid-thumbnail .image-container::before {
  content: "Missing Thumbnail";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #ef4444;
  font-size: 0.9rem;
  font-weight: 500;
  background: rgba(0, 0, 0, 0.7);
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  z-index: 2;
}

.gallery-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
  border-color: rgba(79, 70, 229, 0.3);
}

.image-actions {
  position: absolute;
  top: var(--space-2);
  right: var(--space-2);
  z-index: 3;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.gallery-item:hover .image-actions {
  opacity: 1;
}

.delete-button {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background-color: rgba(0, 0, 0, 0.5);
  border: none;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  backdrop-filter: blur(4px);
}

.delete-button:hover {
  background-color: rgba(0, 0, 0, 0.7);
}

.delete-button svg {
  width: 16px;
  height: 16px;
}

.image-container {
  position: relative;
  height: 200px;
  overflow: hidden;
  cursor: pointer;
}

.gallery-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s ease;
}

.gallery-item:hover .gallery-image {
  transform: scale(1.05);
}

.image-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.7) 0%, rgba(0, 0, 0, 0) 60%);
  display: flex;
  align-items: flex-end;
  padding: var(--space-3);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.gallery-item:hover .image-overlay {
  opacity: 1;
}

.overlay-content {
  width: 100%;
}

.overlay-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: white;
  margin: 0 0 var(--space-2) 0;
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.confidence-badge {
  font-size: 0.7rem;
  background-color: rgba(79, 70, 229, 0.4);
  color: white;
  padding: 2px 6px;
  border-radius: var(--radius-full);
  backdrop-filter: blur(4px);
}

.color-chips {
  display: flex;
  gap: 4px;
}

.color-chip {
  width: 15px;
  height: 15px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.search-score-info {
  margin-top: var(--space-2);
  font-size: 0.75rem;
  color: white;
}

.image-metadata {
  padding: var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  background: rgba(255, 255, 255, 0.02);
  flex: 1;
}

.pattern-type {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  font-size: 0.85rem;
}

.type-label {
  color: var(--color-text-light);
  font-weight: 500;
  font-size: 0.75rem;
}

.type-value {
  color: var(--color-text);
  font-weight: 600;
}

.secondary-patterns {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-2);
  align-items: center;
  font-size: 0.75rem;
}

.sec-pattern-label {
  color: var(--color-text-light);
}

.sec-pattern-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.sec-pattern-tag {
  background-color: rgba(79, 70, 229, 0.1);
  color: var(--color-primary);
  padding: 2px 6px;
  border-radius: var(--radius-full);
  font-size: 0.7rem;
}

.style-keywords {
  margin-top: var(--space-1);
}

.keyword-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.keyword-tag {
  background-color: rgba(34, 211, 238, 0.1);
  color: #22d3ee;
  padding: 2px 6px;
  border-radius: var(--radius-full);
  font-size: 0.7rem;
}

.search-score {
  margin-top: var(--space-2);
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: var(--radius-full);
  height: 6px;
  overflow: hidden;
  position: relative;
}

.score-bar {
  height: 100%;
  background: linear-gradient(to right, #4f46e5, #3b82f6);
  border-radius: var(--radius-full);
}

.score-label {
  font-size: 0.7rem;
  color: var(--color-text-light);
  margin-top: 4px;
  display: block;
}

/* Upload placeholder */
.upload-placeholder {
  height: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-4);
  gap: var(--space-4);
  background-color: rgba(255, 255, 255, 0.03);
}

.upload-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(79, 70, 229, 0.1);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.upload-progress {
  width: 100%;
  text-align: center;
}

.progress-bar {
  height: 6px;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-bottom: var(--space-2);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(to right, #4f46e5, #3b82f6);
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--color-primary);
  margin: 0;
}

.upload-status {
  font-size: 0.8rem;
  color: var(--color-text-light);
  margin: var(--space-1) 0 0 0;
}

/* Modal styles */
.delete-modal,
.reset-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.delete-modal-content,
.reset-modal-content {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  width: 90%;
  max-width: 400px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
  animation: scaleIn 0.3s ease;
}

@keyframes scaleIn {
  from { transform: scale(0.9); }
  to { transform: scale(1); }
}

.delete-modal-content h3,
.reset-modal-content h3 {
  margin-top: 0;
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: var(--space-3);
  color: var(--color-text);
}

.reset-modal-content h3 {
  color: #f43f5e;
}

.delete-modal-content p,
.reset-modal-content p {
  color: var(--color-text-light);
  margin-bottom: var(--space-4);
}

.delete-modal-actions,
.reset-modal-actions {
  display: flex;
  gap: var(--space-3);
  justify-content: flex-end;
}

.cancel-button {
  padding: var(--space-2) var(--space-4);
  background-color: transparent;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: var(--color-text);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-button:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.confirm-button {
  padding: var(--space-2) var(--space-4);
  background-color: rgba(251, 113, 133, 0.1);
  color: #fb7185;
  border: 1px solid rgba(251, 113, 133, 0.2);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.confirm-button:hover {
  background-color: rgba(251, 113, 133, 0.2);
  color: #f43f5e;
}

/* Responsive design */
@media (max-width: 1400px) {
  .gallery-grid {
    grid-template-columns: repeat(5, 1fr);
  }
}

@media (max-width: 1200px) {
  .gallery-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

@media (max-width: 992px) {
  .gallery-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .gallery-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: var(--space-3);
  }
  
  .image-container {
    height: 150px;
  }
  
  .pattern-type {
    font-size: 0.8rem;
  }
  
  .sec-pattern-label,
  .sec-pattern-tag {
    font-size: 0.7rem;
  }
}
</style> 