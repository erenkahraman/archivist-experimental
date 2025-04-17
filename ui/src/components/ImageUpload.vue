<template>
  <div class="upload-card">
    <input
      ref="fileInput"
      type="file"
      multiple
      accept="image/*"
      class="file-input"
      @change="handleFileSelect"
    >
    
    <!-- Upload button -->
    <button 
      class="upload-btn"
      :class="{ 'is-dragging': isDragging }"
      @dragenter.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @dragover.prevent
      @drop.prevent="handleDrop"
      @click="$refs.fileInput.click()"
    >
      <svg class="upload-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 15V3M12 3L7 8M12 3L17 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M3 15V18C3 18.5304 3.21071 19.0391 3.58579 19.4142C3.96086 19.7893 4.46957 20 5 20H19C19.5304 20 20.0391 19.7893 20.4142 19.4142C20.7893 19.0391 21 18.5304 21 18V15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>Add Images</span>
    </button>
    
    <!-- Upload progress indicator -->
    <div v-if="uploading" class="upload-progress">
      <div class="progress-indicator">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: `${uploadProgress}%` }"></div>
        </div>
        <span class="progress-text">{{ uploadProgress }}% · {{ currentFileIndex }}/{{ totalFiles }}</span>
      </div>
    </div>
    
    <!-- Error message -->
    <div v-if="imageStore.error" class="error-message">
      {{ imageStore.error }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useImageStore } from '../stores/imageStore'

const imageStore = useImageStore()
const isDragging = ref(false)
const uploading = ref(false)
const uploadProgress = ref(0)
const currentFileIndex = ref(0)
const totalFiles = ref(0)
const uploadStatus = ref('')

const uploadUrl = 'http://localhost:8000/api/upload/'

const handleDrop = async (e) => {
  isDragging.value = false
  const files = [...e.dataTransfer.files]
  if (files.length > 0) {
    await uploadFiles(files)
  }
}

const handleFileSelect = async (e) => {
  const files = [...e.target.files]
  if (files.length > 0) {
    await uploadFiles(files)
    // Reset file input
    e.target.value = ''
  }
}

const uploadFiles = async (files) => {
  try {
    uploading.value = true
    uploadProgress.value = 0
    currentFileIndex.value = 0
    totalFiles.value = files.length
    
    for (let i = 0; i < files.length; i++) {
      currentFileIndex.value = i + 1
      uploadStatus.value = `Processing ${files[i].name}`
      
      // Upload each file individually for better progress tracking
      await uploadSingleFile(files[i])
      
      // Update overall progress
      uploadProgress.value = Math.round((i + 1) / files.length * 100)
    }
    
    uploadStatus.value = 'All uploads completed successfully!'
    
    // Fetch updated images after all uploads
    await imageStore.fetchImages()
  } catch (error) {
    console.error('Upload failed:', error)
    uploadStatus.value = `Upload failed: ${error.message}`
  } finally {
    // Keep progress visible for a moment so user can see completion
    setTimeout(() => {
      uploading.value = false
      uploadProgress.value = 0
    }, 1500)
  }
}

const uploadSingleFile = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  // Create a temporary image object to show in the gallery
  const tempImage = {
    original_path: file.name,
    isUploading: true,
    uploadProgress: 0,
    uploadStatus: 'Preparing upload...'
  }
  
  // Add to the store
  imageStore.images.unshift(tempImage)
  
  try {
    // Track upload progress with fetch and XMLHttpRequest
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const fileProgress = Math.round((event.loaded / event.total) * 100)
          
          // Update the temporary image's progress
          const index = imageStore.images.findIndex(img => 
            img.original_path === file.name && img.isUploading
          )
          
          if (index !== -1) {
            imageStore.images[index].uploadProgress = fileProgress
            imageStore.images[index].uploadStatus = `Uploading: ${fileProgress}%`
          }
          
          uploadStatus.value = `Uploading ${file.name}: ${fileProgress}%`
        }
      })
      
      xhr.onload = async function() {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const metadata = JSON.parse(xhr.responseText)
            console.log('Received metadata from server:', metadata)
            
            // Replace temp image with actual metadata
            const index = imageStore.images.findIndex(img => 
              img.original_path === file.name && img.isUploading
            )
            
            if (index !== -1) {
              // Log the path information for debugging
              console.log('Image path before update:', imageStore.images[index].original_path)
              
              // Ensure we have the right path from the response
              let imagePath = metadata.file || '';
              if (metadata.metadata && metadata.metadata.original_path) {
                imagePath = metadata.metadata.original_path;
              }
              
              console.log('New image path:', imagePath)
              
              // Update the image with the new metadata
              imageStore.images[index] = { 
                ...metadata, 
                isUploading: false,
                // Ensure we have a valid path that points to the correct endpoint
                original_path: imagePath ? `${imageStore.API_BASE_URL}/images/${imagePath}` : `uploads/${file.name}`
              }
              
              console.log('Updated image in store:', imageStore.images[index])
            }
            
            resolve(metadata)
          } catch (error) {
            console.error('Error parsing server response:', error, xhr.responseText)
            reject(new Error('Invalid response format'))
          }
        } else {
          let errorMessage = 'Upload failed'
          try {
            const errorData = JSON.parse(xhr.responseText)
            errorMessage = errorData.error || `Upload failed with status ${xhr.status}`
          } catch (e) {
            errorMessage = `Upload failed with status ${xhr.status}`
          }
          console.error(errorMessage)
          reject(new Error(errorMessage))
        }
      }
      
      xhr.onerror = () => reject(new Error('Network error during upload'))
      
      xhr.open('POST', uploadUrl, true)
      xhr.send(formData)
    })
  } catch (error) {
    // Remove the temporary image on error
    imageStore.images = imageStore.images.filter(img => 
      !(img.original_path === file.name && img.isUploading)
    )
    throw error
  }
}
</script>

<style scoped>
.upload-card {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 100%;
  gap: var(--space-2);
}

.file-input {
  display: none;
}

/* Upload button */
.upload-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  background: #1a1a1a;
  border: 1px solid #2e392a;
  border-radius: var(--radius-md);
  color: #bfb78f;
  font-size: 0.9rem;
  font-weight: 500;
  padding: var(--space-2) var(--space-3);
  cursor: pointer;
  transition: all 0.2s ease;
  width: 100%;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.upload-btn:hover {
  background: #222;
  border-color: #3b4a37;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}

.upload-btn.is-dragging {
  background: #252525;
  border-color: #4c5c48;
  box-shadow: 0 0 0 2px rgba(46, 57, 42, 0.3);
}

.upload-icon {
  width: 20px;
  height: 20px;
  color: #95a389;
}

/* Progress indicator */
.upload-progress {
  width: 100%;
  overflow: hidden;
  border-radius: var(--radius-md);
  background: rgba(30, 35, 30, 0.3);
  padding: var(--space-1);
  border: 1px solid rgba(46, 57, 42, 0.2);
}

.progress-indicator {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.progress-bar {
  height: 4px;
  background-color: rgba(46, 57, 42, 0.2);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #4c5c48;
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.75rem;
  color: #95a389;
  text-align: right;
}

/* Error message */
.error-message {
  font-size: 0.8rem;
  color: #b59090;
  background-color: rgba(40, 30, 30, 0.4);
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-md);
  border-left: 2px solid #794141;
}

/* Mobile adjustments */
@media (max-width: 768px) {
  .upload-btn {
    padding: var(--space-1) var(--space-2);
    font-size: 0.8rem;
  }
  
  .upload-icon {
    width: 16px;
    height: 16px;
  }
}
</style> 