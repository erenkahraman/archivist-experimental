<template>
  <div class="upload-container">
    <div
      class="dropzone"
      :class="{ 'dropzone-active': isDragging }"
      @dragenter.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @dragover.prevent
      @drop.prevent="handleDrop"
      @click="$refs.fileInput.click()"
    >
      <input
        ref="fileInput"
        type="file"
        multiple
        accept="image/*"
        class="file-input"
        @change="handleFileSelect"
      >
      <div class="dropzone-content">
        <div class="upload-icon-container">
          <svg class="upload-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 16L12 12L8 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M12 20V12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M20.39 18.39C21.3654 17.8583 22.1359 17.0169 22.5799 15.9986C23.024 14.9804 23.1162 13.8432 22.846 12.7667C22.5758 11.6901 21.9574 10.7355 21.0771 10.0534C20.1967 9.37137 19.1085 9.00072 18 8.99998H16.74C16.4373 7.82923 15.8731 6.74232 15.0899 5.82098C14.3067 4.89964 13.3248 4.16785 12.2181 3.68059C11.1113 3.19334 9.90851 2.96329 8.70008 3.00711C7.49164 3.05093 6.30903 3.36754 5.24114 3.93488C4.17325 4.50222 3.24812 5.30425 2.53088 6.28676C1.81365 7.26927 1.32293 8.40984 1.0943 9.62292C0.865665 10.836 0.909332 12.0868 1.22231 13.2784C1.53529 14.47 2.1142 15.5765 2.92 16.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 16L12 12L8 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <h3 class="upload-title">Upload Your Images</h3>
        <p class="upload-text">Drop images here or click to browse</p>
      </div>
    </div>

    <div v-if="uploading" class="upload-progress animate__animated animate__fadeIn">
      <div class="progress-bar">
        <div 
          class="progress-fill"
          :style="{ width: `${uploadProgress}%` }"
        ></div>
      </div>
      <div class="progress-details">
        <p class="progress-status">
          <span class="progress-counter">{{ currentFileIndex }} of {{ totalFiles }}</span>
          <span class="progress-percentage">{{ uploadProgress }}%</span>
        </p>
        <p class="upload-status">{{ uploadStatus }}</p>
      </div>
    </div>

    <!-- Add error message display -->
    <div v-if="imageStore.error" class="upload-error animate__animated animate__shakeX">
      <svg class="error-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 8V12M12 16H12.01M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>{{ imageStore.error }}</span>
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

const uploadUrl = 'http://localhost:8000/api/upload'

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
            
            // Ensure the metadata has a valid path
            if (!metadata.original_path && metadata.path) {
              metadata.original_path = metadata.path
            }
            
            // Replace temp image with actual metadata
            const index = imageStore.images.findIndex(img => 
              img.original_path === file.name && img.isUploading
            )
            
            if (index !== -1) {
              // Log the path information for debugging
              console.log('Image path before update:', imageStore.images[index].original_path)
              console.log('New image path:', metadata.original_path || metadata.path || 'undefined')
              
              // Update the image with the new metadata
              imageStore.images[index] = { 
                ...metadata, 
                isUploading: false,
                // Ensure we have a valid path
                original_path: metadata.original_path || metadata.path || `uploads/${file.name}`
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
.upload-container {
  margin-bottom: var(--space-8);
}

.dropzone {
  border: 2px dashed var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-8);
  text-align: center;
  cursor: pointer;
  transition: all var(--transition-normal);
  background: var(--color-surface);
}

.dropzone-active {
  border-color: var(--color-primary);
  background-color: rgba(79, 70, 229, 0.05);
  transform: scale(1.01);
  box-shadow: var(--shadow-md);
}

.dropzone:hover {
  border-color: var(--color-primary-light);
  background-color: rgba(255, 255, 255, 0.8);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.file-input {
  display: none;
}

.upload-icon-container {
  display: flex;
  justify-content: center;
  margin-bottom: var(--space-4);
}

.upload-icon {
  width: 64px;
  height: 64px;
  color: var(--color-primary);
}

.upload-title {
  font-family: var(--font-heading);
  font-size: 1.5rem;
  margin-bottom: var(--space-2);
  color: var(--color-text);
}

.upload-text {
  color: var(--color-text-light);
  font-size: 1rem;
}

.upload-progress {
  margin-top: var(--space-6);
  padding: var(--space-4);
  border-radius: var(--radius-md);
  background-color: var(--color-surface);
  box-shadow: var(--shadow-sm);
}

.progress-bar {
  height: 8px;
  background-color: var(--color-border);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-bottom: var(--space-2);
}

.progress-fill {
  height: 100%;
  background: var(--gradient-primary);
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
}

.progress-details {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.progress-status {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  color: var(--color-text);
}

.progress-counter {
  font-weight: 500;
}

.progress-percentage {
  font-weight: 600;
  color: var(--color-primary);
}

.upload-status {
  font-size: 0.85rem;
  color: var(--color-text-light);
}

.upload-error {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-top: var(--space-4);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  background-color: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
  color: var(--color-error);
}

.error-icon {
  width: 20px;
  height: 20px;
  stroke: var(--color-error);
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .dropzone {
    padding: var(--space-6);
  }
  
  .upload-icon {
    width: 48px;
    height: 48px;
  }
  
  .upload-title {
    font-size: 1.2rem;
  }
}
</style> 