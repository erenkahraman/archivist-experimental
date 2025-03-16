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
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M7 10v2h10v-2H7zm5-8l-5 5h3v4h4v-4h3l-5-5z"/>
        </svg>
        <p>Drop pattern images here or click to upload</p>
      </div>
    </div>

    <div v-if="uploading" class="upload-progress">
      <div class="progress-bar">
        <div 
          class="progress-fill"
          :style="{ width: `${uploadProgress}%` }"
        ></div>
      </div>
      <p>Uploading {{ currentFileIndex }} of {{ totalFiles }}... {{ uploadProgress }}%</p>
      <p class="upload-status">{{ uploadStatus }}</p>
    </div>

    <!-- Add error message display -->
    <div v-if="imageStore.error" class="upload-error">
      {{ imageStore.error }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useImageStore } from '../store/imageStore'

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
            
            // Replace temp image with actual metadata
            const index = imageStore.images.findIndex(img => 
              img.original_path === file.name && img.isUploading
            )
            
            if (index !== -1) {
              imageStore.images[index] = { ...metadata, isUploading: false }
            }
            
            resolve(metadata)
          } catch (error) {
            reject(new Error('Invalid response format'))
          }
        } else {
          reject(new Error(`Upload failed with status ${xhr.status}`))
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
  margin-bottom: 2rem;
}

.dropzone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.dropzone-active {
  border-color: #4CAF50;
  background-color: rgba(76, 175, 80, 0.1);
}

.dropzone:hover {
  border-color: #666;
}

.file-input {
  display: none;
}

.upload-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 1rem;
  color: #666;
}

.upload-progress {
  margin-top: 1rem;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background-color: #eee;
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}

.upload-status {
  font-size: 0.9rem;
  color: #666;
  margin-top: 0.5rem;
}

.upload-error {
  margin-top: 1rem;
  padding: 0.75rem;
  background-color: #fee;
  border: 1px solid #fcc;
  border-radius: 4px;
  color: #c00;
}
</style> 