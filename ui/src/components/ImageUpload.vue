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
        <p>okan hoca istifa</p>
      </div>
    </div>

    <div v-if="uploading" class="upload-progress">
      <div class="progress-bar">
        <div 
          class="progress-fill"
          :style="{ width: `${uploadProgress}%` }"
        ></div>
      </div>
      <p>Uploading... {{ uploadProgress }}%</p>
    </div>

    <!-- Add error message display -->
    <div v-if="imageStore.error" class="upload-error">
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

const handleDrop = async (e) => {
  isDragging.value = false
  const files = [...e.dataTransfer.files]
  try {
    await imageStore.uploadImages(files)
  } catch (error) {
    console.error('Drop upload failed:', error)
  }
}

const handleFileSelect = async (e) => {
  const files = [...e.target.files]
  try {
    uploading.value = true
    await imageStore.uploadImages(files)
  } catch (error) {
    console.error('File select upload failed:', error)
  } finally {
    uploading.value = false
    uploadProgress.value = 0
    // Reset file input
    e.target.value = ''
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

.upload-error {
  margin-top: 1rem;
  padding: 0.75rem;
  background-color: #fee;
  border: 1px solid #fcc;
  border-radius: 4px;
  color: #c00;
}
</style> 