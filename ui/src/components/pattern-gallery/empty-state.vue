<template>
  <div v-if="loading" class="gallery-loading">
    <div class="loading-spinner"></div>
    <p>Loading images...</p>
  </div>
  
  <div v-else class="gallery-empty">
    <p v-if="searchActive">No images found for your search.</p>
    <p v-else>Upload pattern images to get started</p>
  </div>
</template>

<script setup>
defineProps({
  loading: {
    type: Boolean,
    default: false
  },
  searchActive: {
    type: Boolean,
    default: false
  }
})
</script>

<style scoped>
.gallery-loading,
.gallery-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  color: #666;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #4CAF50;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style> 