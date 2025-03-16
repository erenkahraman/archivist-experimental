<template>
  <div class="upload-placeholder">
    <div class="upload-spinner"></div>
    <div class="upload-progress">
      <div class="progress-bar">
        <div 
          class="progress-fill"
          :style="{ width: `${progress || 0}%` }"
        ></div>
      </div>
      <p class="progress-text">{{ progress || 0 }}%</p>
      <p class="upload-status">{{ status || 'Uploading...' }}</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  progress: {
    type: Number,
    default: 0
  },
  status: {
    type: String,
    default: 'Uploading...'
  }
})
</script>

<style scoped>
.upload-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(76, 175, 80, 0.05);
  border-radius: 8px;
  padding: 1rem;
}

.upload-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #4CAF50;
  border-radius: 50%;
  margin-bottom: 1rem;
  animation: spin 1s linear infinite;
}

.upload-progress {
  width: 100%;
  text-align: center;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background-color: #eee;
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.9rem;
  color: #666;
  margin: 0.25rem 0;
}

.upload-status {
  font-size: 0.8rem;
  color: #666;
  margin: 0;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style> 