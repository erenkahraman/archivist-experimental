<template>
  <div class="app-container">
    <div class="top-bar">
      <h1 class="app-title">Pattern Archivist</h1>
      <div class="controls">
        <ImageUpload />
        <SearchBar />
      </div>
    </div>
    <Gallery />
  </div>
</template>

<script setup>
import ImageUpload from './components/ImageUpload.vue'
import SearchBar from './components/SearchBar.vue'
import Gallery from './components/Gallery.vue'
</script>

<style>
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.top-bar {
  display: flex;
  flex-direction: column;
  margin-bottom: 2rem;
}

.app-title {
  font-size: 1.8rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #2c3e50;
}

.controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

@media (max-width: 768px) {
  .controls {
    flex-direction: column;
    align-items: stretch;
  }
}

header {
  margin-bottom: 2rem;
}

h1 {
  font-size: 2rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #2c3e50;
}

.stats {
  margin-bottom: 1rem;
  color: #666;
}

.error-message {
  padding: 1rem;
  margin: 1rem 0;
  background-color: #fee;
  border: 1px solid #fcc;
  border-radius: 4px;
  color: #c00;
}

.clear-button {
  margin-left: 1rem;
  padding: 0.5rem 1rem;
  background-color: #ff4444;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.clear-button:hover {
  background-color: #cc0000;
}
</style> 