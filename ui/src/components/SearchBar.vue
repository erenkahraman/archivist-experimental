<template>
  <div class="search-container">
    <div class="search-input-wrapper">
      <input
        type="text"
        v-model="searchQuery"
        @input="handleInput"
        placeholder="Search patterns (min. 2 characters)..."
        :disabled="loading"
        class="search-input"
      >
      <div v-if="loading" class="search-spinner"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  loading: Boolean
})

const emit = defineEmits(['search'])
const searchQuery = ref('')
let debounceTimeout = null

const handleInput = (e) => {
  const value = e.target.value.trim()
  
  // Clear previous timeout
  if (debounceTimeout) {
    clearTimeout(debounceTimeout)
  }

  // If query is empty, clear search immediately
  if (!value) {
    emit('search', '')
    return
  }

  // If query is less than 2 characters, don't search
  if (value.length < 2) {
    return
  }

  // Debounce search for longer queries
  debounceTimeout = setTimeout(() => {
    emit('search', value)
  }, 300)  // Reduced from 500ms to 300ms for better responsiveness
}
</script>

<style scoped>
.search-container {
  margin-bottom: 2rem;
}

.search-input-wrapper {
  position: relative;
  width: 100%;
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  border: 2px solid #ddd;
  border-radius: 8px;
  outline: none;
  transition: border-color 0.3s ease;
}

.search-input:focus {
  border-color: #4CAF50;
}

.search-spinner {
  position: absolute;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  width: 20px;
  height: 20px;
  border: 2px solid #f3f3f3;
  border-top: 2px solid #4CAF50;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: translateY(-50%) rotate(0deg); }
  100% { transform: translateY(-50%) rotate(360deg); }
}
</style> 