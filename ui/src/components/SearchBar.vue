<template>
  <div class="search-container">
    <div class="search-input-wrapper">
      <input
        type="text"
        v-model="searchQuery"
        @input="handleSearch"
        placeholder="Search patterns, colors, styles..."
        :disabled="loading"
        class="search-input"
      >
      <div v-if="loading" class="search-spinner"></div>
      <button 
        v-if="searchQuery.length > 0 || hasActiveFilters" 
        @click="clearSearch" 
        class="clear-button"
        title="Clear search"
      >Clear</button>
    </div>
    
    <!-- Advanced search filters -->
    <div class="search-filters">
      <div class="filter-group">
        <label>Pattern Type:</label>
        <select v-model="filters.pattern_type" @change="applyFilters">
          <option value="">Any</option>
          <option value="paisley">Paisley</option>
          <option value="floral">Floral</option>
          <option value="geometric">Geometric</option>
          <option value="abstract">Abstract</option>
          <option value="damask">Damask</option>
          <option value="striped">Striped</option>
          <option value="polka dot">Polka Dot</option>
        </select>
      </div>
      
      <div class="filter-group">
        <label>Color:</label>
        <select v-model="filters.color" @change="applyFilters">
          <option value="">Any</option>
          <option value="red">Red</option>
          <option value="blue">Blue</option>
          <option value="green">Green</option>
          <option value="yellow">Yellow</option>
          <option value="purple">Purple</option>
          <option value="pink">Pink</option>
          <option value="orange">Orange</option>
          <option value="brown">Brown</option>
          <option value="black">Black</option>
          <option value="white">White</option>
          <option value="gray">Gray</option>
        </select>
      </div>
      
      <div class="filter-group">
        <label>Style:</label>
        <select v-model="filters.style" @change="applyFilters">
          <option value="">Any</option>
          <option value="traditional">Traditional</option>
          <option value="modern">Modern</option>
          <option value="abstract">Abstract</option>
          <option value="vintage">Vintage</option>
          <option value="minimalist">Minimalist</option>
          <option value="ornate">Ornate</option>
        </select>
      </div>
      
      <div class="filter-group">
        <label>Sort By:</label>
        <select v-model="sortMethod" @change="applyFilters">
          <option value="relevance">Relevance</option>
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
        </select>
      </div>
    </div>
    
    <div v-if="searchActive" class="search-info">
      <span>
        <strong>{{ displayedQuery }}</strong>
        <span v-if="hasActiveFilters" class="filter-indicators">
          <span v-if="filters.pattern_type" class="filter-tag">
            Pattern: {{ filters.pattern_type }}
          </span>
          <span v-if="filters.color" class="filter-tag">
            Color: {{ filters.color }}
          </span>
          <span v-if="filters.style" class="filter-tag">
            Style: {{ filters.style }}
          </span>
        </span>
      </span>
      <span v-if="resultsCount !== null" class="results-count">
        {{ resultsCount }} result{{ resultsCount !== 1 ? 's' : '' }} found
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive } from 'vue'
import { useImageStore } from '../stores/imageStore'

// Custom debounce function for search optimization
function debounce(fn, delay) {
  let timeoutId
  return (...args) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

const imageStore = useImageStore()
const searchQuery = ref('')
const filters = reactive({
  pattern_type: '',
  color: '',
  style: ''
})
const sortMethod = ref('relevance')

// Computed properties
const loading = computed(() => imageStore.isSearching)
const searchActive = computed(() => imageStore.searchQuery !== '' || hasActiveFilters.value)
const displayedQuery = computed(() => imageStore.searchQuery || 'All Patterns')
const resultsCount = computed(() => imageStore.searchTotalResults)
const hasActiveFilters = computed(() => 
  filters.pattern_type !== '' || 
  filters.color !== '' || 
  filters.style !== ''
)

// Debounced search function
const performSearch = debounce(async (query) => {
  if (query.length < 2 && !hasActiveFilters.value) return
  
  try {
    await imageStore.searchImages(
      query, 
      {
        pattern_type: filters.pattern_type,
        color: filters.color,
        style: filters.style
      },
      sortMethod.value
    )
  } catch (error) {
    console.error('Search failed:', error)
  }
}, 300)

// Methods
const handleSearch = (e) => {
  const value = e.target.value.trim()
  searchQuery.value = value
  
  if (!value && !hasActiveFilters.value) {
    clearSearch()
    return
  }
  
  performSearch(value)
}

const clearSearch = () => {
  searchQuery.value = ''
  imageStore.clearSearch()
}

const applyFilters = () => {
  performSearch(searchQuery.value)
}

// Initialize from store
searchQuery.value = imageStore.searchQuery
if (imageStore.searchFilters) {
  filters.pattern_type = imageStore.searchFilters.pattern_type || ''
  filters.color = imageStore.searchFilters.color || ''
  filters.style = imageStore.searchFilters.style || ''
}
sortMethod.value = imageStore.searchSort || 'relevance'
</script>

<style scoped>
.search-container {
  margin-bottom: 1rem;
  width: 100%;
}

.search-input-wrapper {
  position: relative;
  width: 100%;
  display: flex;
  align-items: center;
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

.clear-button {
  position: absolute;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #999;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}

.clear-button:hover {
  color: #555;
  background-color: #f0f0f0;
}

.search-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
}

.filter-group {
  display: flex;
  flex-direction: column;
  min-width: 150px;
}

.filter-group label {
  font-size: 0.8rem;
  color: #666;
  margin-bottom: 0.25rem;
}

.filter-group select {
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
  font-size: 0.9rem;
}

.search-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
  font-size: 0.85rem;
  color: #666;
  padding: 0.5rem 0;
  border-top: 1px solid #eee;
}

.filter-indicators {
  display: inline-flex;
  gap: 0.5rem;
  margin-left: 0.5rem;
}

.filter-tag {
  background-color: #f0f0f0;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  color: #555;
}

.results-count {
  font-weight: 500;
  color: #4CAF50;
}

@keyframes spin {
  0% { transform: translateY(-50%) rotate(0deg); }
  100% { transform: translateY(-50%) rotate(360deg); }
}

@media (max-width: 768px) {
  .search-filters {
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .filter-group {
    width: 100%;
  }
}
</style> 