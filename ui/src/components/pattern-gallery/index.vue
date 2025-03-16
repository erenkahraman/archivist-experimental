<template>
  <div class="gallery-container">
    <!-- Debug info -->
    <div v-if="loading">Loading images... ({{ imageStore.images.length }} found in store)</div>
    <div v-else-if="images.length === 0">
      No images found. API may be unavailable.
      <pre>Store has {{ imageStore.images.length }} images</pre>
    </div>
    <div v-else>Found {{ images.length }} images</div>
    
    <EmptyState 
      v-if="loading || images.length === 0" 
      :loading="loading" 
      :search-active="searchActive" 
    />
    
    <ItemGrid v-else :images="images" @select-image="selectImage" @confirm-delete="confirmDelete" />

    <DetailModal 
      v-if="selectedImage" 
      :image="selectedImage" 
      @close="selectedImage = null" 
    />

    <DeleteModal 
      v-if="showDeleteConfirm" 
      @cancel="showDeleteConfirm = false" 
      @confirm="handleDelete" 
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useImageStore } from '../../store/imageStore'
import ItemGrid from './item-grid.vue'
import EmptyState from './empty-state.vue'
import DetailModal from './modals/detail-modal.vue'
import DeleteModal from './modals/delete-modal.vue'

const imageStore = useImageStore()
const selectedImage = ref(null)
const showDeleteConfirm = ref(false)
const imageToDelete = ref(null)

const images = computed(() => imageStore.images)
const loading = computed(() => imageStore.loading)
const searchActive = computed(() => imageStore.searchQuery !== '')

onMounted(async () => {
  console.log('Gallery mounted, fetching images...')
  await imageStore.fetchImages()
  imageStore.clearUploadingStates()
  console.log('After fetch, store has', imageStore.images.length, 'images')
})

const selectImage = (image) => {
  console.log('Selected image data:', image)
  console.log('Pattern analysis:', image.patterns)
  console.log('Color analysis:', image.colors)
  selectedImage.value = image
}

const confirmDelete = (image) => {
  imageToDelete.value = image
  showDeleteConfirm.value = true
}

const handleDelete = async () => {
  if (imageToDelete.value) {
    await imageStore.deleteImage(imageToDelete.value.original_path)
    showDeleteConfirm.value = false
    imageToDelete.value = null
    if (selectedImage.value === imageToDelete.value) {
      selectedImage.value = null
    }
  }
}
</script>

<style scoped>
.gallery-container {
  min-height: 200px;
}
</style> 