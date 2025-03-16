<template>
  <div class="gallery-container">
    <EmptyState 
      v-if="loading || images.length === 0" 
      :loading="loading" 
      :search-active="searchActive" 
    />
    
    <GalleryGrid v-else :images="images" @select-image="selectImage" @confirm-delete="confirmDelete" />

    <ImageDetailModal 
      v-if="selectedImage" 
      :image="selectedImage" 
      @close="selectedImage = null" 
    />

    <DeleteConfirmModal 
      v-if="showDeleteConfirm" 
      @cancel="showDeleteConfirm = false" 
      @confirm="handleDelete" 
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useImageStore } from '../../store/imageStore'
import GalleryGrid from './gallery-grid.vue'
import EmptyState from './empty-state.vue'
import ImageDetailModal from './modals/image-detail-modal.vue'
import DeleteConfirmModal from './modals/delete-confirm-modal.vue'

const imageStore = useImageStore()
const selectedImage = ref(null)
const showDeleteConfirm = ref(false)
const imageToDelete = ref(null)

const images = computed(() => imageStore.images || [])
const loading = computed(() => imageStore.loading)
const searchActive = computed(() => imageStore.searchQuery !== '')

onMounted(() => {
  imageStore.fetchImages()
  imageStore.clearUploadingStates()
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