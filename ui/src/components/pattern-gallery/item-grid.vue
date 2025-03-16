<template>
  <div class="gallery-grid">
    <GalleryItem 
      v-for="image in images" 
      :key="image.isUploading ? `uploading-${image.original_path}` : image.thumbnail_path"
      :image="image"
      :search-active="searchActive"
      @select="$emit('select-image', image)"
      @delete="$emit('confirm-delete', image)"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import GalleryItem from './item.vue'
import { useImageStore } from '../../store/imageStore'

const props = defineProps({
  images: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['select-image', 'confirm-delete'])

const imageStore = useImageStore()
const searchActive = computed(() => imageStore.searchQuery !== '')
</script>

<style scoped>
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem;
}
</style> 