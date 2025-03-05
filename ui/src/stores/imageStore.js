import { defineStore } from 'pinia'
import { ref, onMounted } from 'vue'
import { useStorage } from '@vueuse/core'

export const useImageStore = defineStore('images', () => {
  // useStorage ile local storage'da persist ediyoruz
  const images = useStorage('gallery-images', [])
  const loading = ref(false)
  const uploadingFiles = ref([])
  const searchResults = ref([])
  const searchQuery = ref('')
  const isSearching = ref(false)
  const error = ref(null)
  const API_URL = 'http://localhost:5001/api'

  // Uploading durumlarını temizleyen metod
  const clearUploadingStates = () => {
    images.value = images.value.filter(img => !img.isUploading)
  }

  const fetchImages = async () => {
    try {
      loading.value = true
      // Uploading durumlarını temizle ve mevcut resimleri getir
      clearUploadingStates()
      loading.value = false
    } catch (error) {
      console.error('Fetch error:', error)
      loading.value = false
    }
  }

  const uploadImages = async (files) => {
    try {
      const uploadedImages = []
      
      for (const file of files) {
        // Create temporary object with upload status
        const tempImage = {
          original_path: file.name,
          isUploading: true
        }
        images.value.unshift(tempImage)

        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch(`${API_URL}/upload`, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          // Upload başarısız olursa temp image'ı kaldır
          images.value = images.value.filter(img => img !== tempImage)
          throw new Error('Upload failed')
        }

        const metadata = await response.json()
        uploadedImages.push(metadata)

        // Replace temp image with actual metadata
        const index = images.value.findIndex(img => img.original_path === file.name)
        if (index !== -1) {
          images.value[index] = { ...metadata, isUploading: false }
        }
      }

      return uploadedImages
    } catch (error) {
      console.error('Upload error:', error)
      // Hata durumunda tüm uploading durumlarını temizle
      clearUploadingStates()
      throw error
    }
  }

  const deleteImage = async (filepath) => {
    try {
      const response = await fetch(`${API_URL}/delete/${encodeURIComponent(filepath)}`, {
        method: 'DELETE'
      })
      
      if (!response.ok) throw new Error('Delete failed')
      
      images.value = images.value.filter(img => img.original_path !== filepath)
    } catch (error) {
      console.error('Delete error:', error)
      throw error
    }
  }

  const searchImages = async (query) => {
    // If query is empty, just clear the search
    if (!query.trim()) {
      clearSearch()
      return
    }

    // Don't search for very short queries
    if (query.trim().length < 2) {
      return
    }

    // Set searching state without affecting loading state
    isSearching.value = true
    searchQuery.value = query

    try {
      console.log('Searching for:', query)
      
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ query }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(`Search failed: ${data.error || response.statusText}`)
      }

      searchResults.value = data.results || []
      return searchResults.value

    } catch (error) {
      console.error('Search error:', error)
      error.value = error.message
      searchResults.value = []
      throw error
    } finally {
      isSearching.value = false
    }
  }

  const clearSearch = () => {
    searchResults.value = []
    searchQuery.value = ''
    isSearching.value = false
  }

  const clearAllImages = () => {
    images.value = []
    searchResults.value = []
    searchQuery.value = ''
  }

  // Initialize store
  clearUploadingStates() // Store oluşturulduğunda uploading durumlarını temizle

  return {
    images,
    loading,
    uploadingFiles,
    searchResults,
    searchQuery,
    isSearching,
    error,
    fetchImages,
    uploadImages,
    deleteImage,
    searchImages,
    clearSearch,
    clearAllImages,
    clearUploadingStates
  }
}) 