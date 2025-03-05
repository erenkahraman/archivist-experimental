const { createApp } = Vue

createApp({
    data() {
        return {
            searchQuery: '',
            searchResults: [],
            uploading: false,
            uploadProgress: 0
        }
    },
    methods: {
        async handleDrop(e) {
            const files = [...e.dataTransfer.files]
            await this.uploadFiles(files)
        },
        async handleFileSelect(e) {
            const files = [...e.target.files]
            await this.uploadFiles(files)
        },
        async uploadFiles(files) {
            if (!files.length) return

            this.uploading = true
            this.uploadProgress = 0

            const formData = new FormData()
            files.forEach(file => {
                formData.append('files[]', file)
            })

            try {
                await axios.post('http://localhost:5000/api/upload', formData, {
                    onUploadProgress: (e) => {
                        this.uploadProgress = Math.round((e.loaded * 100) / e.total)
                    }
                })
                alert('Upload successful!')
            } catch (error) {
                console.error('Upload failed:', error)
                alert('Upload failed. Please try again.')
            } finally {
                this.uploading = false
            }
        },
        async handleSearch() {
            if (!this.searchQuery.trim()) {
                this.searchResults = []
                return
            }

            try {
                const response = await axios.post('http://localhost:5000/api/search', {
                    query: this.searchQuery,
                    k: 20
                })
                this.searchResults = response.data.results
            } catch (error) {
                console.error('Search failed:', error)
            }
        },
        getThumbnailUrl(path) {
            // Convert file path to URL
            return `http://localhost:5000/thumbnails/${path.split('/').pop()}`
        }
    }
}).mount('#app') 