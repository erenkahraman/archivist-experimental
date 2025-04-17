const { createApp } = Vue

createApp({
    data() {
        return {
            searchQuery: '',
            galleryImages: [], // Master list of all images
            uploading: false,
            uploadProgress: 0
        }
    },
    computed: {
        filteredGalleryImages() {
            const normalizedQuery = this.searchQuery.trim().toLowerCase();
            console.log(`Computed property updating with query: "${normalizedQuery}"`);

            if (!normalizedQuery) {
                return this.galleryImages;
            }

            const filtered = this.galleryImages.filter(image => {
                const searchText = image.searchText;
                return typeof searchText === 'string' && searchText.toLowerCase().includes(normalizedQuery);
            });
            console.log('Computed property finished. Filtered images:', filtered);
            return filtered;
        }
    },
    mounted() {
        this.loadAllImages(); // Load all images on mount
    },
    methods: {
        async loadAllImages() {
            try {
                console.log('Fetching all images...');
                const response = await axios.get(`http://localhost:5000/api/all_images`);
                console.log('Received image data:', response.data);
                this.galleryImages = response.data.images || [];
                console.log('Assigned galleryImages:', this.galleryImages);
            } catch (error) {
                console.error('Failed to load images:', error);
                this.galleryImages = [];
            }
        },

        async handleDrop(e) {
            e.preventDefault();
            const files = [...e.dataTransfer.files];
            await this.uploadFiles(files);
        },
        async handleFileSelect(e) {
            const files = [...e.target.files];
            await this.uploadFiles(files);
        },
        async uploadFiles(files) {
            if (!files.length) return;

            this.uploading = true;
            this.uploadProgress = 0;
            const formData = new FormData();
            files.forEach(file => {
                formData.append('files[]', file);
            });

            try {
                await axios.post('http://localhost:5000/api/upload', formData, {
                    onUploadProgress: (e) => {
                        this.uploadProgress = Math.round((e.loaded * 100) / e.total);
                    }
                });
                alert('Upload successful!');
                await this.loadAllImages();
            } catch (error) {
                console.error('Upload failed:', error);
                alert('Upload failed. Please try again.');
            } finally {
                this.uploading = false;
            }
        },

        getThumbnailUrl(path) {
            const filename = path.split('/').pop();
            return `http://localhost:5000/thumbnails/${filename}`;
        }
    }
}).mount('#app') 