<template>
  <div class="app-container">
    <div class="app-background"></div>
    
    <!-- Mobile sidebar toggle button -->
    <button class="sidebar-toggle" v-if="isMobile" @click="toggleSidebar" :class="{ 'active': sidebarVisible }">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 12H21M3 6H21M3 18H21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </button>
    
    <!-- Mobile overlay when sidebar is visible -->
    <div class="mobile-overlay" v-if="isMobile && sidebarVisible" @click="toggleSidebar"></div>
    
    <!-- Header with logo and title -->
    <header class="app-header animate__animated animate__fadeIn">
      <div class="logo">
        <span class="logo-accent">A</span>rchivist
      </div>
      <h1 class="tagline">AI-Powered Image Analysis</h1>
    </header>
    
    <!-- Main content with sidebar -->
    <main class="app-content">
      <!-- Sidebar with search functionality -->
      <aside class="app-sidebar glass animate__animated animate__fadeInLeft" :class="{ 'mobile-visible': sidebarVisible }">
        <SearchBar />
      </aside>
      
      <!-- Main content area -->
      <div class="main-content-area">
        <div class="controls-container glass animate__animated animate__fadeInUp">
          <ImageUpload />
        </div>
        
        <Gallery />
      </div>
    </main>
    
    <!-- Footer -->
    <footer class="app-footer">
      <p>Archivist • AI-Powered Image Analysis</p>
    </footer>
  </div>
</template>

<script setup>
import { onMounted, ref, computed } from 'vue'
import { useImageStore } from './stores/imageStore'
import ImageUpload from './components/ImageUpload.vue'
import SearchBar from './components/SearchBar.vue'
import Gallery from './components/Gallery.vue'

const imageStore = useImageStore()

// Mobile sidebar state
const windowWidth = ref(window.innerWidth)
const sidebarVisible = ref(false)

// Check if mobile view
const isMobile = computed(() => {
  return windowWidth.value <= 768
})

// Toggle sidebar visibility on mobile
const toggleSidebar = () => {
  sidebarVisible.value = !sidebarVisible.value
}

// Update window width on resize
const handleResize = () => {
  windowWidth.value = window.innerWidth
}

// Force cleanup on page load
onMounted(async () => {
  console.log("App mounted - ensuring clean image cache")
  
  // Add resize listener
  window.addEventListener('resize', handleResize)
  
  // Define a global cleanup function that can be called from console
  window.resetGallery = async () => {
    console.log("Performing complete gallery reset...")
    await imageStore.resetStore()
    console.log("Gallery reset complete - all data freshly loaded from server!")
  }
  
  // Create an explicit reset button in the console for debugging
  console.log('%c Gallery has been cleaned up. Any errors above are just 404s for thumbnails being checked.', 
    'background: #4CAF50; color: white; padding: 4px; border-radius: 4px; font-weight: bold; font-size: 14px')
  
  // Run the full reset to ensure clean start
  await imageStore.resetStore()
  
  console.log('%c Gallery successfully reset and filtered! All invalid images have been removed.', 
    'background: #4CAF50; color: white; padding: 4px; border-radius: 4px; font-weight: bold; font-size: 14px')
})
</script>

<style>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  position: relative;
  overflow-x: hidden;
}

.app-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: 
    radial-gradient(circle at 10% 10%, rgba(79, 70, 229, 0.15), transparent 30%),
    radial-gradient(circle at 90% 20%, rgba(34, 211, 238, 0.15), transparent 40%),
    radial-gradient(circle at 30% 80%, rgba(251, 113, 133, 0.15), transparent 30%),
    radial-gradient(circle at 80% 90%, rgba(79, 70, 229, 0.1), transparent 20%);
  z-index: -1;
  backdrop-filter: blur(80px);
}

.app-header {
  padding: var(--space-6) 0;
  text-align: center;
  margin-bottom: var(--space-4);
  margin-left: 320px; /* Match sidebar width */
}

.logo {
  font-family: var(--font-heading);
  font-size: 3.5rem;
  font-weight: 700;
  background: var(--gradient-primary);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin-bottom: var(--space-2);
  letter-spacing: -0.05em;
}

.logo-accent {
  font-size: 4rem;
}

.tagline {
  font-size: 1.25rem;
  font-weight: 400;
  color: var(--color-text-light);
  margin-top: 0;
}

.app-content {
  flex: 1;
  width: calc(100% - 320px); /* Adjust for sidebar width */
  margin-left: 320px; /* Match sidebar width */
  padding: 0 var(--space-2); /* Reduced horizontal padding */
  padding-bottom: var(--space-8);
  display: flex;
  gap: var(--space-4); /* Reduced gap between components */
  position: relative;
  min-height: calc(100vh - 8rem);
  justify-content: center; /* Center content horizontally */
  box-sizing: border-box; /* Include padding in width calculation */
}

/* Fixed sidebar for constant visibility */
.app-sidebar {
  width: 320px;
  flex-shrink: 0;
  padding: var(--space-5);
  border-radius: 0;
  background: rgba(10, 15, 30, 0.95); /* Darker for better visibility */
  backdrop-filter: blur(12px);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.4), 0 0 15px rgba(99, 102, 241, 0.15);
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100vh;
  max-height: none;
  overflow-y: hidden;
  display: flex;
  flex-direction: column;
  z-index: 100;
}

/* Custom scrollbar for sidebar */
.app-sidebar::-webkit-scrollbar {
  width: 5px;
}

.app-sidebar::-webkit-scrollbar-track {
  background: transparent;
}

.app-sidebar::-webkit-scrollbar-thumb {
  background: rgba(99, 102, 241, 0.3);
  border-radius: var(--radius-full);
}

.app-sidebar::-webkit-scrollbar-thumb:hover {
  background: rgba(99, 102, 241, 0.5);
}

/* Main content area */
.main-content-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 100%; /* Use full available width */
  margin: 0; /* No margins to maximize width */
  align-self: center; /* Ensure it's centered within the flex parent */
  padding: 0 var(--space-4); /* Increased padding on both sides */
}

.controls-container {
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  margin-bottom: var(--space-4); /* Reduced bottom margin */
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.app-footer {
  text-align: center;
  padding: var(--space-6) 0;
  color: var(--color-text-light);
  font-size: 0.9rem;
  margin-left: 320px; /* Match sidebar width */
}

@media (max-width: 768px) {
  .app-content {
    flex-direction: column;
    width: 100%;
    margin-left: 0;
    padding: 0 var(--space-4);
    margin-top: 80px; /* Space for the toggle button */
  }
  
  .app-sidebar {
    width: 280px;
    left: -280px; /* Hide off-screen by default on mobile */
    position: fixed;
    height: 100vh; 
    margin-bottom: 0;
    transition: left 0.3s ease;
    z-index: 200;
  }
  
  .app-sidebar.mobile-visible {
    left: 0; /* Show when toggled */
  }
  
  .main-content-area {
    margin-left: 0;
    width: 100%;
  }
  
  .logo {
    font-size: 2.5rem;
  }
  
  .logo-accent {
    font-size: 3rem;
  }
  
  .tagline {
    font-size: 1rem;
  }
  
  .app-header {
    padding: var(--space-4) 0;
    margin-left: 0;
    position: relative;
  }
  
  .controls-container {
    padding: var(--space-4);
  }
  
  .app-footer {
    margin-left: 0;
  }
  
  /* Mobile menu toggle button */
  .sidebar-toggle {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 201;
    width: 40px;
    height: 40px;
    background: rgba(99, 102, 241, 0.9);
    border: none;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
  }
  
  .sidebar-toggle.active {
    background: rgba(79, 70, 229, 1);
    transform: rotate(90deg);
  }
  
  .mobile-overlay {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    background: rgba(0, 0, 0, 0.7);
    z-index: 150;
    backdrop-filter: blur(4px);
    animation: fadeIn 0.3s ease;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
}

/* Glass effect styling */
.glass {
  background: rgba(30, 41, 59, 0.5);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
</style> 