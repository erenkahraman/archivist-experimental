<template>
  <div class="app-container">
    <div class="app-background"></div>
    
    <!-- Header with logo and title -->
    <header class="app-header animate__animated animate__fadeIn">
      <div class="logo">
        <span class="logo-accent">A</span>rchivist
      </div>
      <h1 class="tagline">AI-Powered Image Analysis</h1>
    </header>
    
    <!-- Main content -->
    <main class="app-content">
      <div class="controls-container glass animate__animated animate__fadeInUp">
        <ImageUpload />
        <SearchBar />
      </div>
      
      <Gallery />
    </main>
    
    <!-- Footer -->
    <footer class="app-footer">
      <p>Archivist • AI-Powered Image Analysis</p>
    </footer>
  </div>
</template>

<script setup>
import ImageUpload from './components/ImageUpload.vue'
import SearchBar from './components/SearchBar.vue'
import Gallery from './components/Gallery.vue'
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
    radial-gradient(circle at 10% 10%, rgba(79, 70, 229, 0.1), transparent 30%),
    radial-gradient(circle at 90% 20%, rgba(34, 211, 238, 0.1), transparent 40%),
    radial-gradient(circle at 30% 80%, rgba(251, 113, 133, 0.1), transparent 30%),
    radial-gradient(circle at 80% 90%, rgba(79, 70, 229, 0.05), transparent 20%);
  z-index: -1;
}

.app-header {
  padding: var(--space-8) 0;
  text-align: center;
  margin-bottom: var(--space-6);
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
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 var(--space-4);
}

.controls-container {
  padding: var(--space-6);
  border-radius: var(--radius-lg);
  margin-bottom: var(--space-8);
}

.app-footer {
  text-align: center;
  padding: var(--space-6) 0;
  color: var(--color-text-light);
  font-size: 0.9rem;
}

@media (max-width: 768px) {
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
    padding: var(--space-6) 0;
  }
  
  .controls-container {
    padding: var(--space-4);
  }
}
</style> 