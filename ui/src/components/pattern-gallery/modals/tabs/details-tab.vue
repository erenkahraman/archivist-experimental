<template>
  <div class="details-analysis">
    <!-- Pattern Analysis -->
    <section class="analysis-section">
      <h4>Pattern Characteristics</h4>
      <div class="characteristics-grid">
        <div class="characteristic-item">
          <span class="label">Layout:</span>
          <span>{{ image.patterns?.layout?.type }}</span>
          <div class="confidence-bar" :style="{ width: `${image.patterns?.layout?.confidence * 100}%` }"></div>
        </div>
        <div class="characteristic-item">
          <span class="label">Scale:</span>
          <span>{{ image.patterns?.scale?.type }}</span>
          <div class="confidence-bar" :style="{ width: `${image.patterns?.scale?.confidence * 100}%` }"></div>
        </div>
        <div class="characteristic-item">
          <span class="label">Texture:</span>
          <span>{{ image.patterns?.texture_type?.type }}</span>
          <div class="confidence-bar" :style="{ width: `${image.patterns?.texture_type?.confidence * 100}%` }"></div>
        </div>
      </div>
    </section>

    <!-- Design Information -->
    <section class="analysis-section">
      <h4>Design Information</h4>
      <div class="design-info-grid">
        <div class="info-item">
          <span class="label">Dimensions:</span>
          <span>{{ image.dimensions?.width }}x{{ image.dimensions?.height }}px</span>
        </div>
        <div class="info-item">
          <span class="label">Aspect Ratio:</span>
          <span>{{ calculateAspectRatio(image.dimensions) }}</span>
        </div>
        <div class="info-item">
          <span class="label">File Name:</span>
          <span>{{ getImageName(image.original_path) }}</span>
        </div>
      </div>
    </section>

    <!-- Color Analysis -->
    <section class="analysis-section">
      <h4>Color Analysis</h4>
      <div class="color-analysis-grid">
        <!-- Color Distribution -->
        <div class="color-stats">
          <h5>Color Distribution</h5>
          <div class="color-bars">
            <div v-for="color in image.colors?.dominant_colors" :key="color.hex" class="color-bar-item">
              <div class="color-bar-label">
                <span class="color-preview" :style="{ backgroundColor: color.hex }"></span>
                <span>{{ color.name }}</span>
              </div>
              <div class="color-bar-container">
                <div class="color-bar-fill" 
                     :style="{ width: `${color.proportion * 100}%`, backgroundColor: color.hex }">
                </div>
                <span class="color-percentage">{{ (color.proportion * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { calculateAspectRatio, getImageName } from '../../utils/helpers'

defineProps({
  image: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.details-analysis {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.analysis-section {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.analysis-section h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.characteristics-grid,
.design-info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.characteristic-item,
.info-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  position: relative;
}

.label {
  font-weight: 500;
  color: #666;
  margin-right: 0.5rem;
}

.confidence-bar {
  height: 4px;
  background-color: #4CAF50;
  border-radius: 2px;
  margin-top: 0.5rem;
}

.color-analysis-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
}

.color-stats {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.color-bars {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.color-bar-item {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.color-bar-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.color-preview {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  border: 1px solid rgba(0,0,0,0.1);
}

.color-bar-container {
  height: 8px;
  background: #eee;
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.color-bar-fill {
  height: 100%;
  border-radius: 4px;
}

.color-percentage {
  position: absolute;
  right: 0;
  top: -18px;
  font-size: 0.8rem;
  color: #666;
}

h5 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-size: 1rem;
}

@media (max-width: 768px) {
  .characteristics-grid,
  .design-info-grid,
  .style-grid {
    grid-template-columns: 1fr;
  }
}
</style> 