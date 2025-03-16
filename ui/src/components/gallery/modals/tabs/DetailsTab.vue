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

        <!-- RGB Values -->
        <div class="rgb-values" v-if="image.colors?.dominant_colors">
          <h5>RGB Values</h5>
          <div class="rgb-grid">
            <div v-for="color in image.colors.dominant_colors" :key="color.hex" class="rgb-item">
              <div class="color-preview" :style="{ backgroundColor: color.hex }"></div>
              <div class="rgb-info">
                <span>R: {{ color.rgb[0] }}</span>
                <span>G: {{ color.rgb[1] }}</span>
                <span>B: {{ color.rgb[2] }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Style Analysis -->
    <section class="analysis-section">
      <h4>Style Analysis</h4>
      <div class="style-grid">
        <div class="style-item">
          <span class="label">Cultural Influence:</span>
          <span>{{ image.patterns?.cultural_influence?.type }}</span>
          <div class="confidence-bar" 
               :style="{ width: `${image.patterns?.cultural_influence?.confidence * 100}%` }">
          </div>
        </div>
        <div class="style-item">
          <span class="label">Historical Period:</span>
          <span>{{ image.patterns?.historical_period?.type }}</span>
          <div class="confidence-bar" 
               :style="{ width: `${image.patterns?.historical_period?.confidence * 100}%` }">
          </div>
        </div>
        <div class="style-item">
          <span class="label">Mood:</span>
          <span>{{ image.patterns?.mood?.type }}</span>
          <div class="confidence-bar" 
               :style="{ width: `${image.patterns?.mood?.confidence * 100}%` }">
          </div>
        </div>
        <div class="style-item">
          <span class="label">Style Keywords:</span>
          <div class="keyword-chips">
            <span v-for="keyword in image.patterns?.style_keywords" 
                  :key="keyword" 
                  class="keyword-chip">
              {{ keyword }}
            </span>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { getImageName, calculateAspectRatio } from '../../utils/imageHelpers'

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

.characteristics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.characteristic-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.characteristic-item .label {
  color: #666;
  font-size: 0.9rem;
  display: block;
  margin-bottom: 0.3rem;
}

.confidence-bar {
  height: 4px;
  background: #4CAF50;
  border-radius: 2px;
  margin-top: 4px;
  transition: width 0.3s ease;
}

.design-info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.info-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.color-analysis-grid {
  display: grid;
  gap: 2rem;
}

.color-bars {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.color-bar-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.color-bar-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.color-bar-container {
  height: 24px;
  background: #f1f1f1;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}

.color-bar-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.color-percentage {
  color: #000000;
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
}

.rgb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 1rem;
}

.rgb-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.rgb-info {
  display: flex;
  flex-direction: column;
  font-size: 0.8rem;
  color: #666;
}

.color-preview {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.style-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.style-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.keyword-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.keyword-chip {
  padding: 0.3rem 0.8rem;
  background: #f1f1f1;
  border-radius: 16px;
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