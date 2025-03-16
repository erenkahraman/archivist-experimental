<template>
  <div class="color-analysis">
    <section class="analysis-section">
      <h4>Color Palette</h4>
      <div class="color-grid">
        <div 
          v-for="color in image.colors?.dominant_colors" 
          :key="color.hex"
          class="color-item"
        >
          <div 
            class="color-preview" 
            :style="{ backgroundColor: color.hex }"
          ></div>
          <div class="color-info">
            <span class="color-name">{{ color.name }}</span>
            <span class="color-percentage">
              {{ (color.proportion * 100).toFixed(1) }}%
            </span>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
defineProps({
  image: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.color-analysis {
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

.color-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
}

.color-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 8px;
  position: relative;
}

.color-info {
  display: flex;
  align-items: center;
  flex: 1;
  position: relative;
  justify-content: space-between;
  padding-right: 10px;
}

.color-name {
  margin-right: 40px;
}

.color-percentage {
  color: #000000;
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
}

.color-preview {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

@media (max-width: 768px) {
  .color-grid {
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  }
}
</style> 