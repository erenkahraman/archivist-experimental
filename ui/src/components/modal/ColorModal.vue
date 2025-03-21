<template>
  <section class="analysis-section">
    <h4>Color Palette</h4>
    <div class="color-grid">
      <div 
        v-for="color in colors" 
        :key="color.hex"
        class="color-item"
      >
        <div class="color-percentage">
          {{ (color.proportion * 100).toFixed(1) }}%
        </div>
        <div 
          class="color-preview" 
          :style="{ backgroundColor: color.hex }"
        ></div>
        <div class="color-info">
          <div class="color-name">{{ color.name }}</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const props = defineProps({
  colors: {
    type: Array,
    default: () => []
  }
})
</script>

<style scoped>
.color-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
}

.color-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 8px;
  position: relative;
}

.color-info {
  display: flex;
  flex-direction: column;
  width: 100%;
  position: relative;
}

.color-name {
  font-size: 0.9rem;
  margin-bottom: 0.25rem;
  word-wrap: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
  line-height: 1.3;
}

.color-percentage {
  position: absolute;
  bottom: 8px;
  right: 8px;
  background-color: rgba(255, 255, 255, 0.85);
  color: #333;
  font-size: 0.75rem;
  padding: 3px 8px;
  border-radius: 12px;
  z-index: 2;
  font-weight: 500;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.color-preview {
  width: 100%;
  height: 40px;
  border-radius: 8px;
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.analysis-section {
  margin-bottom: 2rem;
}

.analysis-section h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}
</style> 