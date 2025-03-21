<template>
  <div class="pattern-analysis">
    <section class="analysis-section">
      <h4>Primary Pattern</h4>
      <div class="pattern-chips">
        <div class="pattern-chip primary">
          {{ primaryPattern }}
        </div>
      </div>
    </section>

    <section class="analysis-section">
      <h4>Secondary Patterns</h4>
      <div class="pattern-chips">
        <div 
          v-for="pattern in secondaryPatterns" 
          :key="pattern.name"
          class="pattern-chip"
        >
          <span>{{ pattern.name }}</span>
          <span class="confidence-badge">
            {{ (pattern.confidence * 100).toFixed(0) }}%
          </span>
        </div>
      </div>
    </section>

    <section class="analysis-section">
      <h4>Pattern Description</h4>
      <p>{{ patternDescription }}</p>
    </section>
  </div>
</template>

<script setup>
const props = defineProps({
  primaryPattern: {
    type: String,
    default: 'Unknown'
  },
  secondaryPatterns: {
    type: Array,
    default: () => []
  },
  patternDescription: {
    type: String,
    default: 'No description available'
  }
})
</script>

<style scoped>
.pattern-analysis {
  padding: 1rem;
}

.analysis-section {
  margin-bottom: 2rem;
}

.analysis-section h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.pattern-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.pattern-chip {
  padding: 0.5rem 1rem;
  background: #f1f1f1;
  border-radius: 16px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.pattern-chip.primary {
  background: #4CAF50;
  color: white;
}

.confidence-badge {
  background: rgba(0,0,0,0.1);
  padding: 0.2rem 0.5rem;
  border-radius: 10px;
  font-size: 0.8rem;
}
</style> 