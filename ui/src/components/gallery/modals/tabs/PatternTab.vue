<template>
  <div class="pattern-analysis">
    <section class="analysis-section">
      <h4>Primary Pattern</h4>
      <div class="pattern-chips">
        <div class="pattern-chip primary">
          {{ image.patterns?.primary_pattern }}
        </div>
      </div>
    </section>

    <section class="analysis-section">
      <h4>Secondary Patterns</h4>
      <div class="pattern-chips">
        <div 
          v-for="pattern in image.patterns?.secondary_patterns" 
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
      <p>{{ getPromptText(image.patterns?.prompt) }}</p>
    </section>
  </div>
</template>

<script setup>
import { getPromptText } from '../../utils/imageHelpers'

defineProps({
  image: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.pattern-analysis {
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