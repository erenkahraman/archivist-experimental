<template>
  <div class="delete-modal" @click="$emit('cancel')">
    <div class="delete-modal-content" @click.stop>
      <h3>Delete Image</h3>
      <p>Are you sure you want to delete this image?</p>
      <div class="delete-modal-actions">
        <button class="cancel-button" @click="$emit('cancel')">Cancel</button>
        <button class="confirm-button" @click="$emit('confirm')">Delete</button>
      </div>
    </div>
  </div>
</template>

<script setup>
defineEmits(['cancel', 'confirm'])
</script>

<style scoped>
.delete-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.delete-modal-content {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  max-width: 400px;
  width: 90%;
}

.delete-modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  margin-top: 1.5rem;
}

.cancel-button {
  padding: 0.5rem 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  cursor: pointer;
}

.confirm-button {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  background: #dc3545;
  color: white;
  cursor: pointer;
}

.confirm-button:hover {
  background: #c82333;
}
</style> 