<template>
  <div class="card">
    <h5>Three.js Scene</h5>
    <div class="mt-3">
      <ThreeJSScene :width="width" :height="height" />
    </div>
  </div>
</template>

<script setup>
import ThreeJSScene from '../SimpleBox.vue';

// You can adjust these values directly here
const width = '400px';
const height = '100px';
</script>