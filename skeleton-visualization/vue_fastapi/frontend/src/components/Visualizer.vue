<script setup>
import ThreeJS from './Skeleton3DPlot.vue';
import Slider from './animation_control_panel/Slider.vue';
</script>

<template>
  <div class="main-container">
    <ThreeJS />
    <Slider />
  </div>
</template>

<style scoped>
.main-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 20px;
}
</style>
