<template>
  <div class="card">
    <div class="mt-3" style="flex-grow: 1;">
      <ThreeJSScene :width="width" :height="height" />
    </div>
  </div>
</template>

<script setup>
import ThreeJSScene from '../Skeleton3DPlot.vue';

// Set width and height as strings
const width = '100%';
const height = '100%';
</script>

<style scoped>
.card {
  display: flex;
  flex-direction: column;
  height: 100%; /* Ensure the card takes full height */
  width: 100%; /* Ensure the card takes full width */
  padding: 1rem;
  border-radius: 15px;
  background-color: #ffffff;
  margin-top: 40px;
  justify-content: center; /* Center content vertically */

}


</style>