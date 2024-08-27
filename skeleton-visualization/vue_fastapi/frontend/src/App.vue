<template>
  <div id="app" class="grid-container">
    <div class="grid-item">
      <PrimeVueThreeJS />
    </div>
  </div>
</template>

<script setup>
import PrimeVueThreeJS from './components/primevue/ThreeJSCard.vue';
</script>

<style>
@import 'primeflex/primeflex.css';

body {
  font-family: var(--font-family);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

html, body, #app {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
}

/* CSS Grid Layout */
.grid-container {
  display: grid;
  grid-template-columns: repeat(10, 1fr); /* Creates 10 equal-width columns */
  grid-template-rows: repeat(5, 1fr); /* Creates 5 equal-height rows */
  gap: 10px; /* Optional: Adds spacing between grid items */
  height: 100vh; /* Makes the grid container fill the viewport height */
  padding: 20px; /* Optional: Padding around the grid */
}

.grid-item {
  grid-column: 3 / span 6; /* Start at column 3 and span 6 columns */
  grid-row: span 3; /* Span 4 out of 5 rows */
  background-color: #1b1b1b; /* Just for visibility */
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
