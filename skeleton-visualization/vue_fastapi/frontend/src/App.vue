<template>
  <div id="app">
    <PrimeVueThreeJS  />
  </div>
</template>

<script setup>
import PrimeVueThreeJS from './components/primevue/ThreeJSCard.vue';
</script>

<style>
body {
  font-family: var(--font-family);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app {
  padding: 2rem;
}
</style>