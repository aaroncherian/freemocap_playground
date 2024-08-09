<script setup>
import SkeletonData from "@/components/SkeletonData.vue";
import Visualizer from './components/Visualizer.vue'
</script>

<template>
  <div id="app">
    <Visualizer />
  </div>
</template>

<style>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
}

#app {
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
}
</style>