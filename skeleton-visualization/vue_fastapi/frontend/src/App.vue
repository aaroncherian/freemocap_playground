<template>
  <div id="app">
    <div class = "grid">
      <div class = "col-10">
        <PrimeVueThreeJS />
      </div>
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

#app {

}
</style>

<!--<template>-->
<!--   <div id = "app">-->
<!--   </div>-->
<!--</template>-->

<!--<script setup>-->
<!--</script>-->

<!--<style>-->
<!--html, body, #app {-->
<!--  margin: 0;-->
<!--  padding: 0;-->
<!--  height: 100%;-->
<!--  width: 100%;-->
<!--}-->

<!--/* Optional: Add a background color to see the full width and height */-->
<!--#app {-->
<!--  background-color: lightblue;-->
<!--}-->

<!--</style>-->