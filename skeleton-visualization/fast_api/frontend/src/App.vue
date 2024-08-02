<template>
  <div id="app">
    <SkeletonData />
    <Test />
  </div>
</template>

<script>
import SkeletonData from './components/SkeletonData.vue';
import Test from './components/plot_3d.vue';

export default {
  name: 'App',
  components: {
    SkeletonData,
    Test
  }
};
</script>

<style>
/* Add some global styles if needed */
</style>
