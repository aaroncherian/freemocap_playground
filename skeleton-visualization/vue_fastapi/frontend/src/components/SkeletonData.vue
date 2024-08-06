<template>
  <div>
    <h1>Skeleton Data</h1>
    <pre>{{ skeletonData }}</pre>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import api from '../services/api';

export default {
  name: 'SkeletonData',
  setup() {
    const skeletonData = ref(null);

    onMounted(() => {
      api.getData().then(response => {
        skeletonData.value = response.data;
      }).catch(error => {
        console.error('Error fetching skeleton data:', error);
      });
    });

    return { skeletonData };
  }
};
</script>

<style scoped>
pre {
  background-color: #000000;
  color: #ffffff;
  padding: 10px;
  border-radius: 5px;
}
</style>
