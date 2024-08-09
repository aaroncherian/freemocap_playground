<template>
  <div ref="controlContainer" class="control-container">
    <input
        ref="frameSlider"
        type="range"
        min="0"
        v-model="currentFrameNumber"
        :max="maxFrames"
        class="slider"
    />
    <span class="frame-label"> Frame: {{ currentFrameNumber }} </span>
  </div>
</template>

<script setup>
import { ref, onMounted} from 'vue';
const frameSlider = ref(null);
const currentFrameNumber = ref(null);
const maxFrames = ref(0);  // Initial max value

onMounted(() => {
});
</script>

<style scoped>
.control-container {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 10px 0;
  background-color: #fefefe;
}
.slider {
  width: 70%;
  margin: 0 auto;
}

.frame-label {
  color: black;
}
</style>
