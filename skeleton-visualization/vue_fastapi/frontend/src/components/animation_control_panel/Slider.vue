<!--<template>-->
<!--  <div ref="controlContainer" class="control-container">-->
<!--    <input-->
<!--        ref="frameSlider"-->
<!--        type="range"-->
<!--        min="0"-->
<!--        v-model="currentFrameNumber"-->
<!--        :max="maxFrames"-->
<!--        class="slider"-->
<!--    />-->
<!--    <span class="frame-label"> Frame: {{ currentFrameNumber }} </span>-->
<!--  </div>-->
<!--</template>-->

<!--<script setup>-->
<!--import { ref, onMounted} from 'vue';-->
<!--const frameSlider = ref(null);-->
<!--const currentFrameNumber = ref(null);-->
<!--const maxFrames = ref(0);  // Initial max value-->

<!--onMounted(() => {-->
<!--});-->
<!--</script>-->

<!--<style scoped>-->
<!--.control-container {-->
<!--  width: 100%;-->
<!--  display: flex;-->
<!--  justify-content: center;-->
<!--  align-items: center;-->
<!--  padding: 10px 0;-->
<!--  background-color: #fefefe;-->
<!--}-->
<!--.slider {-->
<!--  width: 70%;-->
<!--  margin: 0 auto;-->
<!--}-->

<!--.frame-label {-->
<!--  color: black;-->
<!--}-->
<!--</style>-->


<template>
  <div class="slider-container">
    <Slider
        id="frame-slider"
        ref="frameSlider"
        v-model="currentFrameNumber"
        :min="0"
        :max="100"
        class="w-full"
    />
    <label for="frame-slider">Frame:</label>
    <span class="slider-value">{{ currentFrameNumber }}</span>
  </div>
</template>

<script setup>
import Slider from 'primevue/slider';
import { ref } from 'vue';
const currentFrameNumber = ref(0);

</script>

<style scoped>
.slider-container {
  display: flex;
  flex-direction: row;
  gap: 0.5rem;
  padding: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.slider-value {
  text-align: center;
  font-weight: bold;
  color: black;
}
</style>