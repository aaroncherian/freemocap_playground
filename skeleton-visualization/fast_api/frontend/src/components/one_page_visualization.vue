<script setup>
import { ref, onMounted, watch } from 'vue';
import api from '../services/api';


import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import axios from 'axios';

const joints = ref([]);
const allFramesData = ref([]);
const graphContainer = ref(null);
const currentJoint = ref(null);
const segmentConnections = ref([]);
const jointNames = ref([]);
const currentFrame = ref(0);
const maxFrame = ref(0);
const fps = ref(30);
const playPauseText = ref('Play');
let isPlaying = false;
let frameInterval = 1000 / fps.value;
let animationId;

const dataGroup = new THREE.Group();
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
const controls = new OrbitControls(camera, renderer.domElement);

onMounted(() => {
  initThreeJS();
  fetchJointNames();
  fetchData();
});

const initThreeJS = () => {
  scene.background = new THREE.Color(0xffffff);
  camera.position.set(0, 250, 500);
  camera.lookAt(0, 0, 0);
  camera.up.set(0, 0, 1);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  controls.screenSpacePanning = false;
  controls.maxPolarAngle = Math.PI / 2;

  const ambientLight = new THREE.AmbientLight(0x404040);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);
  scene.add(dataGroup);

  const gridHelper = new THREE.GridHelper(500, 10);
  gridHelper.rotation.x = Math.PI / 2;
  scene.add(gridHelper);

  const axesHelper = new THREE.AxesHelper(5);
  scene.add(axesHelper);

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  graphContainer.value.appendChild(renderer.domElement);

  window.addEventListener('resize', onWindowResize);
};

const onWindowResize = () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
};

const fetchJointNames = async () => {
  try {
    const response = await api.getAvailableJointNames();
    joints.value = response.data.joint_names;
    currentJoint.value = joints.value[0];
  } catch (error) {
    console.error('Error fetching joint names:', error);
  }
};

const fetchData = async () => {
  try {
    const response = await api.getData();
    allFramesData.value = response.data;
    segmentConnections.value = response.data.segments;
    maxFrame.value = response.data.num_frames - 1;
    updateVisualization();
  } catch (error) {
    console.error('Error fetching data:', error);
  }
};

const togglePlayPause = () => {
  isPlaying = !isPlaying;
  playPauseText.value = isPlaying ? 'Pause' : 'Play';
  if (isPlaying) {
    animate();
  } else {
    cancelAnimationFrame(animationId);
  }
};

const updateVisualization = () => {
  visualizeData(currentFrame.value);
};

const visualizeData = (frame) => {
  dataGroup.clear();
  // Add visualization logic
};

const animate = (timestamp) => {
  animationId = requestAnimationFrame(animate);
  // Animation logic
  renderer.render(scene, camera);
  controls.update();
};
</script>


<template>
  <div>
    <div ref="graphContainer" id="graph-container"></div>
    <div id="control-container">
      <label for="joint-select">Select joint:</label>
      <select id="joint-select" v-model="currentJoint">
        <option v-for="joint in jointNames" :key="joint" :value="joint">{{ joint }}</option>
      </select>
      <button @click="togglePlayPause">{{ playPauseText }}</button>
      <input type="range" min="0" :max="maxFrame" v-model="currentFrame" @input="updateVisualization" class="slider" id="frame-slider">
      <span>Frame: {{ currentFrame }}</span>
      <label for="fps-input">FPS:</label>
      <input type="number" v-model.number="fps" min="1" style="width:60px;" id="fps-input">
    </div>
  </div>
</template>


<style scoped>
.threejs-container {
  width: 100%;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #ffffff;
}

#graph-container {
  flex: 0 0 50%;
  width: 50%;
  margin-left: auto;
  margin-right: auto;
  margin-top: 10px;
  background-color: #fff;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#control-container {
  flex: 0 0 8%;
  display: flex;
  align-items: center;
  justify-content: center;
}

#frame-slider {
  width: 70%;
  vertical-align: middle;
}

#play-pause {
  vertical-align: middle;
  margin-right: 10px;
}
</style>
