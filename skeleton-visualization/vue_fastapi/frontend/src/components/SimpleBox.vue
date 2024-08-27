<template>
  <div ref="container" class="threejs-container"></div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const props = defineProps({
  width: {
    type: String,
    default: '100%'
  },
  height: {
    type: String,
    default: '100%'
  }
});

const container = ref(null);
let scene, camera, renderer, controls, cube;

onMounted(() => {
  setTimeout(() => {
    initScene();
    animate();
    onWindowResize();  // Trigger a resize to ensure the dimensions are correct
  }, 0);
  window.addEventListener('resize', onWindowResize);
});

function initScene() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75, container.value.clientWidth / container.value.clientHeight, 0.1, 1000);

  renderer = new THREE.WebGLRenderer();
  renderer.setSize(container.value.clientWidth, container.value.clientHeight);
  container.value.appendChild(renderer.domElement);

  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
  cube = new THREE.Mesh(geometry, material);
  scene.add(cube);

  const gridHelper = new THREE.GridHelper(10, 10);
  scene.add(gridHelper);

  camera.position.z = 5;

  controls = new OrbitControls(camera, renderer.domElement);
}

function animate() {
  requestAnimationFrame(animate);
  cube.rotation.x += 0.01;
  cube.rotation.y += 0.01;
  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  if (camera && renderer && container.value) {
    camera.aspect = container.value.clientWidth / container.value.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.value.clientWidth, container.value.clientHeight);
  }
}

watch(() => [props.width, props.height], () => {
  if (container.value) {
    container.value.style.width = props.width;
    container.value.style.height = props.height;
    onWindowResize();
  }
}, { immediate: true });
</script>

<style scoped>
.threejs-container {
  width: v-bind(width);
  height: v-bind(height);
}
</style>