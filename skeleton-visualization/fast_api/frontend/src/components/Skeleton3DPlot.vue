<template>
  <div ref="container" class="threejs-container"></div>
</template>

<script setup>
import {ref, onMounted, onBeforeUnmount} from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import api from '../services/api';


const container = ref(null);
let skeletonData = ref([]);

onMounted(() => {


  const fetchData = async () => {
    try {
      const response = await api.getData();
      skeletonData = response.data;
      visualizeData(100)
    } catch (error) {
      console.error('Error fetching skeleton data', error);
    }
  }

  const visualizeData = (frame) => {
    console.log('Visualizing data for frame:', frame);
    sphereDataGroup.clear();

    const defaultSphereGeometry = new THREE.SphereGeometry(2, 16, 16);
    const selectedSphereGeometry = new THREE.SphereGeometry(2.5, 16, 16);
    const defaultSphereMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const selectedSphereMaterial = new THREE.MeshBasicMaterial({ color: 0x009aa6 });

    for (const markerName in skeletonData.trajectories) {
      const markerData = skeletonData.trajectories[markerName];
      if (markerData && markerData[frame]) {
        const sphereMaterial = defaultSphereMaterial;
        const sphereGeometry = defaultSphereGeometry;
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.set(markerData[frame][0] / 10, markerData[frame][1] / 10, markerData[frame][2] / 10);
        sphere.name = markerName;
        sphereDataGroup.add(sphere);
      }
    }
  }


  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio); // Set the pixel ratio for better clarity
  container.value.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 250, 500);
  camera.lookAt(0, 0, 0);
  camera.up.set(0, 0, 1);

  const controls = new OrbitControls(camera, renderer.domElement);



  const sphereDataGroup = new THREE.Group();
  scene.add(sphereDataGroup);
  const grid_size = 500;
  const grid_divisions = 10;
  const gridHelper = new THREE.GridHelper(grid_size, grid_divisions);
  gridHelper.rotation.x = Math.PI / 2;
  const gridLine = new THREE.Line3(gridHelper.geometry.attributes.position.array[0], gridHelper.geometry.attributes.position.array[1]);
  scene.add(gridHelper);




  const animate = function () {
    requestAnimationFrame(animate);

    controls.update();
    renderer.render(scene, camera);
  };


  fetchData();
  animate();

  // Handle window resize
  const handleResize = () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };

  window.addEventListener('resize', handleResize);

  // Clean up on unmount
  onBeforeUnmount(() => {
    window.removeEventListener('resize', handleResize);
  });
});
</script>

<style scoped>
.threejs-container {
  width: 100%;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #000000;
}
</style>
