<template>
  <div ref="container" class="threejs-container"></div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import api from '../services/api';

const container = ref(null);
const currentFrameNumber = ref(null);
let skeletonData = ref([]);

onMounted(() => {
  const fetchData = async () => {
    try {
      const response = await api.getData();
      skeletonData.value = response.data;
      console.log('Skeleton data fetched: ', skeletonData.value);
      currentFrameNumber.value = 0;
    } catch (error) {
      console.error('Error fetching skeleton data', error);
    }
  };

  const visualizeData = (frame, data, group) => {
    console.log('Visualizing data for frame:', frame);
    clearDataGroup();

    plotSpheresAsJoints(frame, data.value.trajectories, group)
    plotLinesAsConnections(data.value.segments, group)
  };

  const clearDataGroup = () => {
    while (skeletonDataGroup.children.length > 0) {
      const child = skeletonDataGroup.children[0];
      skeletonDataGroup.remove(child);
      child.geometry.dispose();
      child.material.dispose();
    }
  };

  const plotSpheresAsJoints = (frame, trajectories, group) => {

    const defaultSphereGeometry = new THREE.SphereGeometry(2, 16, 16);
    const selectedSphereGeometry = new THREE.SphereGeometry(2.5, 16, 16);
    const defaultSphereMaterial = new THREE.MeshBasicMaterial({color: 0x000000});
    const selectedSphereMaterial = new THREE.MeshBasicMaterial({color: 0x009aa6});

    for (const markerName in trajectories) {
      const markerData = trajectories[markerName];
      if (markerData && markerData[frame]) {
        const sphereMaterial = defaultSphereMaterial;
        const sphereGeometry = defaultSphereGeometry;
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.set(markerData[frame][0] / 10, markerData[frame][1] / 10, markerData[frame][2] / 10);
        sphere.name = markerName;
        group.add(sphere);
      }
    }
  }

  const plotLinesAsConnections = (connections, group) => {
    const lineVertices = [];
    for (const [segmentName, segmentData] of Object.entries(connections)) {
      lineVertices.length = 0
      for (const [connectionPoint, markerName] of Object.entries(segmentData)) {
        lineVertices.push(group.getObjectByName(markerName).position.clone())
      }
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(lineVertices);
      const lineMaterial = new THREE.LineBasicMaterial({color: 0x000000});
      const lineObject = new THREE.Line(lineGeometry, lineMaterial);
      group.add(lineObject);
    }
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const renderer = new THREE.WebGLRenderer();


  renderer.setSize(container.value.clientWidth, container.value.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio); // Set the pixel ratio for better clarity
  container.value.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 250, 500);
  camera.lookAt(0, 0, 0);
  camera.up.set(0, 0, 1);

  const controls = new OrbitControls(camera, renderer.domElement);

  const skeletonDataGroup = new THREE.Group();
  scene.add(skeletonDataGroup);
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

  watch(currentFrameNumber, (newFrame) => {
    visualizeData(newFrame, skeletonData, skeletonDataGroup)
  });
  // Handle window resize
  const handleResize = () => {
    camera.aspect = container.value.clientWidth / container.value.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.value.clientWidth, container.value.clientHeight);
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
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

</style>
