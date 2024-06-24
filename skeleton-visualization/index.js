import * as THREE from 'three';

// Create scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add a simple light
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(0, 1, 1).normalize();
scene.add(light);

// Function to create a single frame of the skeleton
function createSkeletonFrame(data) {
    // Create material for joints and bones
    const jointMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
    const boneMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });

    // Create geometry for joints
    data.joints.forEach(joint => {
        const geometry = new THREE.SphereGeometry(5, 32, 32);
        const jointMesh = new THREE.Mesh(geometry, jointMaterial);
        jointMesh.position.set(joint.x, joint.y, joint.z);
        scene.add(jointMesh);
    });

    // Create geometry for bones
    data.bones.forEach(bone => {
        const points = [];
        points.push(new THREE.Vector3(bone.start.x, bone.start.y, bone.start.z));
        points.push(new THREE.Vector3(bone.end.x, bone.end.y, bone.end.z));
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, boneMaterial);
        scene.add(line);
    });
}

// Sample data for a single frame (replace with your data)
const sampleData = {
    joints: [
        { x: 0, y: 0, z: 0 },
        { x: 50, y: 50, z: 0 },
        { x: -50, y: 50, z: 0 }
    ],
    bones: [
        { start: { x: 0, y: 0, z: 0 }, end: { x: 50, y: 50, z: 0 } },
        { start: { x: 0, y: 0, z: 0 }, end: { x: -50, y: 50, z: 0 } }
    ]
};

createSkeletonFrame(sampleData);

camera.position.z = 200;

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
