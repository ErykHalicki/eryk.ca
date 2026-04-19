import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();

const PIXEL_RES = 150;

function initViewer(container) {
    const url = container.dataset.model;
    const spinSpeed = parseFloat(container.dataset.spinSpeed ?? '0.6');
    const pixelRes = parseInt(container.dataset.pixelRes ?? PIXEL_RES, 10);
    const scaleMul = parseFloat(container.dataset.scale ?? '1');
    const yOffset = parseFloat(container.dataset.yOffset ?? '0');

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.set(0, 0, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setPixelRatio(1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(3, 4, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xc4a7ee, 0.4);
    fill.position.set(-3, 2, -2);
    scene.add(fill);

    const group = new THREE.Group();
    group.position.y = yOffset;
    scene.add(group);
    const pivot = new THREE.Group();
    group.add(pivot);

    loader.load(
        url,
        (gltf) => {
            const model = gltf.scene;

            model.traverse((child) => {
                if (!child.isMesh || !child.material) return;
                const mats = Array.isArray(child.material) ? child.material : [child.material];
                for (const mat of mats) {
                    for (const key of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap', 'aoMap']) {
                        const tex = mat[key];
                        if (!tex) continue;
                        tex.magFilter = THREE.NearestFilter;
                        tex.minFilter = THREE.NearestFilter;
                        tex.generateMipmaps = false;
                        tex.anisotropy = 1;
                        tex.needsUpdate = true;
                    }
                }
            });

            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            model.position.sub(center);
            pivot.add(model);
            const maxDim = Math.max(size.x, size.y, size.z) || 1;
            pivot.scale.setScalar((2 / maxDim) * scaleMul);

            const fitDist = 1 / Math.tan((camera.fov * Math.PI) / 360);
            camera.position.set(0, 0, fitDist * 1.3);
            camera.updateProjectionMatrix();
        },
        undefined,
        (err) => {
            console.warn(`Failed to load ${url}:`, err);
        }
    );

    renderer.setSize(pixelRes, pixelRes, false);
    camera.aspect = 1;
    camera.updateProjectionMatrix();

    Object.assign(renderer.domElement.style, {
        imageRendering: 'pixelated',
        width: '100%',
        height: '100%'
    });

    const clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate);
        group.rotation.y += clock.getDelta() * spinSpeed;
        renderer.render(scene, camera);
    }
    animate();
}

document.querySelectorAll('.model-viewer').forEach(initViewer);
