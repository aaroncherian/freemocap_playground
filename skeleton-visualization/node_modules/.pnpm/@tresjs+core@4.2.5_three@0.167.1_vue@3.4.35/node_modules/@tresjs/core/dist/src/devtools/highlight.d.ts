import * as THREE from 'three';
export declare class HightlightMesh extends THREE.Mesh {
    type: string;
    createTime: number;
    constructor(...args: any[]);
    onBeforeRender(): void;
}
