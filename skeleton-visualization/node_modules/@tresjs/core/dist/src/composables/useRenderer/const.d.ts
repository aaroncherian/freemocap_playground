export declare const rendererPresets: {
    realistic: {
        shadows: boolean;
        physicallyCorrectLights: boolean;
        outputColorSpace: "srgb";
        toneMapping: 4;
        toneMappingExposure: number;
        shadowMap: {
            enabled: boolean;
            type: 2;
        };
    };
    flat: {
        toneMapping: 0;
        toneMappingExposure: number;
    };
};
export type RendererPresetsType = keyof typeof rendererPresets;
