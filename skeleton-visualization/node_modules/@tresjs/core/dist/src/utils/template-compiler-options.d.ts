declare const templateCompilerOptions: {
    template: {
        compilerOptions: {
            isCustomElement: (tag: string) => boolean;
        };
    };
};
export default templateCompilerOptions;
