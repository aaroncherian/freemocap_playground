export declare const isProd: boolean;
type OneOrMore<T> = {
    0: T;
} & Array<T>;
interface LoggerComposition {
    logError: (...args: OneOrMore<any>) => void;
    logWarning: (...args: OneOrMore<any>) => void;
    logMessage: (name: string, value: any) => void;
}
export declare function useLogger(): LoggerComposition;
export {};
