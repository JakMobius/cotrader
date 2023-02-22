import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';
import json from '@rollup/plugin-json'
import alias from '@rollup/plugin-alias'
import styles from "rollup-plugin-styler";
import replace from '@rollup/plugin-replace';
import copy from 'rollup-plugin-copy'

import path from 'path'
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const production = !!process.env.PRODUCTION;
const projectRootDir = path.resolve(__dirname);

export default [{
    input: "src/index.ts",
    output: {
        file: "dist/script.js",
        format: 'iife',
        sourcemap: true,
        assetFileNames: '[name][extname]'
    },
    plugins: [
        alias({
            entries: [
                {
                    find: 'src',
                    replacement: path.resolve(projectRootDir, 'src')
                }
            ]
        }),
        json(),
        resolve({
            extensions: ['.ts', '.tsx', '.js', '.jsx', '.json'],
            browser: true,
            preferBuiltins: true
        }),
        typescript({
            jsx: 'react',
            tsconfig: './tsconfig.json',
        }),
        commonjs({
            include: /node_modules/
        }),
        replace({
            preventAssignment: false,
            'process.env.NODE_ENV': production ? '"production"' : '"development"'
        }),
        production && terser(),
        styles({
            mode: ["extract", "style.css"]
        }),
        copy({
            targets: [
                { src: "src/static/*", dest: "dist" },
            ]
        })
    ]
}];
