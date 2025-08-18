import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { searchForWorkspaceRoot } from 'vite';
import fs from 'fs';
import path from 'path';

const wasmMiddleware = () => {
  return {
    name: 'wasm-middleware',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url.endsWith('.wasm')) {
          // Path ke file WASM di node_modules/onnxruntime-web/dist/
          const wasmPath = path.join(process.cwd(), 'node_modules/onnxruntime-web/dist', path.basename(req.url));
          if (fs.existsSync(wasmPath)) {
            const wasmFile = fs.readFileSync(wasmPath);
            res.setHeader('Content-Type', 'application/wasm');
            res.setHeader('Access-Control-Allow-Origin', '*'); // Opsional, jika ada isu CORS
            res.end(wasmFile);
            return;
          }
        }
        next();
      });
    },
  };
};

export default defineConfig({
  plugins: [react(), wasmMiddleware()],
  optimizeDeps: {
    include: ['onnxruntime-web'],
  },
  build: {
    target: 'esnext',
    assetsInclude: ['**/*.onnx', '**/*.wasm'], // Tambah .wasm untuk build
  },
  server: {
    fs: {
      allow: [searchForWorkspaceRoot(process.cwd())],
    },
  },
});