import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  preview: { port: 5173 },
  build: { chunkSizeWarningLimit: 700 }, // recharts is most of the bundle; fine for a single-page dashboard
});
