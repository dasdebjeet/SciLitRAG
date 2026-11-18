
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Configure Vite's development server and React plugin.
export default defineConfig({
  plugins: [react()],

  server: {
    /*
     * Forward API requests to FastAPI.
     *
     * The browser calls localhost:5173/api/...
     * Vite forwards those requests to localhost:8000/api/...
     */
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },

      // Also proxy health-check requests.
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
