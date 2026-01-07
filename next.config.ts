import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Proxy /upload (and optionally /health) to your deployed FastAPI backend on Vercel
  // This enables relative fetch('/upload') from your frontend index.html
  // - Browser sees same-origin → no CORS, no subdomain SSL/TLS errors
  // - Vercel handles the proxy server-side seamlessly (supports file uploads via FormData)
  async rewrites() {
    return [
      {
        source: "/upload",
        destination: "https://profit-sentinel-saas-backend.vercel.app/upload",
      },
      // Optional: Proxy the health endpoint for easy testing
      {
        source: "/health",
        destination: "https://profit-sentinel-saas-backend.vercel.app/health",
      },
    ];
  },

  // Recommended production settings
  reactStrictMode: true,
  poweredByHeader: false, // Security: hide Next.js header

  // Image optimization (add external domains if you load images from Supabase/S3 later)
  images: {
    domains: [],
  },
};

export default nextConfig;