import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/upload",
        destination: "https://profit-sentinel-saas-backend.vercel.app/upload",
      },
      {
        source: "/health",
        destination: "https://profit-sentinel-saas-backend.vercel.app/health",
      },
    ];
  },

  reactStrictMode: true,
  poweredByHeader: false,
  images: {
    domains: [],
  },
};

export default nextConfig;