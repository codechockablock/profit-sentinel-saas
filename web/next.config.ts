import type { NextConfig } from "next";

// Determine API URL based on environment
// Production: https://api.profitsentinel.com
// Development: http://localhost:8000 (or from env var)
const getApiUrl = () => {
  // If explicitly set, use that
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  // In production (Vercel), default to production API
  if (process.env.VERCEL_ENV === "production" || process.env.NODE_ENV === "production") {
    return "https://api.profitsentinel.com";
  }
  // Development fallback
  return "http://localhost:8000";
};

const nextConfig: NextConfig = {
  // Enable React strict mode for better development experience
  reactStrictMode: true,

  // Environment variables that will be exposed to the browser
  env: {
    NEXT_PUBLIC_API_URL: getApiUrl(),
  },

  // Experimental features
  experimental: {
    // Enable server actions
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },

  // Image optimization
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'profitsentinel.com',
      },
      {
        protocol: 'https',
        hostname: 'i.imgur.com',
      },
    ],
  },

  // Redirects for legacy URLs
  async redirects() {
    return [
      {
        source: "/upload",
        destination: "/analyze",
        permanent: true,
      },
      {
        source: "/diagnostic",
        destination: "/analyze",
        permanent: false,
      },
    ];
  },

  // API rewrites to proxy backend requests
  async rewrites() {
    const apiUrl = getApiUrl();
    return [
      {
        source: "/api/diagnostic/:path*",
        destination: `${apiUrl}/diagnostic/:path*`,
      },
      {
        source: "/api/premium/:path*",
        destination: `${apiUrl}/premium/:path*`,
      },
      {
        source: "/api/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },

  // Headers for security - comprehensive OWASP-compliant headers
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          {
            key: "X-XSS-Protection",
            value: "1; mode=block",
          },
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains",
          },
          {
            key: "Permissions-Policy",
            value: "geolocation=(), microphone=(), camera=()",
          },
          {
            key: "Content-Security-Policy",
            value: "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://challenges.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' http://localhost:8000 https://api.profitsentinel.com https://*.supabase.co https://api.x.ai https://*.s3.amazonaws.com https://*.s3.us-east-1.amazonaws.com; frame-src 'self' https://challenges.cloudflare.com; frame-ancestors 'none';",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
