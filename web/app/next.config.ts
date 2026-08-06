import type { NextConfig } from "next";

const development = process.env.NODE_ENV === "development";

const nextConfig: NextConfig = {
  output: development ? undefined : "export",
  trailingSlash: true,
  images: { unoptimized: true },
  poweredByHeader: false,
  agentRules: false,
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  ...(development
    ? {
        async rewrites() {
          return [
            { source: "/health", destination: "http://127.0.0.1:7878/health" },
            { source: "/api/:path*", destination: "http://127.0.0.1:7878/api/:path*" },
            { source: "/metrics", destination: "http://127.0.0.1:7878/metrics" },
          ];
        },
      }
    : {}),
};

export default nextConfig;
