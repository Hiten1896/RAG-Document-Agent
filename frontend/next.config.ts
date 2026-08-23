import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // There is a stray package-lock.json in the repo root alongside this app's
  // own lockfile, so Turbopack inferred the repo root as the workspace root and
  // warned on every build. Pin it to this directory explicitly.
  turbopack: {
    root: path.resolve(import.meta.dirname),
  },
};

export default nextConfig;
