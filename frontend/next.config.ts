import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the workspace root to this directory. A stray root-level package.json +
  // package-lock.json used to make Turbopack infer the repo root as the
  // workspace and warn on every build; those are gone now, but the untracked
  // root node_modules/ is still on disk, so the explicit pin stays as a guard
  // against the inference drifting back.
  turbopack: {
    root: path.resolve(import.meta.dirname),
  },
};

export default nextConfig;
