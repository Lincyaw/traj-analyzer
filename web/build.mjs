import { build } from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const output = fileURLToPath(new URL("../src/traj_analyzer/viewer/static/vendor/", import.meta.url));
await mkdir(output, { recursive: true });
await build({
  entryPoints: ["perspective.js"],
  outfile: `${output}/perspective.js`,
  bundle: true,
  minify: true,
  format: "esm",
  target: "es2022",
  legalComments: "linked",
});
for (const [source, target] of [
  ["client/dist/wasm/perspective-js.wasm", "perspective-js.wasm"],
  ["server/dist/wasm/perspective-server.wasm", "perspective-server.wasm"],
  ["viewer/dist/wasm/perspective-viewer.wasm", "perspective-viewer.wasm"],
  ["viewer/dist/css/pro.css", "pro.css"],
  ["viewer/dist/css/pro-dark.css", "pro-dark.css"],
  ["viewer/LICENSE.md", "LICENSE.md"],
]) {
  await copyFile(`node_modules/@perspective-dev/${source}`, `${output}/${target}`);
}
for (const [source, target] of [
  ["regular-layout/LICENSE.md", "LICENSE-regular-layout.md"],
  ["regular-table/LICENSE", "LICENSE-regular-table"],
]) {
  await copyFile(`node_modules/${source}`, `${output}/${target}`);
}
