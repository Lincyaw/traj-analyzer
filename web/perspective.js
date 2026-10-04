import perspective from "@perspective-dev/client";
import viewer from "@perspective-dev/viewer";
import "@perspective-dev/viewer-datagrid";
import "@perspective-dev/viewer-charts";

export async function createWorker() {
  await viewer.init_client(fetch(new URL("perspective-viewer.wasm", import.meta.url)));
  await perspective.init_client(fetch(new URL("perspective-js.wasm", import.meta.url)));
  await perspective.init_server(fetch(new URL("perspective-server.wasm", import.meta.url)));
  return perspective.worker();
}
