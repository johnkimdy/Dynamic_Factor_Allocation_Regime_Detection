import { readdir, readFile } from "fs/promises";
import path from "path";
import { NextRequest } from "next/server";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const HYPERPARAM_DIR = path.join(PROJECT_ROOT, "hyperparam");

/**
 * GET /api/hyperparam-configs
 * - No query: returns { files: string[] } (list of .json filenames in hyperparam/)
 * - ?config=<filename>: returns the parsed JSON for that file (for UI to lock dates/params)
 */
export async function GET(request: NextRequest) {
  const config = request.nextUrl.searchParams.get("config");
  try {
    if (config) {
      const safeName = path.basename(config);
      if (!safeName.endsWith(".json")) {
        return new Response(JSON.stringify({ error: "Only .json configs allowed" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      const filePath = path.join(HYPERPARAM_DIR, safeName);
      const raw = await readFile(filePath, "utf-8");
      const data = JSON.parse(raw);
      return new Response(JSON.stringify(data), {
        headers: { "Content-Type": "application/json" },
      });
    }
    const entries = await readdir(HYPERPARAM_DIR, { withFileTypes: true });
    const files = entries.filter((e) => e.isFile() && e.name.endsWith(".json")).map((e) => e.name);
    files.sort();
    return new Response(JSON.stringify({ files }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
