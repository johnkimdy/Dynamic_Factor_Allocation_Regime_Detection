#!/usr/bin/env node
/**
 * Wrapper for `next dev` / `next build` that supports --config to select which backtest JSON to load.
 *
 * Usage:
 *   npm run dev
 *   npm run dev -- --config backtest_data_v2.json
 *
 * The --config value is passed via NEXT_PUBLIC_BACKTEST_JSON.
 * The file must exist in public/ (e.g. public/backtest_data_v2.json).
 */
const { spawn } = require("child_process");

const args = process.argv.slice(2);
const command = args[0] === "build" ? "build" : "dev";
const rest = command === "build" ? args.slice(1) : args;

let backtestJson = "backtest_data.json";
const configIdx = rest.indexOf("--config");
if (configIdx !== -1 && rest[configIdx + 1]) {
  backtestJson = rest[configIdx + 1].replace(/^public\//, "");
  rest.splice(configIdx, 2);
}

const env = { ...process.env, NEXT_PUBLIC_BACKTEST_JSON: backtestJson };
const child = spawn("next", [command, ...rest], {
  stdio: "inherit",
  shell: true,
  env,
});

child.on("exit", (code) => process.exit(code ?? 0));
