import { spawn } from "child_process";
import path from "path";
import { NextRequest } from "next/server";

/** Allow train API only when running locally (admin). Set ALLOW_TRAIN_API=true to bypass host check. */
function isTrainAllowed(request: NextRequest): boolean {
  if (process.env.ALLOW_TRAIN_API === "true") return true;
  const host = request.headers.get("host") ?? "";
  return host.startsWith("localhost:") || host.startsWith("127.0.0.1:");
}

/**
 * POST /api/train — run SJM training (and optional OOS allocation) and stream output via SSE.
 * Local only (admin). Body: { jump_penalty?, sparsity_param?, train_start, train_end, oos_start?, oos_end?, config_path? }
 */
export async function POST(request: NextRequest) {
  if (!isTrainAllowed(request)) {
    return new Response(JSON.stringify({ error: "Train API is only allowed when running locally." }), {
      status: 403,
      headers: { "Content-Type": "application/json" },
    });
  }

  let body: {
    jump_penalty?: number;
    sparsity_param?: number;
    train_start: string;
    train_end: string;
    oos_start?: string;
    oos_end?: string;
    config_path?: string;
    log_mlflow?: boolean;
  };
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }
  const {
    train_start,
    train_end,
    oos_start,
    oos_end,
    jump_penalty = 50,
    sparsity_param = 9.5,
    config_path,
    log_mlflow = true,
  } = body;
  if (!train_start || !train_end) {
    return new Response(
      JSON.stringify({ error: "train_start and train_end required" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const projectRoot = path.resolve(process.cwd(), "..");
  const scriptPath = path.join(projectRoot, "scripts", "run_training_stream.py");
  const args = [
    scriptPath,
    "--train-start",
    train_start,
    "--train-end",
    train_end,
    "--lambda",
    String(jump_penalty),
    "--kappa2",
    String(sparsity_param),
  ];
  if (oos_start) args.push("--oos-start", oos_start);
  if (oos_end) args.push("--oos-end", oos_end);
  if (config_path) args.push("--config", path.join(projectRoot, config_path));
  if (log_mlflow) args.push("--mlflow");

  const encoder = new TextEncoder();
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
  });

  function send(data: object) {
    if (controller) {
      try {
        controller.enqueue(encoder.encode("data: " + JSON.stringify(data) + "\n\n"));
      } catch (_) {}
    }
  }

  // Use your conda env: HELIX_PYTHON in .env, or CONDA_PREFIX when Next is started with conda activated
  const pythonExe =
    process.env.HELIX_PYTHON ||
    (process.env.CONDA_PREFIX ? `${process.env.CONDA_PREFIX}/bin/python` : null) ||
    "python3";
  const child = spawn(pythonExe, args, {
    cwd: projectRoot,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    const lines = chunk.split("\n").filter(Boolean);
    for (const line of lines) {
      if (line.startsWith("LOSS\t")) {
        const parts = line.split("\t");
        if (parts.length >= 4) {
          send({
            type: "loss",
            factor: parts[1],
            iter: parseInt(parts[2], 10),
            value: parseFloat(parts[3]),
          });
        }
      } else if (line.startsWith("LOG\t")) {
        send({ type: "log", line: line.slice(4) });
      } else if (line.startsWith("DONE\t")) {
        send({ type: "done", line: line.slice(5) });
      } else if (line.startsWith("ERR\t")) {
        send({ type: "error", line: line.slice(4) });
      } else {
        send({ type: "log", line });
      }
    }
  });

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    const s = chunk.trim();
    if (s) send({ type: "log", line: "[stderr] " + s });
  });

  child.on("close", (code) => {
    send({ type: "exit", code: code ?? 0 });
    if (controller) {
      try {
        controller.close();
      } catch (_) {}
    }
  });

  child.on("error", (err) => {
    send({ type: "error", line: String(err.message) });
    if (controller) {
      try {
        controller.close();
      } catch (_) {}
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-store",
      Connection: "keep-alive",
    },
  });
}
