const http = require('http');
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const os = require('os');

const { createRpcManager } = require('./lib/rpc-manager');
const { createDocumentParser } = require('./lib/document-parser');
const { createHttpHelpers } = require('./lib/http-helpers');

const HOST = process.env.HOST || '127.0.0.1';
const PORT = Number(process.env.PORT || 8787);
const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const SHARED_DIR = path.join(ROOT, 'shared');
const DATA_DIR = path.join(os.homedir(), '.local', 'share', 'pi-web-chat');
const SESSION_DIR = path.join(DATA_DIR, 'sessions');
const TMP_DIR = path.join(DATA_DIR, 'tmp');
const DOCLING_SCRIPT = path.join(ROOT, 'scripts', 'parse_with_docling.py');
const DOCLING_PARSE_MAX_BYTES = 20 * 1024 * 1024;
const DOCLING_REQUEST_MAX_BYTES = 30 * 1024 * 1024;
const DEFAULT_RPC_TIMEOUT_MS = Number(process.env.PERCY_RPC_TIMEOUT_MS || 30000) > 0
  ? Number(process.env.PERCY_RPC_TIMEOUT_MS || 30000)
  : 30000;
const WEB_CHAT_APPEND_SYSTEM_PROMPT = [
  'In this web chat, the tool list in the system prompt is authoritative.',
  'If read, bash, edit, write, or any other tool is listed there, it is available to you.',
  'Do not claim that tools are unavailable unless you actually attempted the relevant tool call and received an error.',
  'When the user asks you to inspect files, run commands, edit code, or validate behavior, do those actions yourself with the available tools.',
  'Do not tell the user to run commands, inspect files, or paste outputs when you can obtain that information directly with the available tools.',
  'Continue working until the task is complete or you are blocked by missing permissions, credentials, confirmation for a destructive action, or unavailable tools.',
].join('\n');
const PUBLIC_FILE_ROUTES = new Map([
  ['/', { filePath: path.join(PUBLIC_DIR, 'index.html'), contentType: 'text/html' }],
  ['/app.js', { filePath: path.join(PUBLIC_DIR, 'app.js'), contentType: 'application/javascript' }],
  ['/style.css', { filePath: path.join(PUBLIC_DIR, 'style.css'), contentType: 'text/css' }],
  ['/manifest.webmanifest', { filePath: path.join(PUBLIC_DIR, 'manifest.webmanifest'), contentType: 'application/manifest+json' }],
  ['/sw.js', { filePath: path.join(PUBLIC_DIR, 'sw.js'), contentType: 'application/javascript' }],
  ['/icon-192.png', { filePath: path.join(PUBLIC_DIR, 'icon-192.png'), contentType: 'image/png' }],
  ['/icon-512.png', { filePath: path.join(PUBLIC_DIR, 'icon-512.png'), contentType: 'image/png' }],
  ['/apple-touch-icon.png', { filePath: path.join(PUBLIC_DIR, 'apple-touch-icon.png'), contentType: 'image/png' }],
]);

fs.mkdirSync(PUBLIC_DIR, { recursive: true });
fs.mkdirSync(SHARED_DIR, { recursive: true });
fs.mkdirSync(SESSION_DIR, { recursive: true });
fs.mkdirSync(TMP_DIR, { recursive: true });

const sseClients = new Set();

function sendSse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function broadcast(event, data) {
  for (const client of sseClients) {
    sendSse(client, event, data);
  }
}

const rpcManager = createRpcManager({
  sessionDir: SESSION_DIR,
  appendSystemPrompt: WEB_CHAT_APPEND_SYSTEM_PROMPT,
  defaultRpcTimeoutMs: DEFAULT_RPC_TIMEOUT_MS,
  sessionName: 'Web Chat',
  onRpcEvent: (payload) => broadcast('rpc', payload),
  onServerEvent: (payload) => broadcast('server', payload),
});

const { parseDocumentUpload } = createDocumentParser({
  root: ROOT,
  tmpDir: TMP_DIR,
  scriptPath: DOCLING_SCRIPT,
  parseMaxBytes: DOCLING_PARSE_MAX_BYTES,
});

function launchServerRecovery() {
  const { spawn } = require('child_process');
  const child = spawn(path.join(ROOT, 'recover.sh'), [], {
    cwd: ROOT,
    detached: true,
    stdio: 'ignore',
    env: process.env,
  });
  child.unref();
}

function json(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(JSON.stringify(data));
}

const { servePublicRoute, serveSharedRoute } = createHttpHelpers({
  fsp,
  publicDir: PUBLIC_DIR,
  sharedDir: SHARED_DIR,
  publicFileRoutes: PUBLIC_FILE_ROUTES,
  json,
});

function readBody(req, options = {}) {
  const maxBytes = Number(options.maxBytes || 1024 * 1024);

  return new Promise((resolve, reject) => {
    const chunks = [];
    let totalBytes = 0;
    let settled = false;

    req.on('data', (chunk) => {
      if (settled) return;
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      totalBytes += buffer.length;
      if (totalBytes > maxBytes) {
        settled = true;
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(buffer);
    });

    req.on('end', () => {
      if (settled) return;
      if (!chunks.length) return resolve({});
      try {
        const body = Buffer.concat(chunks).toString('utf8');
        resolve(JSON.parse(body));
      } catch {
        reject(new Error('Invalid JSON body'));
      }
    });

    req.on('error', (error) => {
      if (settled) return;
      reject(error);
    });
  });
}

function handleEvents(req, res) {
  if (req.method !== 'GET') return false;

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(': connected\n\n');
  sseClients.add(res);
  sendSse(res, 'server', { type: 'hello', state: rpcManager.getState() });
  const keepAlive = setInterval(() => res.write(': keepalive\n\n'), 15000);
  req.on('close', () => {
    clearInterval(keepAlive);
    sseClients.delete(res);
  });
  return true;
}

function buildPromptCommand(body) {
  const images = Array.isArray(body.images)
    ? body.images
        .filter((image) => image && image.type === 'image' && typeof image.data === 'string' && typeof image.mimeType === 'string')
        .map((image) => ({
          type: 'image',
          data: image.data,
          mimeType: image.mimeType,
        }))
    : undefined;

  const command = { type: 'prompt', message: String(body.message || '').trim() };
  if (images?.length) {
    command.images = images;
  }
  if (rpcManager.getState().isStreaming) {
    command.streamingBehavior = body.streamingBehavior === 'steer' ? 'steer' : 'followUp';
  }
  return command;
}

async function handleGetApi(req, res, pathname) {
  if (req.method !== 'GET') return false;

  if (pathname === '/api/events') {
    return handleEvents(req, res);
  }

  if (pathname === '/api/state') {
    const state = await rpcManager.refreshState();
    json(res, 200, { ok: true, state });
    return true;
  }

  if (pathname === '/api/messages') {
    const messages = await rpcManager.getMessages();
    json(res, 200, { ok: true, messages });
    return true;
  }

  if (pathname === '/api/livez') {
    const health = rpcManager.getHealthPayload();
    json(res, 200, { ...health, ok: true });
    return true;
  }

  if (pathname === '/api/healthz' || pathname === '/api/readyz') {
    const health = rpcManager.getHealthPayload();
    json(res, health.ready ? 200 : 503, health);
    return true;
  }

  return false;
}

async function handlePostApi(req, res, pathname) {
  if (req.method !== 'POST') return false;

  if (pathname === '/api/parse-document') {
    const body = await readBody(req, { maxBytes: DOCLING_REQUEST_MAX_BYTES });
    const result = await parseDocumentUpload(body);
    json(res, 200, {
      ok: true,
      engine: result.engine || 'docling',
      content: result.content,
    });
    return true;
  }

  if (pathname === '/api/prompt') {
    const body = await readBody(req, { maxBytes: 15 * 1024 * 1024 });
    const command = buildPromptCommand(body);
    if (!command.message) {
      json(res, 400, { ok: false, error: 'Message is required' });
      return true;
    }

    const response = await rpcManager.sendRpc(command);
    json(res, 200, {
      ok: response.success,
      queued: Boolean(command.streamingBehavior),
      response,
    });
    return true;
  }

  if (pathname === '/api/abort') {
    const response = await rpcManager.sendRpc({ type: 'abort' });
    json(res, 200, { ok: response.success, response });
    return true;
  }

  if (pathname === '/api/new-session') {
    const response = await rpcManager.sendRpc({ type: 'new_session' });
    await rpcManager.refreshState();
    broadcast('server', { type: 'new_session' });
    json(res, 200, { ok: response.success, response, state: rpcManager.getState() });
    return true;
  }

  if (pathname === '/api/restart-server') {
    broadcast('server', { type: 'server_restarting' });
    launchServerRecovery();
    json(res, 200, { ok: true, restarting: true });
    return true;
  }

  return false;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);

  try {
    if (servePublicRoute(req, res, url.pathname)) return;
    if (await serveSharedRoute(req, res, url.pathname)) return;
    if (await handleGetApi(req, res, url.pathname)) return;
    if (await handlePostApi(req, res, url.pathname)) return;

    json(res, 404, { error: 'Not found' });
  } catch (error) {
    console.error(error);
    json(res, 500, { ok: false, error: error.message || String(error) });
  }
});

rpcManager.startRpc();

server.listen(PORT, HOST, () => {
  console.log(`Pi Web Chat listening on http://${HOST}:${PORT}`);
});
