const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');
const { StringDecoder } = require('string_decoder');
const crypto = require('crypto');

const HOST = process.env.HOST || '127.0.0.1';
const PORT = Number(process.env.PORT || 8787);
const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const SHARED_DIR = path.join(ROOT, 'shared');
const DATA_DIR = path.join(os.homedir(), '.local', 'share', 'pi-web-chat');
const SESSION_DIR = path.join(DATA_DIR, 'sessions');

fs.mkdirSync(PUBLIC_DIR, { recursive: true });
fs.mkdirSync(SHARED_DIR, { recursive: true });
fs.mkdirSync(SESSION_DIR, { recursive: true });

let rpc = null;
let rpcReady = false;
let rpcState = {
  isStreaming: false,
  pendingMessageCount: 0,
  messageCount: 0,
  sessionId: null,
  sessionName: 'Web Chat',
  model: null,
  thinkingLevel: null,
  lastError: null,
};

const pending = new Map();
const sseClients = new Set();
let rpcRestartTimer = null;
let rpcExitReason = 'unexpected_exit';

function makeId() {
  return crypto.randomUUID();
}

function sendSse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function broadcast(event, data) {
  for (const client of sseClients) {
    sendSse(client, event, data);
  }
}

function clearRpcRestartTimer() {
  if (!rpcRestartTimer) return;
  clearTimeout(rpcRestartTimer);
  rpcRestartTimer = null;
}

function scheduleRpcRestart(delayMs = 1000) {
  clearRpcRestartTimer();
  rpcRestartTimer = setTimeout(() => {
    rpcRestartTimer = null;
    try {
      startRpc();
      refreshState().catch(() => {});
    } catch (restartError) {
      console.error('Failed to restart pi RPC:', restartError);
    }
  }, delayMs);
}

function attachJsonlReader(stream, onLine) {
  const decoder = new StringDecoder('utf8');
  let buffer = '';

  stream.on('data', (chunk) => {
    buffer += typeof chunk === 'string' ? chunk : decoder.write(chunk);
    while (true) {
      const newlineIndex = buffer.indexOf('\n');
      if (newlineIndex === -1) break;
      let line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      if (line.endsWith('\r')) line = line.slice(0, -1);
      if (line.trim().length === 0) continue;
      onLine(line);
    }
  });

  stream.on('end', () => {
    buffer += decoder.end();
    const line = buffer.trim();
    if (line) onLine(line);
  });
}

function updateStateFromEvent(event) {
  switch (event.type) {
    case 'agent_start':
      rpcState.isStreaming = true;
      break;
    case 'agent_settled':
      rpcState.isStreaming = false;
      refreshState().catch(() => {});
      break;
    case 'queue_update':
      if (typeof event.pendingMessageCount === 'number') {
        rpcState.pendingMessageCount = event.pendingMessageCount;
      }
      break;
    case 'extension_error':
      rpcState.lastError = event.error || 'Extension error';
      break;
    default:
      break;
  }
}

function startRpc() {
  if (rpc) return;

  rpc = spawn('pi', ['--mode', 'rpc', '--session-dir', SESSION_DIR, '--name', 'Web Chat'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: process.env,
  });

  rpcReady = true;
  rpcState.lastError = null;

  attachJsonlReader(rpc.stdout, (line) => {
    let payload;
    try {
      payload = JSON.parse(line);
    } catch (error) {
      console.error('Failed to parse RPC stdout:', line, error);
      return;
    }

    if (payload.type === 'response' && payload.id && pending.has(payload.id)) {
      const waiter = pending.get(payload.id);
      pending.delete(payload.id);
      waiter.resolve(payload);
      return;
    }

    updateStateFromEvent(payload);
    broadcast('rpc', payload);
  });

  attachJsonlReader(rpc.stderr, (line) => {
    console.error('[pi stderr]', line);
    rpcState.lastError = line;
    broadcast('server', { type: 'stderr', line });
  });

  rpc.on('exit', (code, signal) => {
    const exitReason = rpcExitReason;
    rpcExitReason = 'unexpected_exit';
    console.error(`pi RPC exited (code=${code}, signal=${signal}, reason=${exitReason})`);
    rpcReady = false;
    rpc = null;
    const error = new Error(`pi RPC exited (code=${code}, signal=${signal})`);
    for (const [, waiter] of pending) waiter.reject(error);
    pending.clear();
    rpcState.isStreaming = false;
    rpcState.lastError = error.message;
    broadcast('server', { type: 'rpc_exit', code, signal, message: error.message, reason: exitReason });
    if (exitReason !== 'manual_restart') {
      scheduleRpcRestart(1000);
    }
  });

  refreshState().catch((error) => {
    console.error('Initial state refresh failed:', error);
  });
}

function stopRpcProcess(options = {}) {
  const { reason = 'manual_restart', timeoutMs = 2000 } = options;
  clearRpcRestartTimer();

  if (!rpc) return Promise.resolve(false);

  const child = rpc;
  rpcExitReason = reason;

  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(forceTimer);
      resolve(true);
    };

    child.once('exit', finish);

    const forceTimer = setTimeout(() => {
      try {
        child.kill('SIGKILL');
      } catch {
        finish();
      }
    }, timeoutMs);

    try {
      child.kill('SIGTERM');
    } catch {
      finish();
    }
  });
}

async function restartRpcProcess() {
  clearRpcRestartTimer();
  if (rpc) {
    await stopRpcProcess({ reason: 'manual_restart' });
  }
  startRpc();
  const state = await refreshState();
  broadcast('server', { type: 'rpc_restart', state });
  return state;
}

function launchServerRecovery() {
  const child = spawn(path.join(ROOT, 'recover.sh'), [], {
    cwd: ROOT,
    detached: true,
    stdio: 'ignore',
    env: process.env,
  });
  child.unref();
}

function sendRpc(command) {
  startRpc();
  if (!rpcReady || !rpc) {
    return Promise.reject(new Error('pi RPC is not ready'));
  }

  const id = command.id || makeId();
  const message = { ...command, id };

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    try {
      rpc.stdin.write(JSON.stringify(message) + '\n', (error) => {
        if (error) {
          pending.delete(id);
          reject(error);
        }
      });
    } catch (error) {
      pending.delete(id);
      reject(error);
    }
  });
}

async function refreshState() {
  const response = await sendRpc({ type: 'get_state' });
  if (response.success && response.data) {
    rpcState = { ...rpcState, ...response.data };
  }
  return rpcState;
}

async function getMessages() {
  const response = await sendRpc({ type: 'get_messages' });
  if (!response.success) {
    throw new Error(response.error || 'Failed to load messages');
  }
  return response.data?.messages || [];
}

function json(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(JSON.stringify(data));
}

function serveFile(req, res, filePath, contentType) {
  fs.readFile(filePath, (error, content) => {
    if (error) {
      json(res, 404, { error: 'Not found' });
      return;
    }

    const headers = {
      'Content-Type': /^(text\/|application\/(javascript|json|manifest\+json|xml))/i.test(contentType)
        ? `${contentType}; charset=utf-8`
        : contentType,
      'Content-Length': Buffer.byteLength(content),
    };

    res.writeHead(200, headers);
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    res.end(content);
  });
}

function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function getContentType(filePath) {
  switch (path.extname(filePath).toLowerCase()) {
    case '.html': return 'text/html';
    case '.htm': return 'text/html';
    case '.md': return 'text/markdown';
    case '.txt': return 'text/plain';
    case '.json': return 'application/json';
    case '.csv': return 'text/csv';
    case '.js': return 'application/javascript';
    case '.mjs': return 'application/javascript';
    case '.cjs': return 'application/javascript';
    case '.css': return 'text/css';
    case '.xml': return 'application/xml';
    case '.yml': return 'text/plain';
    case '.yaml': return 'text/plain';
    case '.pdf': return 'application/pdf';
    case '.png': return 'image/png';
    case '.jpg': return 'image/jpeg';
    case '.jpeg': return 'image/jpeg';
    case '.gif': return 'image/gif';
    case '.webp': return 'image/webp';
    case '.svg': return 'image/svg+xml';
    default: return 'application/octet-stream';
  }
}

function resolveSharedPath(urlPathname) {
  const rawRelative = decodeURIComponent(urlPathname.replace(/^\/shared\/?/, ''));
  const relativePath = rawRelative.replace(/^\/+/, '');
  const absolutePath = path.resolve(SHARED_DIR, relativePath);
  if (absolutePath !== SHARED_DIR && !absolutePath.startsWith(`${SHARED_DIR}${path.sep}`)) {
    return null;
  }
  return { relativePath, absolutePath };
}

function listSharedEntries(dirPath, baseUrlPath = '/shared') {
  return fs.readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith('.'))
    .sort((a, b) => {
      if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
      return a.name.localeCompare(b.name);
    })
    .map((entry) => {
      const hrefPath = path.posix.join(baseUrlPath, entry.name);
      const href = `${hrefPath}${entry.isDirectory() ? '/' : ''}`;
      return {
        name: entry.name,
        href,
        isDirectory: entry.isDirectory(),
      };
    });
}

function serveSharedIndex(req, res, dirPath, urlPathname) {
  const entries = listSharedEntries(dirPath, urlPathname === '/shared' ? '/shared' : urlPathname.replace(/\/$/, ''));
  const relativeTitle = dirPath === SHARED_DIR
    ? 'shared'
    : path.relative(SHARED_DIR, dirPath).split(path.sep).join('/');
  const parentHref = dirPath === SHARED_DIR
    ? '/'
    : `${urlPathname.replace(/\/?[^/]+\/?$/, '') || '/shared'}`;
  const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Shared Documents</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 0; background: #081008; color: #c7f7c7; }
      main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
      h1 { margin-top: 0; }
      a { color: #b8ffb8; }
      ul { list-style: none; padding: 0; margin: 16px 0 0; }
      li { margin: 0 0 12px; }
      .entry { display: block; padding: 12px 14px; border: 2px solid #4f8a4f; background: #102010; text-decoration: none; }
      .meta { opacity: 0.75; font-size: 0.9rem; }
    </style>
  </head>
  <body>
    <main>
      <h1>Shared Documents</h1>
      <p>Folder: <code>${escapeHtml(relativeTitle)}</code></p>
      ${dirPath === SHARED_DIR ? '<p>Ask Percy to create a document in the shared folder, then open it from chat.</p>' : `<p><a href="${escapeHtml(parentHref)}">← Up one level</a></p>`}
      <ul>
        ${entries.length ? entries.map((entry) => `<li><a class="entry" href="${escapeHtml(entry.href)}">${escapeHtml(entry.name)}${entry.isDirectory ? '/' : ''}<div class="meta">${entry.isDirectory ? 'Folder' : 'Document'}</div></a></li>`).join('') : '<li>No shared documents yet.</li>'}
      </ul>
    </main>
  </body>
</html>`;

  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  res.end(html);
}

function readBody(req, options = {}) {
  const maxBytes = Number(options.maxBytes || 1024 * 1024);

  return new Promise((resolve, reject) => {
    let body = '';
    let settled = false;

    req.on('data', (chunk) => {
      if (settled) return;
      body += chunk;
      if (body.length > maxBytes) {
        settled = true;
        reject(new Error('Request body too large'));
        req.destroy();
      }
    });
    req.on('end', () => {
      if (settled) return;
      if (!body) return resolve({});
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(new Error('Invalid JSON body'));
      }
    });
    req.on('error', (error) => {
      if (settled) return;
      reject(error);
    });
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);

  try {
    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'index.html'), 'text/html');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/app.js') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'app.js'), 'application/javascript');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/style.css') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'style.css'), 'text/css');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/manifest.webmanifest') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'manifest.webmanifest'), 'application/manifest+json');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/sw.js') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'sw.js'), 'application/javascript');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/icon-192.png') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'icon-192.png'), 'image/png');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/icon-512.png') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'icon-512.png'), 'image/png');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname === '/apple-touch-icon.png') {
      return serveFile(req, res, path.join(PUBLIC_DIR, 'apple-touch-icon.png'), 'image/png');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && (url.pathname === '/shared' || url.pathname === '/shared/')) {
      return serveSharedIndex(req, res, SHARED_DIR, '/shared');
    }

    if ((req.method === 'GET' || req.method === 'HEAD') && url.pathname.startsWith('/shared/')) {
      const resolved = resolveSharedPath(url.pathname);
      if (!resolved) return json(res, 404, { error: 'Not found' });

      let stat;
      try {
        stat = fs.statSync(resolved.absolutePath);
      } catch {
        return json(res, 404, { error: 'Not found' });
      }

      if (stat.isDirectory()) {
        return serveSharedIndex(req, res, resolved.absolutePath, url.pathname.replace(/\/$/, ''));
      }

      return serveFile(req, res, resolved.absolutePath, getContentType(resolved.absolutePath));
    }

    if (req.method === 'GET' && url.pathname === '/api/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      });
      res.write(': connected\n\n');
      sseClients.add(res);
      sendSse(res, 'server', { type: 'hello', state: rpcState });
      const keepAlive = setInterval(() => res.write(': keepalive\n\n'), 15000);
      req.on('close', () => {
        clearInterval(keepAlive);
        sseClients.delete(res);
      });
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/state') {
      const state = await refreshState();
      return json(res, 200, { ok: true, state });
    }

    if (req.method === 'GET' && url.pathname === '/api/messages') {
      const messages = await getMessages();
      return json(res, 200, { ok: true, messages });
    }

    if (req.method === 'POST' && url.pathname === '/api/prompt') {
      const body = await readBody(req, { maxBytes: 15 * 1024 * 1024 });
      const message = String(body.message || '').trim();
      if (!message) return json(res, 400, { ok: false, error: 'Message is required' });

      const images = Array.isArray(body.images)
        ? body.images
            .filter((image) => image && image.type === 'image' && typeof image.data === 'string' && typeof image.mimeType === 'string')
            .map((image) => ({
              type: 'image',
              data: image.data,
              mimeType: image.mimeType,
            }))
        : undefined;

      const command = { type: 'prompt', message };
      if (images?.length) {
        command.images = images;
      }
      if (rpcState.isStreaming) {
        command.streamingBehavior = body.streamingBehavior === 'steer' ? 'steer' : 'followUp';
      }

      const response = await sendRpc(command);
      return json(res, 200, {
        ok: response.success,
        queued: Boolean(command.streamingBehavior),
        response,
      });
    }

    if (req.method === 'POST' && url.pathname === '/api/abort') {
      const response = await sendRpc({ type: 'abort' });
      return json(res, 200, { ok: response.success, response });
    }

    if (req.method === 'POST' && url.pathname === '/api/new-session') {
      const response = await sendRpc({ type: 'new_session' });
      await refreshState();
      broadcast('server', { type: 'new_session' });
      return json(res, 200, { ok: response.success, response, state: rpcState });
    }

    if (req.method === 'POST' && url.pathname === '/api/restart-server') {
      broadcast('server', { type: 'server_restarting' });
      launchServerRecovery();
      return json(res, 200, { ok: true, restarting: true });
    }

    if (req.method === 'GET' && url.pathname === '/api/healthz') {
      return json(res, 200, { ok: true, rpcReady, state: rpcState });
    }

    json(res, 404, { error: 'Not found' });
  } catch (error) {
    console.error(error);
    json(res, 500, { ok: false, error: error.message || String(error) });
  }
});

startRpc();

server.listen(PORT, HOST, () => {
  console.log(`Pi Web Chat listening on http://${HOST}:${PORT}`);
});
