const { spawn } = require('child_process');
const { StringDecoder } = require('string_decoder');
const crypto = require('crypto');

function createInitialRpcState(sessionName) {
  return {
    isStreaming: false,
    pendingMessageCount: 0,
    messageCount: 0,
    sessionId: null,
    sessionName,
    model: null,
    thinkingLevel: null,
    lastError: null,
  };
}

function createRpcManager(options) {
  const {
    sessionDir,
    appendSystemPrompt,
    defaultRpcTimeoutMs,
    env = process.env,
    sessionName = 'Web Chat',
    onRpcEvent = () => {},
    onServerEvent = () => {},
  } = options;

  let rpc = null;
  let rpcReady = false;
  let rpcState = createInitialRpcState(sessionName);
  const pending = new Map();
  let rpcRestartTimer = null;
  let rpcExitReason = 'unexpected_exit';

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

  function getHealthPayload() {
    const ready = rpcReady && Boolean(rpc);
    return {
      ok: ready,
      live: true,
      ready,
      rpcReady: ready,
      state: rpcState,
    };
  }

  function startRpc() {
    if (rpc) return;

    rpc = spawn('pi', [
      '--approve',
      '--mode', 'rpc',
      '--session-dir', sessionDir,
      '--name', sessionName,
      '--append-system-prompt', appendSystemPrompt,
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
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
      onRpcEvent(payload);
    });

    attachJsonlReader(rpc.stderr, (line) => {
      console.error('[pi stderr]', line);
      rpcState.lastError = line;
      onServerEvent({ type: 'stderr', line });
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
      onServerEvent({ type: 'rpc_exit', code, signal, message: error.message, reason: exitReason });
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
    rpcReady = false;

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

  function sendRpc(command, options = {}) {
    const timeoutMs = Number(options.timeoutMs || defaultRpcTimeoutMs);

    startRpc();
    if (!rpcReady || !rpc) {
      return Promise.reject(new Error('pi RPC is not ready'));
    }

    const id = command.id || crypto.randomUUID();
    const message = { ...command, id };

    return new Promise((resolve, reject) => {
      let timer = null;

      const finalize = (handler, value) => {
        if (timer) clearTimeout(timer);
        pending.delete(id);
        handler(value);
      };

      timer = setTimeout(() => {
        const error = new Error(`pi RPC timed out after ${timeoutMs}ms`);
        rpcState.lastError = error.message;
        pending.delete(id);
        onServerEvent({
          type: 'rpc_timeout',
          commandType: command.type || 'unknown',
          timeoutMs,
          message: error.message,
        });
        reject(error);
        stopRpcProcess({ reason: 'timeout_restart' }).catch((stopError) => {
          console.error('Failed to stop timed out pi RPC:', stopError);
        });
      }, timeoutMs);

      pending.set(id, {
        resolve: (payload) => finalize(resolve, payload),
        reject: (error) => finalize(reject, error),
      });

      try {
        rpc.stdin.write(JSON.stringify(message) + '\n', (error) => {
          if (error && pending.has(id)) {
            finalize(reject, error);
          }
        });
      } catch (error) {
        if (pending.has(id)) {
          finalize(reject, error);
        }
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

  function getState() {
    return rpcState;
  }

  return {
    getHealthPayload,
    getMessages,
    getState,
    refreshState,
    sendRpc,
    startRpc,
    stopRpcProcess,
  };
}

module.exports = {
  createRpcManager,
};
