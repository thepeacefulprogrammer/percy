const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const { spawn } = require('child_process');

function createDocumentParser(options) {
  const {
    root,
    tmpDir,
    scriptPath,
    parseMaxBytes,
    env = process.env,
  } = options;

  function sanitizeUploadName(name) {
    const base = path.basename(String(name || 'document.bin')).trim();
    const cleaned = base.replace(/[^a-zA-Z0-9._() -]+/g, '_').replace(/\s+/g, ' ');
    return cleaned || 'document.bin';
  }

  function decodeBase64Payload(data) {
    const value = String(data || '').trim();
    if (!value) {
      throw new Error('Document data is required');
    }

    const buffer = Buffer.from(value, 'base64');
    if (!buffer.length) {
      throw new Error('Invalid document data');
    }

    return buffer;
  }

  function runDoclingParser(filePath) {
    return new Promise((resolve, reject) => {
      const child = spawn('python3', [scriptPath, filePath], {
        cwd: root,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      let finished = false;
      let timedOut = false;

      const finish = (error, value) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve(value);
      };

      const timer = setTimeout(() => {
        timedOut = true;
        try {
          child.kill('SIGKILL');
        } catch {
          finish(new Error('Docling parse timed out'));
        }
      }, 120000);

      child.stdout.on('data', (chunk) => {
        stdout += chunk;
      });

      child.stderr.on('data', (chunk) => {
        stderr += chunk;
      });

      child.on('error', (error) => {
        finish(error);
      });

      child.on('close', (code) => {
        if (timedOut) {
          finish(new Error('Docling parse timed out'));
          return;
        }

        let payload = null;
        const output = stdout.trim();
        if (output) {
          try {
            payload = JSON.parse(output);
          } catch {
            payload = null;
          }
        }

        if (code === 0 && payload?.ok && typeof payload.content === 'string') {
          finish(null, payload);
          return;
        }

        finish(new Error(payload?.error || stderr.trim() || `Docling parser exited with code ${code || 0}`));
      });
    });
  }

  async function parseDocumentUpload(body) {
    const name = sanitizeUploadName(body.name || 'document.bin');
    const buffer = decodeBase64Payload(body.data);

    if (buffer.length > parseMaxBytes) {
      throw new Error(`Document is too large (max ${Math.round(parseMaxBytes / (1024 * 1024))} MB)`);
    }

    const tempDir = await fsp.mkdtemp(path.join(tmpDir, 'docling-'));
    const tempPath = path.join(tempDir, name);

    try {
      await fsp.writeFile(tempPath, buffer);
      return await runDoclingParser(tempPath);
    } finally {
      await fsp.rm(tempDir, { recursive: true, force: true });
    }
  }

  return {
    parseDocumentUpload,
  };
}

module.exports = {
  createDocumentParser,
};
