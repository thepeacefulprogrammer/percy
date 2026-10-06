const fs = require('fs');
const path = require('path');

function createHttpHelpers(options) {
  const {
    fsp,
    publicDir,
    sharedDir,
    publicFileRoutes,
    json,
  } = options;

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

  function resolvePublicPath(urlPathname) {
    const rawRelative = decodeURIComponent(urlPathname.replace(/^\/+/, ''));
    const relativePath = rawRelative.replace(/^\/+/, '');
    const absolutePath = path.resolve(publicDir, relativePath);
    if (absolutePath !== publicDir && !absolutePath.startsWith(`${publicDir}${path.sep}`)) {
      return null;
    }
    return { relativePath, absolutePath };
  }

  function resolveSharedPath(urlPathname) {
    const rawRelative = decodeURIComponent(urlPathname.replace(/^\/shared\/?/, ''));
    const relativePath = rawRelative.replace(/^\/+/, '');
    const absolutePath = path.resolve(sharedDir, relativePath);
    if (absolutePath !== sharedDir && !absolutePath.startsWith(`${sharedDir}${path.sep}`)) {
      return null;
    }
    return { relativePath, absolutePath };
  }

  async function listSharedEntries(dirPath, baseUrlPath = '/shared') {
    return (await fsp.readdir(dirPath, { withFileTypes: true }))
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

  async function serveSharedIndex(req, res, dirPath, urlPathname) {
    const entries = await listSharedEntries(dirPath, urlPathname === '/shared' ? '/shared' : urlPathname.replace(/\/$/, ''));
    const relativeTitle = dirPath === sharedDir
      ? 'shared'
      : path.relative(sharedDir, dirPath).split(path.sep).join('/');
    const parentHref = dirPath === sharedDir
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
      ${dirPath === sharedDir ? '<p>Ask Percy to create a document in the shared folder, then open it from chat.</p>' : `<p><a href="${escapeHtml(parentHref)}">← Up one level</a></p>`}
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

  function isReadRequest(req) {
    return req.method === 'GET' || req.method === 'HEAD';
  }

  async function servePublicRoute(req, res, pathname) {
    if (!isReadRequest(req)) return false;

    const route = publicFileRoutes.get(pathname);
    if (route) {
      serveFile(req, res, route.filePath, route.contentType);
      return true;
    }

    const resolved = resolvePublicPath(pathname);
    if (!resolved) return false;

    let stat;
    try {
      stat = await fsp.stat(resolved.absolutePath);
    } catch {
      return false;
    }

    if (!stat.isFile()) return false;

    serveFile(req, res, resolved.absolutePath, getContentType(resolved.absolutePath));
    return true;
  }

  async function serveSharedRoute(req, res, pathname) {
    if (!isReadRequest(req)) return false;

    if (pathname === '/shared' || pathname === '/shared/') {
      await serveSharedIndex(req, res, sharedDir, '/shared');
      return true;
    }

    if (!pathname.startsWith('/shared/')) return false;

    const resolved = resolveSharedPath(pathname);
    if (!resolved) {
      json(res, 404, { error: 'Not found' });
      return true;
    }

    let stat;
    try {
      stat = await fsp.stat(resolved.absolutePath);
    } catch {
      json(res, 404, { error: 'Not found' });
      return true;
    }

    if (stat.isDirectory()) {
      await serveSharedIndex(req, res, resolved.absolutePath, pathname.replace(/\/$/, ''));
      return true;
    }

    serveFile(req, res, resolved.absolutePath, getContentType(resolved.absolutePath));
    return true;
  }

  return {
    servePublicRoute,
    serveSharedRoute,
  };
}

module.exports = {
  createHttpHelpers,
};
