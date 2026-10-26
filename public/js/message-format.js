export const ATTACHMENT_BLOCK_PATTERN = /<percy-files>([\s\S]*?)<\/percy-files>/g;

export function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function escapeAttribute(text) {
  return escapeHtml(text).replaceAll("'", '&#39;');
}

function sanitizeUrl(url) {
  const value = String(url || '').trim();
  if (/^(https?:|mailto:)/i.test(value)) return value;
  if (/^(\/(?!\/)|\.\/|\.\.\/)/.test(value)) return value;
  return '#';
}

function sanitizeImageUrl(url) {
  const value = String(url || '').trim();
  if (/^(https?:|data:image\/|blob:)/i.test(value)) return value;
  if (/^(\/|\.\/|\.\.\/)/.test(value)) return value;
  return '';
}

function renderInlineMarkdown(text) {
  const tokens = [];
  const stash = (html) => `\u0000${tokens.push(html) - 1}\u0000`;

  let html = escapeHtml(text || '');

  html = html.replace(/`([^`\n]+)`/g, (_, code) => stash(`<code>${code}</code>`));
  html = html.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, url) => {
    const src = sanitizeImageUrl(url);
    if (!src) return alt || '';
    return stash(`<img src="${escapeAttribute(src)}" alt="${escapeAttribute(alt)}" loading="lazy">`);
  });
  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
    const href = escapeAttribute(sanitizeUrl(url));
    return stash(`<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  });

  html = html.replace(/\*\*([^*][\s\S]*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_][\s\S]*?)__/g, '<strong>$1</strong>');
  html = html.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
  html = html.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, '$1<em>$2</em>');

  return html.replace(/\u0000(\d+)\u0000/g, (_, index) => tokens[Number(index)] || '');
}

export function renderMarkdown(text) {
  const source = String(text || '').replace(/\r\n?/g, '\n').trim();
  if (!source) return '';

  const blockTokens = [];
  const stashBlock = (html) => `@@BLOCK${blockTokens.push(html) - 1}@@`;

  const withCodeBlocks = source.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, rawLanguage, rawCode) => {
    const language = String(rawLanguage || '').trim();
    const code = escapeHtml(String(rawCode || '').replace(/\n$/, ''));
    const languageAttr = language ? ` data-language="${escapeAttribute(language)}"` : '';
    return stashBlock(`<pre><code${languageAttr}>${code}</code></pre>`);
  });

  const rendered = withCodeBlocks
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      if (/^@@BLOCK\d+@@$/.test(block)) return block;

      const headingMatch = block.match(/^(#{1,6})\s+(.+)$/);
      if (headingMatch) {
        const level = headingMatch[1].length;
        return `<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`;
      }

      const lines = block.split('\n');

      if (lines.every((line) => /^>\s?/.test(line))) {
        const content = lines.map((line) => renderInlineMarkdown(line.replace(/^>\s?/, ''))).join('<br>');
        return `<blockquote>${content}</blockquote>`;
      }

      if (lines.every((line) => /^[-*+]\s+/.test(line))) {
        const items = lines.map((line) => `<li>${renderInlineMarkdown(line.replace(/^[-*+]\s+/, ''))}</li>`).join('');
        return `<ul>${items}</ul>`;
      }

      if (lines.every((line) => /^\d+\.\s+/.test(line))) {
        const items = lines.map((line) => `<li>${renderInlineMarkdown(line.replace(/^\d+\.\s+/, ''))}</li>`).join('');
        return `<ol>${items}</ol>`;
      }

      return `<p>${lines.map((line) => renderInlineMarkdown(line)).join('<br>')}</p>`;
    })
    .join('');

  return rendered.replace(/@@BLOCK(\d+)@@/g, (_, index) => blockTokens[Number(index)] || '');
}

export function serializeFileAttachments(attachments) {
  return JSON.stringify(
    attachments.map((attachment) => {
      const kind = ['text', 'document', 'audio', 'video', 'file'].includes(attachment?.kind)
        ? attachment.kind
        : 'file';

      return {
        kind,
        name: attachment?.name || 'Attachment',
        size: Number(attachment?.size) || 0,
        mimeType: attachment?.mimeType || (kind === 'text' ? 'text/plain' : 'application/octet-stream'),
        parser: attachment?.parser || null,
        content: kind === 'text' || kind === 'document' ? String(attachment?.content || '') : '',
        path: typeof attachment?.path === 'string' && attachment.path ? attachment.path : null,
      };
    })
  )
    .replaceAll('&', '\\u0026')
    .replaceAll('<', '\\u003c')
    .replaceAll('>', '\\u003e');
}

function normalizeParsedFileAttachment(file) {
  if (!file || typeof file !== 'object') return null;

  const kind = ['text', 'document', 'audio', 'video', 'file'].includes(file.kind)
    ? file.kind
    : 'file';
  const name = typeof file.name === 'string' && file.name.trim() ? file.name.trim() : 'Attachment';
  const content = typeof file.content === 'string' ? file.content : '';
  const path = typeof file.path === 'string' && file.path.trim() ? file.path.trim() : null;

  return {
    kind,
    name,
    content,
    path,
    parser: typeof file.parser === 'string' && file.parser ? file.parser : null,
    size: Number(file.size) > 0 ? Number(file.size) : undefined,
    mimeType: typeof file.mimeType === 'string' && file.mimeType ? file.mimeType : undefined,
  };
}

function parseStructuredFileBlocks(text) {
  const files = [];
  const cleaned = String(text || '').replace(ATTACHMENT_BLOCK_PATTERN, (match, rawPayload) => {
    try {
      const payload = JSON.parse(rawPayload);
      if (!Array.isArray(payload)) return match;
      for (const entry of payload) {
        const normalized = normalizeParsedFileAttachment(entry);
        if (normalized) files.push(normalized);
      }
      return '';
    } catch {
      return match;
    }
  });

  return { text: cleaned, files };
}

function parseLegacyFileBlocks(text) {
  const files = [];
  const cleaned = String(text || '').replace(/<file\s+name="([^"]*)">([\s\S]*?)<\/file>/g, (_, rawName, rawContent) => {
    const name = rawName
      .replaceAll('&quot;', '"')
      .replaceAll('&lt;', '<')
      .replaceAll('&gt;', '>')
      .replaceAll('&amp;', '&');

    const content = String(rawContent || '').trim();
    files.push({
      name: name || 'Attachment',
      content,
      kind: content ? 'text' : 'file',
    });

    return '';
  });

  return { text: cleaned, files };
}

function inferAttachmentKindFromName(name) {
  const extension = String(name || '').toLowerCase().match(/\.([a-z0-9]+)$/)?.[1] || '';
  if (['mp4', 'mov', 'webm', 'mkv', 'avi', 'm4v'].includes(extension)) return 'video';
  if (['mp3', 'wav', 'm4a', 'aac', 'ogg', 'oga', 'flac', 'opus'].includes(extension)) return 'audio';
  if (['txt', 'md', 'markdown', 'json', 'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'css', 'html', 'xml', 'yml', 'yaml', 'csv', 'py', 'sh', 'java', 'kt', 'go', 'rs', 'rb', 'php', 'sql'].includes(extension)) return 'text';
  if (['pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'odt', 'ods', 'odp', 'rtf', 'epub'].includes(extension)) return 'document';
  return 'file';
}

function inferMimeTypeFromName(name) {
  switch (String(name || '').toLowerCase().match(/\.([a-z0-9]+)$/)?.[1] || '') {
    case 'mp4':
    case 'm4v':
      return 'video/mp4';
    case 'mov':
      return 'video/quicktime';
    case 'webm':
      return 'video/webm';
    case 'mkv':
      return 'video/x-matroska';
    case 'avi':
      return 'video/x-msvideo';
    case 'mp3':
      return 'audio/mpeg';
    case 'wav':
      return 'audio/wav';
    case 'm4a':
      return 'audio/mp4';
    case 'aac':
      return 'audio/aac';
    case 'ogg':
    case 'oga':
      return 'audio/ogg';
    case 'flac':
      return 'audio/flac';
    case 'opus':
      return 'audio/opus';
    default:
      return undefined;
  }
}

function parsePathLikeFileReferences(text) {
  const files = [];
  const seen = new Set();
  const cleanedLines = String(text || '')
    .split('\n')
    .map((line) => {
      const match = line.match(/^\s*File:\s*(`?)(.+?)\1\s*$/i)
        || line.match(/^\s*Path:\s*(`?)(.+?)\1\s*$/i)
        || line.match(/^\s*(`?)(shared\/[A-Za-z0-9._()\-\/ ]+)\1\s*$/i)
        || line.match(/^\s*(`?)(\/data\/data\/com\.termux\/files\/home\/[^\s`]+)\1\s*$/i);

      if (!match) return line;

      const rawValue = String(match[2] || '').trim();
      if (!rawValue) return line;

      const normalizedValue = rawValue.replace(/^\.\//, '');
      const isRelativeShared = normalizedValue.startsWith('shared/') && !normalizedValue.includes('..');
      const isAbsoluteShared = /\/shared\//.test(normalizedValue) && !normalizedValue.includes('/../');
      const isAbsoluteUpload = /\/\.local\/share\/pi-web-chat\/uploads\//.test(normalizedValue) && !normalizedValue.includes('/../');

      if (!isRelativeShared && !isAbsoluteShared && !isAbsoluteUpload) return line;

      const previewUrl = isRelativeShared
        ? `/${normalizedValue.split('/').map(encodeURIComponent).join('/')}`
        : `/api/file?path=${encodeURIComponent(normalizedValue)}`;
      const name = normalizedValue.split('/').filter(Boolean).at(-1) || 'Attachment';
      const key = `${name}|${previewUrl}`;
      if (seen.has(key)) return '';
      seen.add(key);

      files.push({
        kind: inferAttachmentKindFromName(name),
        name,
        path: isRelativeShared ? normalizedValue : normalizedValue,
        previewUrl,
        mimeType: inferMimeTypeFromName(name),
      });
      return '';
    });

  return {
    text: cleanedLines.join('\n'),
    files,
  };
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = units[0];
  for (let index = 0; index < units.length; index += 1) {
    unit = units[index];
    if (value < 1024 || index === units.length - 1) break;
    value /= 1024;
  }
  const precision = value >= 10 || unit === 'B' ? 0 : 1;
  return `${value.toFixed(precision)} ${unit}`;
}

export function extractMessageParts(message) {
  if (!message) return { text: '', images: [] };

  if (typeof message.content === 'string') {
    return { text: message.content, images: [] };
  }

  if (!Array.isArray(message.content)) {
    return { text: '', images: [] };
  }

  const text = message.content
    .filter((part) => part && part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text)
    .join('');

  const images = message.content
    .map((part) => {
      if (!part || typeof part !== 'object') return null;

      if (part.type === 'image' && typeof part.data === 'string' && typeof part.mimeType === 'string') {
        return {
          data: part.data,
          mimeType: part.mimeType,
          name: part.name,
        };
      }

      const rawUrl = part.type === 'image_url'
        ? part.image_url?.url || part.url
        : part.type === 'image'
          ? part.url || part.src
          : null;

      if (typeof rawUrl === 'string') {
        const previewUrl = sanitizeImageUrl(rawUrl);
        if (!previewUrl) return null;
        return {
          previewUrl,
          name: part.name,
        };
      }

      return null;
    })
    .filter(Boolean);

  return { text, images };
}

export function parseFileBlocks(text) {
  const structured = parseStructuredFileBlocks(text);
  const legacy = parseLegacyFileBlocks(structured.text);
  const pathLike = parsePathLikeFileReferences(legacy.text);

  return {
    text: pathLike.text.replace(/\n{3,}/g, '\n\n').trim(),
    files: [...structured.files, ...legacy.files, ...pathLike.files],
  };
}
