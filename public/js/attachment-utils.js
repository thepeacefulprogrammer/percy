const TEXT_FILE_EXTENSIONS = new Set([
  'txt', 'md', 'markdown', 'json', 'js', 'cjs', 'mjs', 'ts', 'tsx', 'jsx', 'css', 'scss', 'less',
  'html', 'htm', 'xml', 'yml', 'yaml', 'csv', 'py', 'sh', 'bash', 'zsh', 'java', 'kt', 'go', 'rs',
  'rb', 'php', 'sql', 'toml', 'ini', 'conf', 'log', 'env', 'gitignore', 'dockerfile',
]);

const DOCLING_FILE_EXTENSIONS = new Set([
  'pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'odt', 'ods', 'odp', 'rtf', 'epub',
]);

function getFileExtension(name) {
  const value = String(name || '');
  return value.includes('.') ? value.split('.').pop().toLowerCase() : value.toLowerCase();
}

export function isTextLikeFile(file) {
  if (!file) return false;
  if (typeof file.type === 'string' && (
    file.type.startsWith('text/')
    || file.type.includes('json')
    || file.type.includes('javascript')
    || file.type.includes('typescript')
    || file.type.includes('xml')
    || file.type.includes('yaml')
    || file.type.includes('csv')
    || file.type.includes('shellscript')
  )) {
    return true;
  }

  return TEXT_FILE_EXTENSIONS.has(getFileExtension(file.name));
}

export function isDoclingLikeFile(file) {
  if (!file) return false;

  const mimeType = String(file.type || '').toLowerCase();
  if (
    mimeType.includes('pdf')
    || mimeType.includes('word')
    || mimeType.includes('officedocument')
    || mimeType.includes('msword')
    || mimeType.includes('powerpoint')
    || mimeType.includes('presentation')
    || mimeType.includes('excel')
    || mimeType.includes('spreadsheet')
    || mimeType.includes('opendocument')
    || mimeType.includes('rtf')
    || mimeType.includes('epub')
  ) {
    return true;
  }

  return DOCLING_FILE_EXTENSIONS.has(getFileExtension(file.name));
}

export function shouldShowDoclingStatus(file) {
  return isDoclingLikeFile(file) && !isTextLikeFile(file) && !file.type.startsWith('image/');
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function dataUrlToBase64(dataUrl) {
  const value = String(dataUrl || '');
  const commaIndex = value.indexOf(',');
  return commaIndex >= 0 ? value.slice(commaIndex + 1) : value;
}

async function parseDocumentAttachment(file, api, makeId) {
  const dataUrl = await readFileAsDataUrl(file);
  const response = await api('/api/parse-document', {
    method: 'POST',
    body: JSON.stringify({
      name: file.name,
      size: file.size,
      mimeType: file.type || 'application/octet-stream',
      data: dataUrlToBase64(dataUrl),
    }),
  });

  return {
    id: makeId(),
    kind: 'document',
    name: file.name,
    size: file.size,
    mimeType: file.type || 'application/octet-stream',
    content: String(response.content || '').trim(),
    parser: response.engine || 'docling',
  };
}

export async function fileToAttachment(file, api, makeId) {
  if (file.type.startsWith('image/')) {
    const dataUrl = await readFileAsDataUrl(file);

    return {
      id: makeId(),
      kind: 'image',
      name: file.name,
      size: file.size,
      mimeType: file.type || 'image/png',
      data: dataUrlToBase64(dataUrl),
      previewUrl: dataUrl,
    };
  }

  if (isTextLikeFile(file)) {
    return {
      id: makeId(),
      kind: 'text',
      name: file.name,
      size: file.size,
      mimeType: file.type || 'text/plain',
      content: await file.text(),
    };
  }

  if (isDoclingLikeFile(file)) {
    return parseDocumentAttachment(file, api, makeId);
  }

  throw new Error(`Unsupported attachment type: ${file.name}`);
}
