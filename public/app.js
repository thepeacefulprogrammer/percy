const messagesEl = document.getElementById('messages');
const statusBarEl = document.getElementById('statusBar');
const subtitleEl = document.getElementById('subtitle');
const formEl = document.getElementById('chatForm');
const inputEl = document.getElementById('messageInput');
const attachmentInputEl = document.getElementById('attachmentInput');
const attachmentListEl = document.getElementById('attachmentList');
const attachBtnEl = document.getElementById('attachBtn');
const composerAbortBtnEl = document.getElementById('composerAbortBtn');
const composerNewSessionBtnEl = document.getElementById('composerNewSessionBtn');
const themeToggleBtnEl = document.getElementById('themeToggleBtn');

const THEME_STORAGE_KEY = 'percy-web-chat-theme';
const THEMES = ['percy', 'c64'];

const TEXT_FILE_EXTENSIONS = new Set([
  'txt', 'md', 'markdown', 'json', 'js', 'cjs', 'mjs', 'ts', 'tsx', 'jsx', 'css', 'scss', 'less',
  'html', 'htm', 'xml', 'yml', 'yaml', 'csv', 'py', 'sh', 'bash', 'zsh', 'java', 'kt', 'go', 'rs',
  'rb', 'php', 'sql', 'toml', 'ini', 'conf', 'log', 'env', 'gitignore', 'dockerfile',
]);

let state = {
  isStreaming: false,
  pendingMessageCount: 0,
  model: null,
  sessionName: 'Web Chat',
};
let liveAssistantEl = null;
let liveAssistantText = '';
let pendingAttachments = [];
let currentTheme = 'percy';

function getStoredTheme() {
  try {
    const storedTheme = globalThis.localStorage?.getItem(THEME_STORAGE_KEY);
    return THEMES.includes(storedTheme) ? storedTheme : 'percy';
  } catch {
    return 'percy';
  }
}

function updateThemeToggleButton() {
  const nextTheme = currentTheme === 'percy' ? 'c64' : 'percy';
  const label = nextTheme === 'c64' ? 'Switch to C64 theme' : 'Switch to Monochrome theme';
  themeToggleBtnEl.title = label;
  themeToggleBtnEl.setAttribute('aria-label', label);
}

function applyTheme(theme, options = {}) {
  currentTheme = THEMES.includes(theme) ? theme : 'percy';
  document.body.dataset.theme = currentTheme;
  updateThemeToggleButton();

  if (options.persist === false) return;

  try {
    globalThis.localStorage?.setItem(THEME_STORAGE_KEY, currentTheme);
  } catch {}
}

function toggleTheme() {
  const nextTheme = currentTheme === 'percy' ? 'c64' : 'percy';
  applyTheme(nextTheme);
  setStatus(`Theme: ${nextTheme === 'c64' ? 'C64' : 'Monochrome'}`);
}

function escapeHtml(text) {
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

function renderMarkdown(text) {
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

function makeId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `att-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function escapeFileTagValue(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function formatBytes(bytes) {
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

function extractMessageParts(message) {
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

function parseFileBlocks(text) {
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

  return {
    text: cleaned.replace(/\n{3,}/g, '\n\n').trim(),
    files,
  };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function autoResizeInput() {
  inputEl.style.height = 'auto';

  const maxHeight = parseFloat(globalThis.getComputedStyle(inputEl).maxHeight);
  const nextHeight = Number.isFinite(maxHeight)
    ? Math.min(inputEl.scrollHeight, maxHeight)
    : inputEl.scrollHeight;

  inputEl.style.height = `${Math.max(nextHeight, 104)}px`;
  inputEl.style.overflowY = inputEl.scrollHeight > nextHeight ? 'auto' : 'hidden';
}

async function copyTextToClipboard(text) {
  const value = String(text || '');
  if (!value) return;

  if (globalThis.navigator?.clipboard?.writeText) {
    await globalThis.navigator.clipboard.writeText(value);
    return;
  }

  const tempEl = document.createElement('textarea');
  tempEl.value = value;
  tempEl.setAttribute('readonly', '');
  tempEl.style.position = 'absolute';
  tempEl.style.left = '-9999px';
  document.body.appendChild(tempEl);
  tempEl.select();
  document.execCommand('copy');
  document.body.removeChild(tempEl);
}

async function handleCopyMessage(wrapper, buttonEl) {
  const bodyEl = wrapper.querySelector('.body');
  const text = bodyEl?.textContent || '';
  if (!text.trim()) return;

  try {
    await copyTextToClipboard(text);
    const previousLabel = buttonEl.textContent;
    const previousTitle = buttonEl.title;
    buttonEl.textContent = '✓';
    buttonEl.title = 'Copied';
    buttonEl.setAttribute('aria-label', 'Copied');
    globalThis.setTimeout(() => {
      buttonEl.textContent = previousLabel;
      buttonEl.title = previousTitle;
      buttonEl.setAttribute('aria-label', previousTitle);
    }, 1200);
  } catch (error) {
    addMessage('system', { text: `Copy failed: ${error.message}` });
  }
}

function renderMessageBody(container, payload = {}) {
  const bodyEl = document.createElement('div');
  bodyEl.className = 'body';
  bodyEl.innerHTML = renderMarkdown(payload.text || '');
  container.appendChild(bodyEl);

  if (Array.isArray(payload.images) && payload.images.length) {
    const galleryEl = document.createElement('div');
    galleryEl.className = 'message-images';

    for (const image of payload.images) {
      const src = image.previewUrl || (image.data && image.mimeType ? `data:${image.mimeType};base64,${image.data}` : '');
      if (!src) continue;
      const imgEl = document.createElement('img');
      imgEl.alt = image.name || 'Attached image';
      imgEl.loading = 'lazy';
      imgEl.src = src;
      galleryEl.appendChild(imgEl);
    }

    if (galleryEl.childElementCount) {
      container.appendChild(galleryEl);
    }
  }

  const visibleFiles = Array.isArray(payload.files)
    ? payload.files.filter((file) => file && (file.kind === 'text' || file.content))
    : [];

  if (visibleFiles.length) {
    const filesEl = document.createElement('div');
    filesEl.className = 'message-files';

    for (const file of visibleFiles) {
      const chipEl = document.createElement('div');
      chipEl.className = 'file-chip';

      const metaEl = document.createElement('div');
      const nameEl = document.createElement('strong');
      nameEl.textContent = file.name || 'Attachment';
      metaEl.appendChild(nameEl);

      const detailEl = document.createElement('small');
      const detailParts = [];
      if (file.kind === 'text') detailParts.push('Text attachment');
      if (file.size) detailParts.push(formatBytes(file.size));
      if (file.content) detailParts.push(`${file.content.length.toLocaleString()} chars`);
      detailEl.textContent = detailParts.join(' · ') || 'Attachment';
      metaEl.appendChild(detailEl);

      chipEl.appendChild(metaEl);
      filesEl.appendChild(chipEl);
    }

    container.appendChild(filesEl);
  }

  return bodyEl;
}

function setMessageContent(wrapper, payload = {}) {
  wrapper.querySelector('.body')?.remove();
  wrapper.querySelector('.message-images')?.remove();
  wrapper.querySelector('.message-files')?.remove();

  const copyButtonEl = wrapper.querySelector('.message-copy-button');
  if (!copyButtonEl) {
    renderMessageBody(wrapper, payload);
    return;
  }

  const contentEl = document.createElement('div');
  renderMessageBody(contentEl, payload);
  wrapper.insertBefore(contentEl.querySelector('.body'), copyButtonEl);
  const galleryEl = contentEl.querySelector('.message-images');
  if (galleryEl) wrapper.insertBefore(galleryEl, copyButtonEl);
  const filesEl = contentEl.querySelector('.message-files');
  if (filesEl) wrapper.insertBefore(filesEl, copyButtonEl);
}

function addMessage(role, payload = {}) {
  const normalizedPayload = typeof payload === 'string' ? { text: payload } : payload;
  const wrapper = document.createElement('article');
  wrapper.className = `message ${role}`;

  const roleEl = document.createElement('span');
  roleEl.className = 'role';
  roleEl.textContent = role === 'assistant' ? 'Percy' : role === 'user' ? 'Randy' : role;
  wrapper.appendChild(roleEl);

  setMessageContent(wrapper, normalizedPayload);

  const copyBtnEl = document.createElement('button');
  copyBtnEl.className = 'message-copy-button';
  copyBtnEl.type = 'button';
  copyBtnEl.textContent = '⧉';
  copyBtnEl.title = 'Copy message';
  copyBtnEl.setAttribute('aria-label', 'Copy message');
  copyBtnEl.addEventListener('click', () => {
    handleCopyMessage(wrapper, copyBtnEl);
  });
  wrapper.appendChild(copyBtnEl);

  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function renderHistory(messages) {
  messagesEl.innerHTML = '';
  liveAssistantEl = null;
  liveAssistantText = '';

  for (const message of messages) {
    if (message.role === 'user' || message.role === 'assistant') {
      const { text, images } = extractMessageParts(message);
      const parsed = parseFileBlocks(text);
      if (parsed.text.trim() || images.length || parsed.files.length) {
        addMessage(message.role, {
          text: parsed.text,
          images,
          files: parsed.files,
        });
      }
    }
  }
}

function setStatus(text) {
  statusBarEl.textContent = text;
}

function updateUiState() {
  const modelName = state.model?.name || state.model?.id || 'default model';
  const queue = state.pendingMessageCount ? ` · queued: ${state.pendingMessageCount}` : '';
  subtitleEl.textContent = `${state.sessionName || 'Web Chat'} · ${modelName}`;
  setStatus(state.isStreaming ? `Percy is responding${queue}` : `Ready${queue}`);
  composerAbortBtnEl.disabled = !state.isStreaming;
}

function ensureLiveAssistant() {
  if (!liveAssistantEl) {
    liveAssistantText = '';
    liveAssistantEl = addMessage('assistant', { text: '' });
  }
  return liveAssistantEl;
}

function updateLiveAssistant(text) {
  const el = ensureLiveAssistant();
  setMessageContent(el, { text });
  scrollToBottom();
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

async function loadInitialData() {
  const [{ state: freshState }, { messages }] = await Promise.all([
    api('/api/state'),
    api('/api/messages'),
  ]);
  state = { ...state, ...freshState };
  renderHistory(messages);
  updateUiState();
}

function isTextLikeFile(file) {
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

  const extension = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : file.name.toLowerCase();
  return TEXT_FILE_EXTENSIONS.has(extension);
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function fileToAttachment(file) {
  if (file.type.startsWith('image/')) {
    const dataUrl = await readFileAsDataUrl(file);
    const commaIndex = dataUrl.indexOf(',');
    const base64 = commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;

    return {
      id: makeId(),
      kind: 'image',
      name: file.name,
      size: file.size,
      mimeType: file.type || 'image/png',
      data: base64,
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

  throw new Error(`Unsupported attachment type: ${file.name}`);
}

function renderPendingAttachments() {
  attachmentListEl.innerHTML = '';

  for (const attachment of pendingAttachments) {
    const chipEl = document.createElement('div');
    chipEl.className = 'file-chip';

    const leftEl = document.createElement('div');
    leftEl.style.display = 'flex';
    leftEl.style.alignItems = 'center';
    leftEl.style.gap = '10px';

    if (attachment.kind === 'image') {
      const thumbEl = document.createElement('img');
      thumbEl.className = 'attachment-thumb';
      thumbEl.alt = attachment.name;
      thumbEl.src = attachment.previewUrl || `data:${attachment.mimeType};base64,${attachment.data}`;
      leftEl.appendChild(thumbEl);
    }

    const metaEl = document.createElement('div');
    const nameEl = document.createElement('strong');
    nameEl.textContent = attachment.name;
    metaEl.appendChild(nameEl);

    const infoEl = document.createElement('small');
    const info = [attachment.kind === 'image' ? 'Image' : 'Text file'];
    if (attachment.size) info.push(formatBytes(attachment.size));
    infoEl.textContent = info.join(' · ');
    metaEl.appendChild(infoEl);

    leftEl.appendChild(metaEl);
    chipEl.appendChild(leftEl);

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'secondary';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => {
      pendingAttachments = pendingAttachments.filter((item) => item.id !== attachment.id);
      renderPendingAttachments();
    });
    chipEl.appendChild(removeBtn);

    attachmentListEl.appendChild(chipEl);
  }
}

function clearPendingAttachments() {
  pendingAttachments = [];
  attachmentInputEl.value = '';
  renderPendingAttachments();
}

async function addAttachments(files) {
  if (!files.length) return;

  const nextAttachments = [];

  for (const file of files) {
    try {
      nextAttachments.push(await fileToAttachment(file));
    } catch (error) {
      addMessage('system', { text: `Attachment error: ${error.message}` });
    }
  }

  if (!nextAttachments.length) return;

  pendingAttachments = [...pendingAttachments, ...nextAttachments];
  renderPendingAttachments();
  setStatus(`Attached ${nextAttachments.length} file${nextAttachments.length === 1 ? '' : 's'}`);
}

function buildPromptPayload(messageText, attachments) {
  const chunks = [];
  const text = String(messageText || '').trim();
  if (text) chunks.push(text);

  const images = [];

  for (const attachment of attachments) {
    if (attachment.kind === 'image') {
      images.push({
        type: 'image',
        data: attachment.data,
        mimeType: attachment.mimeType,
      });
      chunks.push(`<file name="${escapeFileTagValue(attachment.name)}"></file>`);
      continue;
    }

    if (attachment.kind === 'text') {
      chunks.push(`<file name="${escapeFileTagValue(attachment.name)}">\n${attachment.content}\n</file>`);
    }
  }

  return {
    message: chunks.join('\n\n').trim(),
    images,
  };
}

function connectEvents() {
  const es = new EventSource('/api/events');

  es.addEventListener('server', async (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'hello' && payload.state) {
      state = { ...state, ...payload.state };
      updateUiState();
    }
    if (payload.type === 'new_session') {
      messagesEl.innerHTML = '';
      liveAssistantEl = null;
      liveAssistantText = '';
      clearPendingAttachments();
      setStatus('Started a new chat');
      await loadInitialData();
    }
    if (payload.type === 'rpc_exit') {
      setStatus('Percy RPC restarted…');
    }
  });

  es.addEventListener('rpc', (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === 'agent_start') {
      state.isStreaming = true;
      updateUiState();
      return;
    }

    if (payload.type === 'agent_settled') {
      state.isStreaming = false;
      liveAssistantEl = null;
      liveAssistantText = '';
      api('/api/state').then((data) => {
        state = { ...state, ...data.state };
        updateUiState();
      }).catch(() => {});
      return;
    }

    if (payload.type === 'queue_update') {
      if (typeof payload.pendingMessageCount === 'number') {
        state.pendingMessageCount = payload.pendingMessageCount;
        updateUiState();
      }
      return;
    }

    if (payload.type === 'message_update') {
      const delta = payload.assistantMessageEvent || {};
      if (delta.type === 'text_delta') {
        liveAssistantText += delta.delta || '';
        updateLiveAssistant(liveAssistantText);
      }
      return;
    }

    if (payload.type === 'message_end' && payload.message?.role === 'assistant') {
      const { text, images } = extractMessageParts(payload.message);
      if (text.trim() || images.length) {
        liveAssistantText = text;
        const el = liveAssistantEl || addMessage('assistant', { text, images });
        setMessageContent(el, { text, images });
        liveAssistantEl = null;
        liveAssistantText = '';
      }
      return;
    }

    if (payload.type === 'tool_execution_start') {
      setStatus(`Running tool: ${payload.toolName}`);
      return;
    }

    if (payload.type === 'tool_execution_end') {
      updateUiState();
    }
  });

  es.onerror = () => {
    setStatus('Connection lost. Retrying…');
  };
}

attachBtnEl.addEventListener('click', () => {
  attachmentInputEl.click();
});

attachmentInputEl.addEventListener('change', async (event) => {
  const files = Array.from(event.target.files || []);
  await addAttachments(files);
  attachmentInputEl.value = '';
  autoResizeInput();
  inputEl.focus();
});

formEl.addEventListener('submit', async (event) => {
  event.preventDefault();

  const draftText = inputEl.value;
  const draftAttachments = [...pendingAttachments];
  const payload = buildPromptPayload(draftText, draftAttachments);
  if (!payload.message) return;

  addMessage('user', {
    text: draftText.trim(),
    images: draftAttachments.filter((item) => item.kind === 'image'),
    files: draftAttachments.filter((item) => item.kind === 'text'),
  });

  inputEl.value = '';
  clearPendingAttachments();
  autoResizeInput();
  inputEl.focus();

  try {
    await api('/api/prompt', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (state.isStreaming) {
      setStatus('Message queued');
    }
  } catch (error) {
    addMessage('system', { text: `Error: ${error.message}` });
  }
});

inputEl.addEventListener('input', () => {
  autoResizeInput();
});

inputEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

globalThis.addEventListener('resize', autoResizeInput);

async function abortResponse() {
  try {
    await api('/api/abort', { method: 'POST', body: '{}' });
    setStatus('Abort sent');
  } catch (error) {
    addMessage('system', { text: `Error: ${error.message}` });
  }
}

composerAbortBtnEl.addEventListener('click', abortResponse);

composerNewSessionBtnEl.addEventListener('click', async () => {
  if (!confirm('Start a new Percy chat session?')) return;
  try {
    await api('/api/new-session', { method: 'POST', body: '{}' });
  } catch (error) {
    addMessage('system', { text: `Error: ${error.message}` });
  }
});

themeToggleBtnEl.addEventListener('click', toggleTheme);

applyTheme(getStoredTheme(), { persist: false });

loadInitialData().catch((error) => {
  addMessage('system', { text: `Startup error: ${error.message}` });
  setStatus('Startup error');
}).finally(() => {
  updateUiState();
  connectEvents();
  autoResizeInput();
  inputEl.focus();
});
