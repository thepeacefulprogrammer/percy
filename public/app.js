import {
  extractMessageParts,
  formatBytes,
  parseFileBlocks,
  renderMarkdown,
  serializeFileAttachments,
} from './js/message-format.js';
import {
  fileToAttachment,
  shouldShowDoclingStatus,
} from './js/attachment-utils.js';

const messagesEl = document.getElementById('messages');
const statusBarEl = document.getElementById('statusBar');
const formEl = document.getElementById('chatForm');
const inputEl = document.getElementById('messageInput');
const attachmentInputEl = document.getElementById('attachmentInput');
const attachmentListEl = document.getElementById('attachmentList');
const attachBtnEl = document.getElementById('attachBtn');
const composerAbortBtnEl = document.getElementById('composerAbortBtn');
const composerNewSessionBtnEl = document.getElementById('composerNewSessionBtn');
const themeToggleBtnEl = document.getElementById('themeToggleBtn');

const THEME_STORAGE_KEY = 'percy-web-chat-theme';
const THEMES = ['percy', 'c64', 'modern', 'futuristic'];
const THEME_LABELS = {
  percy: 'Monochrome',
  c64: 'C64',
  modern: 'Modern',
  futuristic: 'Futuristic',
};

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
let startupLoaded = false;
let startupErrorShown = false;
let startupRetryTimer = null;
let eventsConnected = false;
let promptRequestInFlight = false;
let eventSource = null;

function getStoredTheme() {
  try {
    const storedTheme = globalThis.localStorage?.getItem(THEME_STORAGE_KEY);
    return THEMES.includes(storedTheme) ? storedTheme : 'percy';
  } catch {
    return 'percy';
  }
}

function getNextTheme(theme = currentTheme) {
  const index = THEMES.indexOf(theme);
  return THEMES[(index + 1 + THEMES.length) % THEMES.length] || THEMES[0];
}

function updateThemeToggleButton() {
  const nextTheme = getNextTheme();
  const label = `Switch to ${THEME_LABELS[nextTheme] || nextTheme} theme`;
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
  const nextTheme = getNextTheme();
  applyTheme(nextTheme);
  setStatus(`Theme: ${THEME_LABELS[nextTheme] || nextTheme}`);
}

async function registerServiceWorker() {
  if (!('serviceWorker' in globalThis.navigator)) return;

  try {
    await globalThis.navigator.serviceWorker.register('/sw.js');
  } catch (error) {
    console.warn('Service worker registration failed:', error);
  }
}

function makeId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `att-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
      if (file.kind === 'document') detailParts.push('Parsed document');
      if (file.parser) detailParts.push(`via ${file.parser}`);
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
  resetConversationView();

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
  const queue = state.pendingMessageCount ? ` · queued: ${state.pendingMessageCount}` : '';
  if (state.isStreaming) {
    setStatus(`Percy is responding${queue}`);
  } else if (promptRequestInFlight) {
    setStatus('Sending…');
  } else {
    setStatus(`Ready${queue}`);
  }

  composerAbortBtnEl.disabled = !state.isStreaming;
  attachBtnEl.disabled = promptRequestInFlight;
  composerNewSessionBtnEl.disabled = promptRequestInFlight;
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

function resetLiveAssistant() {
  liveAssistantEl = null;
  liveAssistantText = '';
}

function resetConversationView() {
  messagesEl.innerHTML = '';
  resetLiveAssistant();
}

function setPromptRequestState(isInFlight) {
  promptRequestInFlight = isInFlight;
  updateUiState();
}

function focusComposer() {
  autoResizeInput();
  inputEl.focus();
}

async function refreshStateFromServer() {
  try {
    const data = await api('/api/state');
    state = { ...state, ...data.state };
    updateUiState();
  } catch {}
}

function finalizeAssistantMessage(message) {
  const { text, images } = extractMessageParts(message);
  if (!text.trim() && !images.length) return;

  liveAssistantText = text;
  const el = liveAssistantEl || addMessage('assistant', { text, images });
  setMessageContent(el, { text, images });
  resetLiveAssistant();
}

function addUserDraftMessage(draftText, draftAttachments) {
  addMessage('user', {
    text: draftText.trim(),
    images: draftAttachments.filter((item) => item.kind === 'image'),
    files: draftAttachments.filter((item) => item.kind === 'text' || item.kind === 'document'),
  });
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
  startupLoaded = true;
  startupErrorShown = false;
}

function scheduleStartupRetry() {
  if (startupLoaded || startupRetryTimer) return;
  startupRetryTimer = globalThis.setTimeout(() => {
    startupRetryTimer = null;
    bootstrap();
  }, 2000);
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
    const info = [
      attachment.kind === 'image'
        ? 'Image'
        : attachment.kind === 'document'
          ? 'Parsed document'
          : 'Text file',
    ];
    if (attachment.parser) info.push(`via ${attachment.parser}`);
    if (attachment.size) info.push(formatBytes(attachment.size));
    if (attachment.content && attachment.kind !== 'image') info.push(`${attachment.content.length.toLocaleString()} chars`);
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
      if (shouldShowDoclingStatus(file)) {
        setStatus(`Parsing ${file.name} with Docling…`);
      }
      nextAttachments.push(await fileToAttachment(file, api, makeId));
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
  const files = [];

  for (const attachment of attachments) {
    if (attachment.kind === 'image') {
      images.push({
        type: 'image',
        data: attachment.data,
        mimeType: attachment.mimeType,
      });
      continue;
    }

    if (attachment.kind === 'text' || attachment.kind === 'document') {
      files.push(attachment);
    }
  }

  if (files.length) {
    chunks.push(`<percy-files>${serializeFileAttachments(files)}</percy-files>`);
  }

  if (!text && !files.length && images.length) {
    chunks.push(`[Attached ${images.length} image${images.length === 1 ? '' : 's'}]`);
  }

  return {
    message: chunks.join('\n\n').trim(),
    images,
  };
}

function parseEventPayload(event) {
  try {
    return JSON.parse(event.data);
  } catch (error) {
    console.warn('Invalid SSE payload:', error, event.data);
    return null;
  }
}

async function handleServerEvent(payload) {
  if (payload.type === 'hello' && payload.state) {
    state = { ...state, ...payload.state };
    updateUiState();
    if (!startupLoaded) {
      bootstrap();
    }
    return;
  }

  if (payload.type === 'new_session') {
    resetConversationView();
    clearPendingAttachments();
    setStatus('Started a new chat');
    await loadInitialData();
    return;
  }

  if (payload.type === 'rpc_exit') {
    setStatus('Percy RPC restarting…');
    return;
  }

  if (payload.type === 'rpc_timeout') {
    setStatus('Percy RPC timed out. Restarting…');
    return;
  }

  if (payload.type === 'server_restarting') {
    setStatus('Restarting Percy…');
  }
}

function handleRpcEvent(payload) {
  if (payload.type === 'agent_start') {
    state.isStreaming = true;
    updateUiState();
    return;
  }

  if (payload.type === 'agent_settled') {
    state.isStreaming = false;
    resetLiveAssistant();
    refreshStateFromServer();
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
    finalizeAssistantMessage(payload.message);
    return;
  }

  if (payload.type === 'tool_execution_start') {
    setStatus(`Running tool: ${payload.toolName}`);
    return;
  }

  if (payload.type === 'tool_execution_end') {
    updateUiState();
  }
}

function connectEvents() {
  if (eventSource) return eventSource;

  const es = new EventSource('/api/events');
  eventSource = es;

  es.onopen = () => {
    if (!startupLoaded) {
      setStatus('Connected. Loading…');
      return;
    }
    updateUiState();
  };

  es.addEventListener('server', async (event) => {
    const payload = parseEventPayload(event);
    if (!payload) return;
    await handleServerEvent(payload);
  });

  es.addEventListener('rpc', (event) => {
    const payload = parseEventPayload(event);
    if (!payload) return;
    handleRpcEvent(payload);
  });

  es.onerror = () => {
    setStatus('Connection lost. Retrying…');
  };

  return es;
}

attachBtnEl.addEventListener('click', () => {
  attachmentInputEl.click();
});

attachmentInputEl.addEventListener('change', async (event) => {
  const files = Array.from(event.target.files || []);
  await addAttachments(files);
  attachmentInputEl.value = '';
  focusComposer();
});

formEl.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (promptRequestInFlight) {
    setStatus('Message already sending…');
    return;
  }

  const draftText = inputEl.value;
  const draftAttachments = [...pendingAttachments];
  const payload = buildPromptPayload(draftText, draftAttachments);
  if (!payload.message) return;

  addUserDraftMessage(draftText, draftAttachments);

  inputEl.value = '';
  clearPendingAttachments();
  focusComposer();
  setPromptRequestState(true);

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
  } finally {
    setPromptRequestState(false);
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
  if (promptRequestInFlight) return;
  if (!confirm('Start a new Percy chat session?')) return;
  try {
    await api('/api/new-session', { method: 'POST', body: '{}' });
  } catch (error) {
    addMessage('system', { text: `Error: ${error.message}` });
  }
});

themeToggleBtnEl.addEventListener('click', toggleTheme);

applyTheme(getStoredTheme(), { persist: false });

function ensureEventConnection() {
  if (eventsConnected) return;
  connectEvents();
  eventsConnected = true;
}

async function bootstrap() {
  try {
    await loadInitialData();
  } catch (error) {
    startupLoaded = false;
    if (!startupErrorShown) {
      addMessage('system', { text: `Startup error: ${error.message}. Retrying…` });
      startupErrorShown = true;
    }
    setStatus('Startup error. Retrying…');
    scheduleStartupRetry();
  } finally {
    updateUiState();
    ensureEventConnection();
    focusComposer();
    registerServiceWorker();
  }
}

bootstrap();
