/* AI Doctor - chat interface. No build step: served as-is by nginx. */

const state = {
  chats: [],
  chatId: null,
  documents: [],
  streaming: false,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

const api = {
  async request(path, options = {}) {
    const response = await fetch(`/api${path}`, options);
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = (await response.json()).detail || detail;
      } catch { /* response had no JSON body */ }
      throw new Error(detail);
    }
    return response.status === 204 ? null : response.json();
  },
  get: (path) => api.request(path),
  post: (path, body) =>
    api.request(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    }),
  patch: (path, body) =>
    api.request(path, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  remove: (path) => api.request(path, { method: 'DELETE' }),
};

/* ------------------------------------------------------------ rendering */

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* Deliberately small: escape everything first, then re-introduce the handful
   of constructs the answers actually use - bold, code, citations, headings,
   both list kinds, and tables for comparisons. Nothing here can inject markup. */
const inlineMd = (s) =>
  s
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(\d+)\]/g, '<span class="cite" data-cite="$1">$1</span>');

function renderRun(kind, lines) {
  if (kind === 'ul') {
    return `<ul>${lines
      .map((l) => `<li>${inlineMd(l.replace(/^[-*]\s+/, ''))}</li>`)
      .join('')}</ul>`;
  }
  if (kind === 'ol') {
    return `<ol>${lines
      .map((l) => `<li>${inlineMd(l.replace(/^\d+[.)]\s+/, ''))}</li>`)
      .join('')}</ol>`;
  }
  if (kind === 'tr' && lines.length >= 2 && /^\|[\s:|-]+\|$/.test(lines[1])) {
    const cells = (line) =>
      line.replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => inlineMd(c.trim()));
    const head = cells(lines[0]);
    const rows = lines.slice(2).map(cells);
    return `<div class="table-wrap"><table><thead><tr>${head
      .map((c) => `<th>${c}</th>`)
      .join('')}</tr></thead><tbody>${rows
      .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`)
      .join('')}</tbody></table></div>`;
  }
  return `<p>${lines.map(inlineMd).join(' ')}</p>`;
}

function renderMarkdown(text) {
  const kindOf = (line) =>
    /^#{1,6}\s+/.test(line) ? 'h'
    : /^[-*]\s+/.test(line) ? 'ul'
    : /^\d+[.)]\s+/.test(line) ? 'ol'
    : /^\|.*\|$/.test(line) ? 'tr'
    : 'p';

  return escapeHtml(text)
    .split(/\n{2,}/)
    .map((block) => {
      const lines = block.split('\n').map((l) => l.trim()).filter(Boolean);
      const parts = [];
      let run = [];
      let kind = null;
      const flush = () => {
        if (run.length) parts.push(renderRun(kind, run));
        run = [];
      };
      for (const line of lines) {
        const k = kindOf(line);
        if (k === 'h') {
          flush();
          kind = null;
          const m = line.match(/^(#{1,6})\s+(.*)$/);
          parts.push(`<h${Math.min(m[1].length + 1, 5)}>${inlineMd(m[2])}</h${Math.min(m[1].length + 1, 5)}>`);
          continue;
        }
        if (k !== kind) {
          flush();
          kind = k;
        }
        run.push(line);
      }
      flush();
      return parts.join('');
    })
    .join('');
}

function renderSources(citations) {
  if (!citations || !citations.length) return '';
  const items = citations
    .map(
      (c) => `
      <div class="source" data-source="${c.index}">
        <div class="source-head">
          <span class="source-index">${c.index}</span>
          <span class="source-name">${escapeHtml(c.source)}</span>
          <span class="source-page">${escapeHtml(c.pages || pageLabel(c))}</span>
        </div>
        <div class="source-body" hidden>${escapeHtml(c.excerpt || '')}</div>
      </div>`)
    .join('');
  return `<div class="sources"><div class="sources-head">Sources</div>${items}</div>`;
}

const pageLabel = (c) =>
  c.page_start === c.page_end ? `p.${c.page_start}` : `pp.${c.page_start}-${c.page_end}`;

function messageElement(role, html, citations) {
  const turn = document.createElement('div');
  turn.className = `turn ${role}`;
  turn.innerHTML =
    role === 'user'
      ? `<div class="bubble">${html}</div>`
      : `<div class="answer">${html}</div>${renderSources(citations)}`;
  return turn;
}

function scrollDown() {
  const box = $('messages');
  box.scrollTop = box.scrollHeight;
}

/* ---------------------------------------------------------------- chats */

async function loadChats() {
  const { chats } = await api.get('/chats');
  state.chats = chats;
  const list = $('chat-list');
  list.innerHTML = '';

  chats.forEach((chat) => {
    const item = document.createElement('div');
    item.className = `chat-item${chat.id === state.chatId ? ' active' : ''}`;
    item.innerHTML = `<span>${escapeHtml(chat.title)}</span>
      <button class="delete icon-button" title="Delete">
        <svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13"/></svg>
      </button>`;
    // The whole row, not only the title text.
    item.onclick = () => openChat(chat.id);
    item.querySelector('.delete').onclick = async (event) => {
      event.stopPropagation();
      await api.remove(`/chats/${chat.id}`);
      if (state.chatId === chat.id) {
        state.chatId = null;
        showEmptyState();
      }
      loadChats();
    };
    list.appendChild(item);
  });
}

function showEmptyState() {
  $('messages').innerHTML = '';
  $('chat-title').textContent = 'AI Doctor';
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.innerHTML = `
    <div class="empty-icon">🩺</div>
    <h2>Ask about your documents</h2>
    <p>Every answer cites the document and page it came from.</p>`;
  $('messages').appendChild(empty);
}

async function openChat(chatId) {
  state.chatId = chatId;
  const chat = state.chats.find((c) => c.id === chatId);
  $('chat-title').textContent = chat ? chat.title : 'Chat';

  const { messages } = await api.get(`/chats/${chatId}/messages`);
  const box = $('messages');
  box.innerHTML = '';
  messages.forEach((message) => {
    const html =
      message.role === 'user'
        ? escapeHtml(message.content)
        : renderMarkdown(message.content);
    box.appendChild(messageElement(message.role, html, message.citations));
  });
  scrollDown();
  loadChats();
}

async function ensureChat() {
  if (state.chatId) return state.chatId;
  const chat = await api.post('/chats');
  state.chatId = chat.id;
  await loadChats();
  return chat.id;
}

/* ------------------------------------------------------------- questions */

async function ask(question) {
  if (state.streaming || !question.trim()) return;
  const chatId = await ensureChat();

  const box = $('messages');
  box.querySelector('.empty')?.remove();
  box.appendChild(messageElement('user', escapeHtml(question)));

  const turn = document.createElement('div');
  turn.className = 'turn assistant';
  turn.innerHTML = '<div class="status"><span class="dot"></span><span>Thinking…</span></div>';
  box.appendChild(turn);
  scrollDown();

  state.streaming = true;
  $('send').disabled = true;

  let answer = '';
  let citations = [];
  const trace = { members: [], findings: [], review: null, reasoning: null, suggestions: [] };

  const traceHtml = () => {
    if (!trace.members.length) return '';
    const members = trace.members
      .map((m) => {
        const finding = trace.findings.find((f) => f.specialist === m.key);
        const mark = !finding ? '<span class="spin"></span>'
          : finding.found ? '<span class="tick">✓</span>'
          : '<span class="none">—</span>';
        const note = finding
          ? (finding.found ? `${finding.passages} passages` : 'nothing relevant')
          : 'working…';
        return `<li>${mark}<strong>${escapeHtml(m.title)}</strong>
                  <span class="brief">${escapeHtml(m.brief)}</span>
                  <span class="note">${note}</span></li>`;
      })
      .join('');
    const reasoning = trace.reasoning
      ? `<div class="reasoning"><strong>Bedside reasoning</strong>
          ${trace.reasoning.findings ? `<p>${escapeHtml(trace.reasoning.findings)}</p>` : ''}
          <ul>${(trace.reasoning.differential || [])
            .map((d) => `<li><strong>${escapeHtml(d.diagnosis)}</strong>${
              d.for ? ` — ${escapeHtml(d.for)}` : ''}${
              d.against ? `<span class="against"> · against: ${escapeHtml(d.against)}</span>` : ''}</li>`)
            .join('')}</ul>
          ${trace.reasoning.leading
            ? `<p class="leading">Leading diagnosis: <strong>${escapeHtml(trace.reasoning.leading)}</strong></p>`
            : ''}</div>`
      : '';
    const problems = trace.review?.problems?.length
      ? `<div class="review"><strong>Review found:</strong><ul>${
          trace.review.problems.map((p) => `<li>${escapeHtml(p)}</li>`).join('')
        }</ul></div>`
      : '';
    const followUp = trace.review?.follow_up?.length
      ? `<div class="review"><strong>Went back for:</strong><ul>${
          trace.review.follow_up.map((q) => `<li>${escapeHtml(q)}</li>`).join('')
        }</ul></div>`
      : '';
    return `<details class="trace"${state.streaming ? ' open' : ''}>
        <summary>Consultation · ${trace.members.length} specialist${trace.members.length > 1 ? 's' : ''}</summary>
        ${reasoning}<ul class="panel-members">${members}</ul>${problems}${followUp}
      </details>`;
  };

  const chipsHtml = () =>
    trace.suggestions.length
      ? `<div class="followups"><div class="followups-head">Keep going</div>${trace.suggestions
          .map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`)
          .join('')}</div>`
      : '';

  try {
    const response = await fetch(`/api/chats/${chatId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: question }),
    });
    if (!response.ok) throw new Error(`request failed (${response.status})`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const event = JSON.parse(line.slice(5).trim());

        if (event.type === 'status') {
          turn.innerHTML = traceHtml() +
            `<div class="status"><span class="dot"></span><span>${escapeHtml(event.text)}</span></div>`;
          scrollDown();
        } else if (event.type === 'team') {
          trace.members = event.members;
          turn.innerHTML = traceHtml() +
            '<div class="status"><span class="dot"></span><span>Consulting</span></div>';
          scrollDown();
        } else if (event.type === 'finding') {
          trace.findings.push(event);
          turn.innerHTML = traceHtml() +
            '<div class="status"><span class="dot"></span><span>Consulting</span></div>';
          scrollDown();
        } else if (event.type === 'reasoning') {
          trace.reasoning = event;
          turn.innerHTML = traceHtml() +
            '<div class="status"><span class="dot"></span><span>Working through the case</span></div>';
          scrollDown();
        } else if (event.type === 'review') {
          trace.review = event;
        } else if (event.type === 'citations') {
          citations = event.citations;
        } else if (event.type === 'token') {
          answer += event.text;
          turn.innerHTML = traceHtml() + `<div class="answer">${renderMarkdown(answer)}</div>`;
          scrollDown();
        } else if (event.type === 'answer') {
          // The backend stripped citation markers that pointed at nothing.
          answer = event.text;
        } else if (event.type === 'suggestions') {
          trace.suggestions = event.questions || [];
        } else if (event.type === 'error') {
          turn.innerHTML = traceHtml() + `<div class="error">${escapeHtml(event.text)}</div>`;
        } else if (event.type === 'done') {
          state.streaming = false; // collapses the trace in the final render
          turn.innerHTML = traceHtml() +
            `<div class="answer">${renderMarkdown(answer)}</div>${renderSources(citations)}` +
            chipsHtml();
        }
      }
    }
  } catch (error) {
    turn.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  } finally {
    state.streaming = false;
    $('send').disabled = false;
    scrollDown();
    loadChats();
  }
}

/* ------------------------------------------------------------- documents */

async function loadDocuments() {
  const { documents, indexing } = await api.get('/documents');
  state.documents = documents;

  const indexed = documents.filter((d) => d.status === 'indexed').length;
  $('doc-badge').textContent = documents.length ? `${indexed}/${documents.length}` : '';

  const list = $('doc-list');
  list.innerHTML = documents.length
    ? ''
    : '<p class="status-line">No documents yet. Add some above.</p>';

  documents.forEach((doc) => {
    const row = document.createElement('div');
    row.className = 'doc';
    const percent = Math.round((doc.progress || 0) * 100);
    const excluded = doc.status === 'excluded';
    row.innerHTML = `
      <div class="doc-top">
        <span class="doc-name">${escapeHtml(doc.filename)}</span>
        <span class="pill ${doc.status}">${doc.status}</span>
        <button class="row-action">${excluded ? 'Include' : 'Remove from index'}</button>
      </div>
      <div class="doc-meta">
        ${doc.pages || 0} pages · ${(doc.size_bytes / 1e6).toFixed(1)} MB
        ${doc.chunk_count ? ` · ${doc.chunk_count.toLocaleString()} passages` : ''}
        ${doc.year ? ` · ${doc.year}` : ''}
      </div>
      ${doc.status === 'indexing' ? `<div class="bar"><div style="width:${percent}%"></div></div>` : ''}
      ${doc.error ? `<div class="doc-error">${escapeHtml(doc.error)}</div>` : ''}`;
    row.querySelector('.row-action').onclick = async () => {
      // Never deletes the file. To remove a document entirely, delete it from
      // the knowledgebase folder yourself.
      if (!excluded && !confirm(
        `Remove "${doc.filename}" from the search index?\n\n` +
        `The file stays in your knowledgebase folder — this only drops it from ` +
        `search results. You can include it again at any time.`)) return;
      await api.post(`/documents/${doc.id}/${excluded ? 'include' : 'exclude'}`);
      loadDocuments();
      loadSettings();
    };
    list.appendChild(row);
  });

  $('index-button').disabled = indexing;
  $('index-button').textContent = indexing ? 'Indexing…' : 'Index new documents';

  // Poll only while there is something to watch.
  clearTimeout(state.pollTimer);
  const busy = indexing || documents.some((d) => d.status === 'indexing');
  if (busy) {
    state.pollTimer = setTimeout(loadDocuments, 1500);
  } else if (state.wasIndexing) {
    loadSettings(); // refresh the sidebar counts once indexing finishes
  }
  state.wasIndexing = busy;
}

async function uploadFiles(files) {
  if (!files.length) return;
  const form = new FormData();
  Array.from(files).forEach((file) => form.append('files', file));
  $('dropzone').classList.add('over');
  try {
    await fetch('/api/documents', { method: 'POST', body: form });
    await loadDocuments();
  } finally {
    $('dropzone').classList.remove('over');
  }
}

/* -------------------------------------------------------------- settings */

const PROVIDERS = {
  openai: { label: 'OpenAI', key: true },
  anthropic: { label: 'Claude', key: true },
  nvidia: { label: 'NVIDIA NIM', key: true },
  ollama: { label: 'Ollama', key: false },
};
const DEFAULT_EMBED = {
  openai: 'text-embedding-3-large',
  nvidia: 'nvidia/llama-3.2-nv-embedqa-1b-v2',
  ollama: 'nomic-embed-text',
};
const CUSTOM = '__custom__';
const label = (p) => PROVIDERS[p]?.label || p;

const settingsUi = {
  config: null,
  chatProvider: 'openai',
  embedProvider: 'openai',
  lists: { chat: {}, embedding: {} },    // kind -> provider -> models
  choice: { chat: {}, embedding: {} },   // kind -> provider -> model picked this session
  loading: {},
};

const KIND = {
  chat: { select: 'chat-model', custom: 'chat-model-custom', refresh: 'chat-refresh' },
  embedding: { select: 'embed-model', custom: 'embed-model-custom', refresh: 'embed-refresh' },
};

function setSettingsMessage(text, kind = '') {
  const el = $('settings-message');
  el.textContent = text;
  el.className = `settings-message${kind ? ` ${kind}` : ''}`;
}

function showSettingsTab(tab) {
  document.querySelectorAll('.settings-tabs button').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  document.querySelectorAll('.tab-pane').forEach((pane) => {
    pane.hidden = pane.dataset.pane !== tab;
  });
}

/* ------------------------------------------------------------ model lists */

function fillModels(kind, list, selected) {
  const { select: sid, custom: cid } = KIND[kind];
  const select = $(sid);
  const names = [...new Set(list || [])];
  if (selected && !names.includes(selected)) names.unshift(selected);
  select.innerHTML =
    names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('') +
    `<option value="${CUSTOM}">Other (type a name)…</option>`;
  select.value = names.length ? (selected || names[0]) : CUSTOM;
  $(cid).hidden = select.value !== CUSTOM;
}

function chosen(kind) {
  const { select, custom } = KIND[kind];
  return $(select).value === CUSTOM ? $(custom).value.trim() : $(select).value;
}

function currentProvider(kind) {
  return kind === 'chat' ? settingsUi.chatProvider : settingsUi.embedProvider;
}

/* Values typed in the API keys tab, so a key can be tried before saving. */
function trialBody(provider, kind) {
  const body = { provider, kind };
  if (provider === 'ollama') {
    body.host = $('ollama-host').value.trim();
  } else {
    body.api_key = $(`${provider}-key`).value.trim();
    if (provider === 'openai') body.base_url = $('openai-base').value.trim();
    if (provider === 'nvidia') {
      body.base_url = kind === 'embedding'
        ? ($('nvidia-embed-base').value.trim() || $('nvidia-base').value.trim())
        : $('nvidia-base').value.trim();
    }
  }
  return body;
}

function canReach(provider, kind = 'chat') {
  if (!PROVIDERS[provider].key) return true;
  if ($(`${provider}-key`).value.trim()) return true;
  if (settingsUi.config?.providers?.[provider]?.key_set) return true;
  if (provider === 'nvidia') {
    const url = kind === 'embedding'
      ? ($('nvidia-embed-base').value || $('nvidia-base').value)
      : $('nvidia-base').value;
    return Boolean(url.trim()) && !url.includes('api.nvidia.com');
  }
  return false;
}

async function fetchModels(provider, kind) {
  const id = `${kind}:${provider}`;
  if (!settingsUi.loading[id]) {
    settingsUi.loading[id] = api.post('/config/models', trialBody(provider, kind))
      .then(({ models }) => {
        settingsUi.lists[kind][provider] = models;
        return models;
      })
      .finally(() => { settingsUi.loading[id] = null; });
  }
  return settingsUi.loading[id];
}

async function refreshModels(kind, { quiet = false } = {}) {
  const provider = currentProvider(kind);
  const button = $(KIND[kind].refresh);
  button.disabled = true;
  button.textContent = 'Loading…';
  try {
    const models = await fetchModels(provider, kind);
    if (currentProvider(kind) !== provider) return;
    fillModels(kind, models, chosen(kind));
    if (!quiet) {
      setSettingsMessage(`${label(provider)}: ${models.length} ${kind === 'embedding' ? 'embedding ' : ''}model${models.length === 1 ? '' : 's'} available.`, 'ok');
    }
    if (kind === 'embedding' && !models.length) {
      setSettingsMessage(`${label(provider)} listed no embedding models. Type one with "Other…".`, 'bad');
    }
  } catch (error) {
    if (currentProvider(kind) === provider) setSettingsMessage(error.message, 'bad');
  } finally {
    button.disabled = false;
    button.textContent = 'Refresh';
  }
}

/* -------------------------------------------------------------- providers */

function selectProvider(kind, provider) {
  const previous = currentProvider(kind);
  if (previous) settingsUi.choice[kind][previous] = chosen(kind);
  if (kind === 'chat') settingsUi.chatProvider = provider;
  else settingsUi.embedProvider = provider;

  const group = kind === 'chat' ? 'chat-providers' : 'embed-providers';
  document.querySelectorAll(`#${group} button`).forEach((b) =>
    b.setAttribute('aria-checked', String(b.dataset.provider === provider)));

  const config = settingsUi.config;
  const saved = kind === 'chat'
    ? config.providers[provider]?.model
    : (config.embedding.provider === provider ? config.embedding.model : DEFAULT_EMBED[provider]);
  fillModels(kind, settingsUi.lists[kind][provider], settingsUi.choice[kind][provider] || saved);

  if (kind === 'chat') renderChatNote();
  else renderEmbedWarning();

  if (!canReach(provider, kind)) {
    setSettingsMessage(`Add your ${label(provider)} key in the API keys tab to list its models.`, 'bad');
  } else {
    setSettingsMessage('');
    if (!settingsUi.lists[kind][provider]) refreshModels(kind, { quiet: true });
  }
}

function renderChatNote() {
  const active = settingsUi.config.llm_provider;
  const p = settingsUi.chatProvider;
  $('chat-note').textContent = active === p
    ? `${label(p)} is writing the answers now.`
    : `Answers come from ${label(active)} now. Save to switch to ${label(p)}.`;
}

function embeddingChanged() {
  const e = settingsUi.config.embedding;
  return settingsUi.embedProvider !== e.provider || chosen('embedding') !== e.model;
}

function renderEmbedWarning() {
  const warn = $('embed-warning');
  const { index } = settingsUi.config;
  if (!embeddingChanged() || !chosen('embedding')) {
    warn.hidden = true;
    return;
  }
  const n = index.documents_total;
  warn.hidden = false;
  warn.textContent = n
    ? `Saving rebuilds the index: ${n} document${n === 1 ? '' : 's'} will be re-embedded with ` +
      `${label(settingsUi.embedProvider)} · ${chosen('embedding')}. Questions can't be answered until it finishes.`
    : `The index will be built with ${label(settingsUi.embedProvider)} · ${chosen('embedding')}.`;
}

function renderIndexCard() {
  const { embedding, index } = settingsUi.config;
  const state = index.indexing
    ? '<span class="pill indexing">indexing</span>'
    : index.documents_total && index.documents_indexed < index.documents_total
      ? '<span class="pill pending">needs indexing</span>'
      : '<span class="pill indexed">ready</span>';
  $('index-card').innerHTML = `
    <div class="index-top"><strong>Current index</strong>${state}</div>
    <div class="index-meta">
      ${escapeHtml(label(embedding.provider))} · ${escapeHtml(embedding.model)} ·
      ${embedding.dimensions} dimensions${embedding.shortened ? ' (shortened)' : ''}
      ${embedding.vector_index ? '' : ' · exact search (too wide for the vector index)'}
    </div>
    <div class="index-meta">
      ${index.documents_indexed}/${index.documents_total} documents ·
      ${Number(index.passages).toLocaleString()} passages
    </div>`;
}

/* ------------------------------------------------------------------- keys */

function renderKeys(config) {
  ['openai', 'anthropic', 'nvidia'].forEach((provider) => {
    const info = config.providers[provider];
    const input = $(`${provider}-key`);
    input.value = '';
    input.type = 'password';
    document.querySelector(`[data-reveal="${provider}-key"]`).textContent = 'Show';
    input.placeholder = info.key_set
      ? `${info.key_hint}  (saved, leave blank to keep)`
      : { openai: 'sk-…', anthropic: 'sk-ant-…', nvidia: 'nvapi-…' }[provider];

    const tag = document.querySelector(`[data-source-for="${provider}"]`);
    tag.className = `source-tag ${info.key_source}`;
    tag.textContent = info.key_source === 'saved' ? 'saved'
      : info.key_source === 'env' ? 'from environment' : 'not set';

    tag.parentElement.querySelector('.remove-key')?.remove();
    if (info.key_source === 'saved') {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'remove-key';
      remove.textContent = 'Remove';
      remove.onclick = () => removeKey(provider);
      tag.after(remove);
    }
  });
  $('openai-base').value = config.providers.openai.base_url || '';
  $('nvidia-base').value = config.providers.nvidia.base_url || '';
  const embedBase = config.providers.nvidia.embed_base_url || '';
  $('nvidia-embed-base').value = embedBase === config.providers.nvidia.base_url ? '' : embedBase;
  $('ollama-host').value = config.providers.ollama.host || '';
  document.querySelectorAll('.key-status').forEach((el) => { el.textContent = ''; el.className = 'key-status'; });
}

async function removeKey(provider) {
  if (!confirm(`Remove the saved ${label(provider)} key?`)) return;
  try {
    const config = await api.put('/config', { [`${provider}_api_key`]: '' });
    settingsUi.lists.chat[provider] = null;
    settingsUi.lists.embedding[provider] = null;
    applyConfig(config, { keepSelection: true });
    setSettingsMessage(`${label(provider)} key removed.`, 'ok');
    loadSettings();
  } catch (error) {
    setSettingsMessage(error.message, 'bad');
  }
}

async function testProvider(provider) {
  const status = document.querySelector(`[data-status-for="${provider}"]`);
  const button = document.querySelector(`[data-test="${provider}"]`);
  if (!canReach(provider)) {
    status.className = 'key-status bad';
    status.textContent = 'Enter a key first.';
    return;
  }
  button.disabled = true;
  status.className = 'key-status';
  status.textContent = 'Connecting…';
  settingsUi.lists.chat[provider] = null;
  try {
    const models = await fetchModels(provider, 'chat');
    status.className = 'key-status ok';
    status.textContent = `✓ Connected · ${models.length} model${models.length === 1 ? '' : 's'} available`;
  } catch (error) {
    status.className = 'key-status bad';
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------ open & save */

function applyConfig(config, { keepSelection = false } = {}) {
  settingsUi.config = config;
  renderKeys(config);
  renderIndexCard();
  document.querySelectorAll('#chat-providers button').forEach((b) =>
    b.classList.toggle('active-provider', b.dataset.provider === config.llm_provider));
  document.querySelectorAll('#embed-providers button').forEach((b) =>
    b.classList.toggle('active-provider', b.dataset.provider === config.embedding.provider));
  if (!keepSelection) {
    settingsUi.choice = { chat: {}, embedding: {} };
    settingsUi.chatProvider = null;
    settingsUi.embedProvider = null;
    selectProvider('chat', config.llm_provider in PROVIDERS ? config.llm_provider : 'openai');
    selectProvider('embedding', config.embedding.provider);
  } else {
    renderChatNote();
    renderEmbedWarning();
  }
}

async function openSettings(tab) {
  $('settings-panel').hidden = false;
  setSettingsMessage('Loading…');
  try {
    const config = await api.get('/config');
    applyConfig(config);
    const noKeys = ['openai', 'anthropic', 'nvidia'].every((p) => !config.providers[p].key_set);
    showSettingsTab(typeof tab === 'string' ? tab : (noKeys ? 'keys' : 'models'));
    if (noKeys) setSettingsMessage('Start by adding an API key for at least one provider.');
    else if (config.problems?.length) setSettingsMessage(config.problems[0], 'bad');
  } catch (error) {
    setSettingsMessage(`Could not load settings: ${error.message}`, 'bad');
  }
}

function closeSettings() {
  $('settings-panel').hidden = true;
}

async function saveSettings() {
  const chatProvider = settingsUi.chatProvider;
  const chatModel = chosen('chat');
  const embedModel = chosen('embedding');
  if (!chatModel || !embedModel) {
    showSettingsTab('models');
    setSettingsMessage('Choose a model for answers and for search, or type its name.', 'bad');
    return;
  }

  const body = {
    llm_provider: chatProvider,
    [`${chatProvider}_chat_model`]: chatModel,
    embed_provider: settingsUi.embedProvider,
    embed_model: embedModel,
    openai_base_url: $('openai-base').value.trim(),
    nvidia_base_url: $('nvidia-base').value.trim(),
    nvidia_embed_base_url: $('nvidia-embed-base').value.trim(),
    ollama_host: $('ollama-host').value.trim(),
  };
  ['openai', 'anthropic', 'nvidia'].forEach((p) => {
    const key = $(`${p}-key`).value.trim();
    if (key) body[`${p}_api_key`] = key; // blank keeps the current key
  });

  for (const [provider, purpose, kind] of [
    [chatProvider, 'answers', 'chat'], [settingsUi.embedProvider, 'search', 'embedding']]) {
    if (!canReach(provider, kind)) {
      showSettingsTab('keys');
      setSettingsMessage(`${label(provider)} is selected for ${purpose}. Add its API key first.`, 'bad');
      $(`${provider}-key`)?.focus();
      return;
    }
  }

  const rebuild = embeddingChanged();
  if (rebuild && settingsUi.config.index.documents_total && !confirm(
    `Switch search to ${label(settingsUi.embedProvider)} · ${embedModel}?\n\n` +
    `The current index will be cleared and all ${settingsUi.config.index.documents_total} ` +
    'documents re-embedded with the new model. This can take a while and uses the ' +
    'provider\'s API.')) return;

  const button = $('save-settings');
  button.disabled = true;
  button.textContent = rebuild ? 'Checking model…' : 'Saving…';
  try {
    const result = await api.put('/config', body);
    applyConfig(result);
    loadSettings();
    if (result.reindexing) {
      setSettingsMessage(
        `Saved. Re-indexing with ${label(result.embedding.provider)} · ${result.embedding.model} ` +
        `(${result.embedding.dimensions} dimensions); follow progress in Documents.`, 'ok');
      loadDocuments();
    } else if (result.problems?.length) {
      setSettingsMessage(`Saved, but: ${result.problems[0]}`, 'bad');
    } else {
      setSettingsMessage(`Saved. Answers: ${label(chatProvider)} · ${chatModel}.`, 'ok');
      setTimeout(() => { if (!$('settings-panel').hidden) closeSettings(); }, 1000);
    }
  } catch (error) {
    setSettingsMessage(error.message, 'bad');
  } finally {
    button.disabled = false;
    button.textContent = 'Save';
  }
}

function wireSettings() {
  ['open-settings', 'open-settings-side', 'model-chip', 'model-line'].forEach((id) => {
    $(id).onclick = () => openSettings();
  });
  $('close-settings').onclick = closeSettings;
  $('cancel-settings').onclick = closeSettings;
  $('settings-panel').onclick = (event) => {
    if (event.target === $('settings-panel')) closeSettings();
  };
  $('save-settings').onclick = saveSettings;

  document.querySelectorAll('.settings-tabs button').forEach((b) => {
    b.onclick = () => showSettingsTab(b.dataset.tab);
  });
  document.querySelectorAll('#chat-providers button').forEach((b) => {
    b.onclick = () => selectProvider('chat', b.dataset.provider);
  });
  document.querySelectorAll('#embed-providers button').forEach((b) => {
    b.onclick = () => selectProvider('embedding', b.dataset.provider);
  });
  $('chat-refresh').onclick = () => {
    settingsUi.lists.chat[settingsUi.chatProvider] = null;
    refreshModels('chat');
  };
  $('embed-refresh').onclick = () => {
    settingsUi.lists.embedding[settingsUi.embedProvider] = null;
    refreshModels('embedding');
  };
  Object.entries(KIND).forEach(([kind, ids]) => {
    $(ids.select).onchange = () => {
      const custom = $(ids.custom);
      custom.hidden = $(ids.select).value !== CUSTOM;
      if (!custom.hidden) custom.focus();
      if (kind === 'embedding') renderEmbedWarning();
    };
    $(ids.custom).oninput = () => { if (kind === 'embedding') renderEmbedWarning(); };
  });
  document.querySelectorAll('[data-test]').forEach((b) => {
    b.onclick = () => testProvider(b.dataset.test);
  });
  document.querySelectorAll('[data-reveal]').forEach((b) => {
    b.onclick = () => {
      const input = $(b.dataset.reveal);
      input.type = input.type === 'password' ? 'text' : 'password';
      b.textContent = input.type === 'password' ? 'Show' : 'Hide';
    };
  });
  // A newly typed key or address makes cached model lists stale.
  ['openai-key', 'anthropic-key', 'nvidia-key', 'openai-base', 'nvidia-base',
    'nvidia-embed-base', 'ollama-host'].forEach((id) => {
    $(id).addEventListener('change', () => {
      const provider = id.split('-')[0];
      settingsUi.lists.chat[provider] = null;
      settingsUi.lists.embedding[provider] = null;
    });
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('settings-panel').hidden) closeSettings();
  });
}

function showModelChip(llm) {
  const chip = $('model-chip');
  if (!llm) { chip.hidden = true; return; }
  const [provider, ...rest] = llm.split(':');
  chip.innerHTML =
    `<span class="prov">${escapeHtml(label(provider))}</span>` +
    `<span>${escapeHtml(rest.join(':'))}</span>`;
  chip.hidden = false;
}

/* ------------------------------------------------------------------ init */

async function loadSettings() {
  try {
    const info = await api.get('/settings');
    showModelChip(info.llm);
    $('model-line').textContent =
      `${info.llm} · ${info.embedder} (${info.dimensions}d)\n` +
      `${info.documents_indexed} documents, ${Number(info.passages_indexed).toLocaleString()} passages`;
    showProblems(info.problems || []);
  } catch (error) {
    $('model-line').textContent = `Backend unreachable: ${error.message}`;
  }
}

/* Configuration issues belong in front of the user, not only in the log. */
function showProblems(problems) {
  document.getElementById('config-problem')?.remove();
  if (!problems.length) return;
  const banner = document.createElement('div');
  banner.id = 'config-problem';
  banner.className = 'turn';
  banner.innerHTML = `<div class="error">⚠ ${escapeHtml(problems[0])}</div>`;
  $('messages').prepend(banner);
}

function autoGrow(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 190)}px`;
}

function wireEvents() {
  $('composer').onsubmit = (event) => {
    event.preventDefault();
    const input = $('input');
    const question = input.value;
    input.value = '';
    autoGrow(input);
    ask(question);
  };

  $('input').addEventListener('input', (event) => autoGrow(event.target));
  $('input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      $('composer').requestSubmit();
    }
  });

  $('new-chat').onclick = async () => {
    state.chatId = null;
    showEmptyState();
    await loadChats();
    $('input').focus();
  };

  $('toggle-sidebar').onclick = () => $('sidebar').classList.toggle('collapsed');

  $('open-documents').onclick = () => {
    $('documents-panel').hidden = false;
    loadDocuments();
  };
  $('close-documents').onclick = () => {
    $('documents-panel').hidden = true;
    clearTimeout(state.pollTimer);
    loadSettings();
  };
  $('documents-panel').onclick = (event) => {
    if (event.target === $('documents-panel')) $('close-documents').click();
  };
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('documents-panel').hidden) $('close-documents').click();
  });

  $('dropzone').onclick = () => $('file-input').click();
  $('file-input').onchange = (event) => uploadFiles(event.target.files);
  ['dragover', 'dragenter'].forEach((name) =>
    $('dropzone').addEventListener(name, (event) => {
      event.preventDefault();
      $('dropzone').classList.add('over');
    }));
  ['dragleave', 'drop'].forEach((name) =>
    $('dropzone').addEventListener(name, (event) => {
      event.preventDefault();
      $('dropzone').classList.remove('over');
      if (name === 'drop') uploadFiles(event.dataTransfer.files);
    }));

  $('index-button').onclick = async () => {
    await api.post('/documents/index');
    loadDocuments();
  };
  $('rebuild-button').onclick = async () => {
    if (!confirm('Re-read and re-embed every document? This can take a while.')) return;
    await api.post('/documents/index?rebuild=true');
    loadDocuments();
  };

  // Citation chips, follow-up chips, and source cards.
  document.addEventListener('click', (event) => {
    const followUp = event.target.closest('.chip');
    if (followUp) {
      ask(followUp.textContent);
      return;
    }
    const chip = event.target.closest('.cite');
    if (chip) {
      const card = chip
        .closest('.turn')
        ?.querySelector(`.source[data-source="${chip.dataset.cite}"]`);
      if (card) {
        card.querySelector('.source-body').hidden = false;
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      return;
    }
    const head = event.target.closest('.source-head');
    if (head) {
      const body = head.nextElementSibling;
      body.hidden = !body.hidden;
    }
  });
}

/* A silent JavaScript error used to leave the page half-working with no clue
   why. Surface it instead. */
function reportFailure(message) {
  let bar = document.getElementById('js-error');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'js-error';
    bar.className = 'js-error';
    document.body.appendChild(bar);
  }
  bar.textContent = `Interface error: ${message}`;
  bar.onclick = () => bar.remove();
}

window.addEventListener('error', (event) => reportFailure(event.message));
window.addEventListener('unhandledrejection', (event) =>
  reportFailure(event.reason?.message || String(event.reason)));

async function start() {
  wireEvents();
  wireSettings();
  showEmptyState();
  // Settled, not all-or-nothing: one failing call must not leave the page blank.
  const results = await Promise.allSettled([loadChats(), loadSettings(), loadDocuments()]);
  results
    .filter((r) => r.status === 'rejected')
    .forEach((r) => reportFailure(r.reason?.message || String(r.reason)));
  $('input').focus();
}

start();
