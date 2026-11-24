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
  put: (path, body) =>
    api.request(path, {
      method: 'PUT',
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

/* References: the book, the exact page and lines, the passage itself, a link
   that opens the PDF at that page, and a Vancouver-style reference to copy. */
function renderSources(citations) {
  if (!citations || !citations.length) return '';
  const items = citations
    .map((c) => {
      const title = c.title || c.source;
      const byline = [c.author, c.year].filter(Boolean).join(', ');
      const where = c.location || pageLabel(c);
      const isPdf = /\.pdf$/i.test(c.source || '');
      const open = c.document_id
        ? `<a class="ref-action" href="/api/documents/${c.document_id}/file${isPdf ? `#page=${c.page_start}` : ''}"
              target="_blank" rel="noopener">Open ${isPdf ? `PDF at p. ${c.page_start}` : 'document'}</a>`
        : '';
      const reference = c.vancouver || `${title}${c.year ? `. ${c.year}` : ''}. ${pageLabel(c)}.`;
      return `
      <div class="source" data-source="${c.index}">
        <div class="source-head">
          <span class="source-index">${c.index}</span>
          <span class="source-name"><strong>${escapeHtml(title)}</strong>${
            byline ? ` <span class="byline">· ${escapeHtml(byline)}</span>` : ''}</span>
          <span class="source-page">${escapeHtml(where)}</span>
        </div>
        <div class="source-body" hidden>
          ${c.section ? `<div class="source-section">${escapeHtml(c.section)}</div>` : ''}
          <blockquote>${escapeHtml(c.excerpt || '')}</blockquote>
          <div class="source-actions">
            ${open}
            <button class="ref-action copy-ref" type="button" data-ref="${escapeHtml(reference)}">Copy reference</button>
            <span class="source-file">${escapeHtml(c.source || '')}</span>
          </div>
        </div>
      </div>`;
    })
    .join('');
  const all = citations
    .map((c) => `${c.index}. ${c.vancouver || `${c.title || c.source}. ${c.location || pageLabel(c)}.`}`)
    .join('\n');
  return `<div class="sources">
      <div class="sources-head"><span>References</span>
        <button class="ref-action copy-ref" type="button" data-ref="${escapeHtml(all)}">Copy all</button>
      </div>${items}</div>`;
}

const pageLabel = (c) =>
  c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}-${c.page_end}`;

/* ---------------------------------------------------------------- tokens */

const fmtTokens = (n) => {
  n = Number(n) || 0;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
  return String(n);
};
const STEP_LABEL = {
  plan: 'Planning', search: 'Search embeddings', grade: 'Ranking passages',
  write: 'Writing the answer', verify: 'Fact-check', repair: 'Correction',
  suggest: 'Follow-up questions', chat: 'Chat reply', memory: 'Conversation memory',
  index: 'Indexing documents', check: 'Model check', call: 'Other',
};

/* One line under an answer: total, in/out, cache, calls; steps on click. */
function usageLine(usage, { live = false } = {}) {
  if (!usage || !(usage.input || usage.output)) return '';
  const total = (usage.input || 0) + (usage.output || 0);
  const cached = usage.cached ? ` <span class="cached">(${fmtTokens(usage.cached)} cached)</span>` : '';
  const steps = Object.entries(usage.steps || {})
    .sort((a, b) => (b[1].input + b[1].output) - (a[1].input + a[1].output))
    .map(([step, u]) => `<tr><td>${escapeHtml(STEP_LABEL[step] || step)}</td>
        <td>${fmtTokens(u.input)}</td><td>${fmtTokens(u.output)}</td>
        <td>${u.cached ? fmtTokens(u.cached) : '—'}</td><td>${u.calls}</td></tr>`)
    .join('');
  return `<details class="usage-line${live ? ' live' : ''}">
      <summary><span class="usage-total">${live ? '~' : ''}${fmtTokens(total)} tokens</span>
        <span class="sep">·</span>${fmtTokens(usage.input)} in${cached}
        <span class="sep">·</span>${live ? '~' : ''}${fmtTokens(usage.output)} out
        <span class="sep">·</span>${usage.calls || 0} call${usage.calls === 1 ? '' : 's'}</summary>
      ${steps ? `<table class="usage-steps"><thead><tr><th>Step</th><th>In</th><th>Out</th>
        <th>Cached</th><th>Calls</th></tr></thead><tbody>${steps}</tbody></table>` : ''}
    </details>`;
}

function setChatTokens(usage) {
  const chip = $('chat-tokens');
  const total = usage ? (Number(usage.input) || 0) + (Number(usage.output) || 0) : 0;
  state.chatUsage = usage || null;
  if (!total) { chip.hidden = true; return; }
  chip.hidden = false;
  chip.innerHTML = `<span class="sigma">Σ</span> ${fmtTokens(total)} <span class="unit">tokens</span>`;
  chip.title = `This chat: ${Number(usage.input || 0).toLocaleString()} input ` +
    `(${Number(usage.cached || 0).toLocaleString()} cached) · ` +
    `${Number(usage.output || 0).toLocaleString()} output · ${usage.calls || 0} model calls`;
}

function messageElement(role, html, citations, usage) {
  const turn = document.createElement('div');
  turn.className = `turn ${role}`;
  turn.innerHTML =
    role === 'user'
      ? `<div class="bubble">${html}</div>`
      : `<div class="answer">${html}</div>${renderSources(citations)}${usageLine(usage)}`;
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
      ${Number(chat.tokens) ? `<small class="chat-item-tokens">${fmtTokens(chat.tokens)}</small>` : ''}
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
  setChatTokens(null);
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.innerHTML = `
    <div class="empty-icon">🩺</div>
    <h2>Ask about your documents</h2>
    <p>Ask a question, paste a clinical case, or paste an exam MCQ with its options. Every answer cites the page it came from and is fact-checked against it.</p>`;
  $('messages').appendChild(empty);
}

async function openChat(chatId) {
  state.chatId = chatId;
  const chat = state.chats.find((c) => c.id === chatId);
  $('chat-title').textContent = chat ? chat.title : 'Chat';

  const [{ messages }, usage] = await Promise.all([
    api.get(`/chats/${chatId}/messages`),
    api.get(`/chats/${chatId}/usage`).catch(() => null),
  ]);
  setChatTokens(usage);
  const box = $('messages');
  box.innerHTML = '';
  messages.forEach((message) => {
    const html =
      message.role === 'user'
        ? escapeHtml(message.content)
        : renderMarkdown(message.content);
    box.appendChild(messageElement(message.role, html, message.citations, message.usage));
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

const MODE_LABEL = {
  lookup: 'Quick lookup', concept: 'Concept', management: 'Management',
  comparison: 'Comparison', vignette: 'Clinical case', mcq: 'Exam question',
};
const STRENGTH_LABEL = {
  strong: 'strong evidence', partial: 'partial evidence', limited: 'limited evidence', none: 'no evidence',
};

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
  const trace = { plan: null, reasoning: null, evidence: null, verify: null, suggestions: [] };

  // Token counter: exact totals arrive after each step; while the answer
  // streams, output tokens are estimated (~4 characters a token) until the
  // exact count lands.
  let usage = null;
  let streamedSince = 0;
  const chatBefore = state.chatUsage ? { ...state.chatUsage } : null;
  const liveUsage = () => {
    const base = usage || { input: 0, output: 0, cached: 0, calls: 0, steps: {} };
    const estimate = Math.round(streamedSince / 4);
    return estimate ? { ...base, output: (base.output || 0) + estimate } : base;
  };
  const showLiveTotals = () => {
    const now = liveUsage();
    const before = chatBefore || { input: 0, output: 0, cached: 0, calls: 0 };
    setChatTokens({
      input: (Number(before.input) || 0) + (now.input || 0),
      output: (Number(before.output) || 0) + (now.output || 0),
      cached: (Number(before.cached) || 0) + (now.cached || 0),
      calls: (Number(before.calls) || 0) + (now.calls || 0),
    });
  };

  const traceHtml = () => {
    if (!trace.plan || trace.plan.mode === 'chat') return '';
    const ev = trace.evidence;
    const missing = new Set(ev?.missing || []);
    const subquestions = (trace.plan.subquestions || [])
      .map((q) => {
        const open = missing.has(q);
        const mark = !ev ? '<span class="spin"></span>'
          : open ? '<span class="none">○</span>' : '<span class="tick">✓</span>';
        const note = !ev ? 'searching…' : open ? 'not in your documents' : 'covered';
        return `<li>${mark}<span class="sq">${escapeHtml(q)}</span><span class="note">${note}</span></li>`;
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
    const retried = ev?.retried?.length
      ? `<div class="review"><strong>Looked again for:</strong><ul>${
          ev.retried.map((q) => `<li>${escapeHtml(q)}</li>`).join('')
        }</ul></div>`
      : '';
    const issues = trace.verify?.issues || [];
    const verify = !trace.verify ? ''
      : issues.length
        ? `<div class="review"><strong>Fact-check ${trace.verify.repaired ? 'corrected' : 'flagged'}:</strong><ul>${
            issues.map((i) => `<li>${i.quote ? `“${escapeHtml(i.quote)}” — ` : ''}${escapeHtml(i.problem)}</li>`).join('')
          }</ul></div>`
        : '<div class="review ok">✓ Every claim checked against its cited source</div>';

    const badges = [
      `<span class="badge-mode">${escapeHtml(MODE_LABEL[trace.plan.mode] || trace.plan.mode)}</span>`,
      ev ? `<span>${ev.sources} source${ev.sources === 1 ? '' : 's'}</span>` : '<span>searching…</span>',
      ev ? `<span class="strength ${ev.strength}">${STRENGTH_LABEL[ev.strength] || ''}</span>` : '',
      trace.verify ? `<span class="checked">${issues.length && !trace.verify.repaired ? '⚠ check flagged' : '✓ fact-checked'}</span>` : '',
    ].filter(Boolean).join('<span class="sep">·</span>');
    return `<details class="trace"${state.streaming ? ' open' : ''}>
        <summary>${badges}</summary>
        ${reasoning}<ul class="panel-members">${subquestions}</ul>${retried}${verify}
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
            `<div class="status"><span class="dot"></span><span>${escapeHtml(event.text)}</span></div>` +
            usageLine(liveUsage(), { live: true });
          scrollDown();
        } else if (event.type === 'usage') {
          // Exact counts after each step; the estimate restarts from here.
          usage = event;
          streamedSince = 0;
          showLiveTotals();
        } else if (event.type === 'plan') {
          trace.plan = event;
        } else if (event.type === 'reasoning') {
          trace.reasoning = event;
        } else if (event.type === 'evidence') {
          trace.evidence = event;
        } else if (event.type === 'verify') {
          trace.verify = event;
        } else if (event.type === 'citations') {
          citations = event.citations;
        } else if (event.type === 'token') {
          answer += event.text;
          streamedSince += event.text.length;
          turn.innerHTML = traceHtml() + `<div class="answer">${renderMarkdown(answer)}</div>` +
            usageLine(liveUsage(), { live: true });
          showLiveTotals();
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
          if (event.usage) usage = event.usage;
          streamedSince = 0;
          turn.innerHTML = traceHtml() +
            `<div class="answer">${renderMarkdown(answer)}</div>${renderSources(citations)}` +
            usageLine(usage) + chipsHtml();
          if (event.chat_usage) setChatTokens(event.chat_usage);
          refreshUsageBadge();
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
    const indexingNow = doc.status === 'indexing';
    row.innerHTML = `
      <div class="doc-top">
        <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.title || doc.filename)}</span>
        <span class="pill ${doc.status}">${doc.status}</span>
        <button class="row-action">${excluded ? 'Include' : 'Exclude'}</button>
        <button class="row-action danger delete-doc" ${indexingNow ? 'disabled title="Wait for indexing to finish"' : ''}>Delete</button>
      </div>
      <div class="doc-meta">
        ${doc.pages || 0} pages · ${(doc.size_bytes / 1e6).toFixed(1)} MB
        ${doc.chunk_count ? ` · ${doc.chunk_count.toLocaleString()} passages` : ''}
        ${doc.year ? ` · ${doc.year}` : ''}
      </div>
      ${doc.status === 'indexing' ? `<div class="bar"><div style="width:${percent}%"></div></div>` : ''}
      ${doc.error ? `<div class="doc-error">${escapeHtml(doc.error)}</div>` : ''}`;
    row.querySelector('.row-action').onclick = async () => {
      // Excluding keeps the file and its passages; it only stops searching them.
      if (!excluded && !confirm(
        `Exclude "${doc.filename}" from search?\n\n` +
        `The file and its index stay - answers just stop drawing on it. ` +
        `You can include it again at any time, at no cost.`)) return;
      await api.post(`/documents/${doc.id}/${excluded ? 'include' : 'exclude'}`);
      loadDocuments();
      loadSettings();
    };
    row.querySelector('.delete-doc').onclick = async () => {
      const passages = doc.chunk_count ? ` and its ${doc.chunk_count.toLocaleString()} indexed passages` : '';
      if (!confirm(
        `Delete "${doc.filename}" permanently?\n\n` +
        `This deletes the file from your knowledgebase folder${passages}. ` +
        `It cannot be undone - you would need to upload and index it again.`)) return;
      try {
        await api.remove(`/documents/${doc.id}`);
      } catch (error) {
        alert(`Could not delete: ${error.message}`);
      }
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

const fmtBytes = (n) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB`
    : `${Math.max(1, Math.round(n / 1e3))} KB`;
const fmtSeconds = (s) =>
  !Number.isFinite(s) ? '' : s >= 90 ? `${Math.round(s / 60)} min left` : `${Math.max(1, Math.round(s))} s left`;

/* Upload with a real progress bar. fetch() cannot report upload progress, so
   this uses XMLHttpRequest: bytes sent, speed, time left, then a "processing"
   phase while the server saves the file and reads its pages. */
function uploadFiles(files) {
  files = Array.from(files || []);
  if (!files.length || state.uploading) return;
  const total = files.reduce((sum, f) => sum + f.size, 0);
  const name = files.length === 1 ? files[0].name : `${files.length} files`;
  const card = $('upload-card');
  const show = (percent, meta, phase = '') => {
    card.hidden = false;
    card.className = `upload-card ${phase}`;
    $('upload-name').textContent = name;
    $('upload-percent').textContent = phase === 'error' ? 'Failed'
      : phase === 'processing' ? 'Processing…' : phase === 'done' ? 'Done' : `${percent}%`;
    $('upload-bar').style.width = `${percent}%`;
    $('upload-meta').textContent = meta;
  };

  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  const xhr = new XMLHttpRequest();
  const started = performance.now();
  state.uploading = true;
  $('dropzone').classList.add('busy');
  show(0, `0 of ${fmtBytes(total)}`);

  xhr.upload.onprogress = (event) => {
    if (!event.lengthComputable) return;
    const seconds = (performance.now() - started) / 1000;
    const speed = event.loaded / Math.max(seconds, 0.001);
    const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
    show(percent, `${fmtBytes(event.loaded)} of ${fmtBytes(event.total)} · ` +
      `${fmtBytes(speed)}/s · ${fmtSeconds((event.total - event.loaded) / speed)}`);
  };
  xhr.upload.onload = () =>
    show(100, 'Uploaded. Saving and reading the pages - large PDFs take a moment.', 'processing');

  const finish = () => {
    state.uploading = false;
    $('dropzone').classList.remove('busy');
    $('file-input').value = '';
  };
  xhr.onload = () => {
    finish();
    if (xhr.status >= 200 && xhr.status < 300) {
      show(100, `${name} added. Press "Index new documents" to make it searchable.`, 'done');
      setTimeout(() => { if (!state.uploading) card.hidden = true; }, 6000);
    } else {
      let detail = xhr.statusText;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch { /* not JSON */ }
      show(100, `Upload failed: ${detail}`, 'error');
    }
    loadDocuments();
  };
  xhr.onerror = () => {
    finish();
    show(0, 'Upload failed: the connection was interrupted.', 'error');
  };
  xhr.open('POST', '/api/documents');
  xhr.send(form);
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
  fastChoice: {},                        // provider -> helper model picked this session
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

/* The helper model: same provider as the answer model, optional. */
function fillHelper() {
  const provider = settingsUi.chatProvider;
  const select = $('fast-model');
  const saved = settingsUi.fastChoice[provider] ?? settingsUi.config.providers[provider]?.fast_model ?? '';
  const names = [...new Set(settingsUi.lists.chat[provider] || [])];
  if (saved && !names.includes(saved)) names.unshift(saved);
  select.innerHTML = '<option value="">Same as answer model</option>' +
    names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');
  select.value = saved;
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
    if (kind === 'chat') {
      settingsUi.fastChoice[provider] = $('fast-model').value;
      fillHelper();
    }
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
  if (previous && kind === 'chat') settingsUi.fastChoice[previous] = $('fast-model').value;
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

  if (kind === 'chat') {
    renderChatNote();
    fillHelper();
  } else {
    renderEmbedWarning();
  }

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

function renderProfile(profile = {}) {
  $('learner-level').value = profile.learner_level || '';
  $('exam-target').value = profile.exam_target || '';
  setAnswerLength(profile.answer_length || '');
}

function setAnswerLength(value) {
  document.querySelectorAll('#answer-length button').forEach((b) =>
    b.setAttribute('aria-checked', String(b.dataset.length === value)));
}

const answerLength = () =>
  document.querySelector('#answer-length button[aria-checked="true"]')?.dataset.length || '';

function applyConfig(config, { keepSelection = false } = {}) {
  settingsUi.config = config;
  renderKeys(config);
  renderIndexCard();
  renderProfile(config.profile);
  document.querySelectorAll('#chat-providers button').forEach((b) =>
    b.classList.toggle('active-provider', b.dataset.provider === config.llm_provider));
  document.querySelectorAll('#embed-providers button').forEach((b) =>
    b.classList.toggle('active-provider', b.dataset.provider === config.embedding.provider));
  if (!keepSelection) {
    settingsUi.choice = { chat: {}, embedding: {} };
    settingsUi.fastChoice = {};
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
    [`${chatProvider}_fast_model`]: $('fast-model').value, // '' = same as answer model
    embed_provider: settingsUi.embedProvider,
    embed_model: embedModel,
    openai_base_url: $('openai-base').value.trim(),
    nvidia_base_url: $('nvidia-base').value.trim(),
    nvidia_embed_base_url: $('nvidia-embed-base').value.trim(),
    ollama_host: $('ollama-host').value.trim(),
    learner_level: $('learner-level').value,
    exam_target: $('exam-target').value.trim(),
    answer_length: answerLength(),
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

/* ----------------------------------------------------------------- usage */

const shortDay = (iso) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
const sumTokens = (u) => (Number(u?.input) || 0) + (Number(u?.output) || 0);

async function refreshUsageBadge() {
  try {
    const report = await api.get('/usage?days=1');
    const today = sumTokens(report.today);
    $('usage-badge').textContent = today ? `${fmtTokens(today)} today` : '';
  } catch { /* the badge is a nicety */ }
}

function statTile(label, u, note = '') {
  return `<div class="stat">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${fmtTokens(sumTokens(u))}</div>
      <div class="stat-note">${note || `${fmtTokens(u.input)} in · ${fmtTokens(u.output)} out · ${Number(u.calls || 0).toLocaleString()} calls`}</div>
    </div>`;
}

/* One series, one hue: total tokens per day. Thin bars anchored to the
   baseline, a hover tooltip per bar, and the tables below as the text view. */
function dailyChart(days) {
  const peak = Math.max(1, ...days.map(sumTokens));
  const bars = days.map((d, i) => {
    const total = sumTokens(d);
    const height = total ? Math.max(2, (total / peak) * 100) : 0;
    const label = i % 2 === (days.length - 1) % 2 ? shortDay(d.day) : '';
    return `<div class="bar-col" tabindex="0" aria-label="${shortDay(d.day)}: ${total.toLocaleString()} tokens">
        <div class="bar-fill" style="height:${height}%"></div>
        <div class="bar-tip"><strong>${shortDay(d.day)}</strong>${total.toLocaleString()} tokens
          <span>${fmtTokens(d.input)} in · ${fmtTokens(d.output)} out${Number(d.cached) ? ` · ${fmtTokens(d.cached)} cached` : ''}</span></div>
        <span class="bar-label">${label}</span>
      </div>`;
  }).join('');
  return `<div class="chart" role="img" aria-label="Tokens per day for the last ${days.length} days">
      <div class="chart-top"><span>Tokens per day</span><span class="chart-peak">peak ${fmtTokens(peak)}</span></div>
      <div class="chart-plot">${bars}</div>
    </div>`;
}

function usageTable(title, rows, nameOf, extra = () => '') {
  if (!rows.length) return '';
  const body = rows.map((r) => `<tr>
      <td class="name">${nameOf(r)}</td><td>${fmtTokens(r.input)}</td><td>${fmtTokens(r.output)}</td>
      <td>${Number(r.cached) ? fmtTokens(r.cached) : '—'}</td><td>${Number(r.calls).toLocaleString()}</td>${extra(r)}
    </tr>`).join('');
  return `<section class="usage-section"><h3>${title}</h3>
      <table class="usage-table"><thead><tr><th></th><th>In</th><th>Out</th><th>Cached</th><th>Calls</th></tr></thead>
      <tbody>${body}</tbody></table></section>`;
}

async function openUsage() {
  $('usage-panel').hidden = false;
  const body = $('usage-body');
  body.innerHTML = '<p class="status-line">Loading…</p>';
  try {
    const r = await api.get('/usage?days=14');
    const all = r.all_time;
    const cacheShare = Number(all.input) ? Math.round((Number(all.cached) / Number(all.input)) * 100) : 0;
    if (!sumTokens(all)) {
      body.innerHTML = '<p class="status-line">No tokens used yet. Ask a question or index a document and the numbers appear here.</p>';
      return;
    }
    body.innerHTML = `
      <div class="stats">
        ${statTile('Today', r.today)}
        ${statTile('Last 7 days', r.week)}
        ${statTile('All time', all)}
        ${statTile('Served from cache', { input: all.cached, output: 0 },
          `${cacheShare}% of input tokens, billed at a discount`)}
      </div>
      ${dailyChart(r.daily)}
      <div class="usage-split">
        ${statTile('Answering questions', r.chats)}
        ${statTile('Indexing documents', r.indexing, `${fmtTokens(r.indexing.input)} embedded · one-time cost`)}
      </div>
      ${usageTable('By step', r.by_step, (s) => escapeHtml(STEP_LABEL[s.step] || s.step))}
      ${usageTable('By model', r.by_model, (m) => `${escapeHtml(PROVIDERS[m.provider]?.label || m.provider)} · ${escapeHtml(m.model)}`)}
      ${usageTable('Chats using the most', r.top_chats,
        (c) => c.chat_id ? `<a href="#" class="open-chat" data-chat="${c.chat_id}">${escapeHtml(c.title)}</a>` : escapeHtml(c.title))}
      <p class="hint">Input counts every prompt token, including those served from the provider's cache. Long chats stay cheap: older messages are folded into a short running summary instead of being resent.</p>`;
  } catch (error) {
    body.innerHTML = `<p class="error">Could not load usage: ${escapeHtml(error.message)}</p>`;
  }
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
    if (event.key === 'Escape' && !$('usage-panel').hidden) $('usage-panel').hidden = true;
  });

  $('open-usage').onclick = openUsage;
  $('chat-tokens').onclick = openUsage;
  $('close-usage').onclick = () => { $('usage-panel').hidden = true; };
  $('usage-panel').onclick = (event) => {
    if (event.target === $('usage-panel')) $('usage-panel').hidden = true;
    const link = event.target.closest('.open-chat');
    if (link) {
      event.preventDefault();
      $('usage-panel').hidden = true;
      openChat(Number(link.dataset.chat));
    }
  };
  document.querySelectorAll('#answer-length button').forEach((b) => {
    b.onclick = () => setAnswerLength(b.dataset.length);
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
    const copy = event.target.closest('.copy-ref');
    if (copy) {
      event.stopPropagation();
      navigator.clipboard?.writeText(copy.dataset.ref).then(() => {
        const label = copy.textContent;
        copy.textContent = 'Copied ✓';
        setTimeout(() => { copy.textContent = label; }, 1400);
      }).catch(() => window.prompt('Copy this reference:', copy.dataset.ref));
      return;
    }
    if (event.target.closest('.ref-action')) return; // links open on their own
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
  const results = await Promise.allSettled([
    loadChats(), loadSettings(), loadDocuments(), refreshUsageBadge(),
  ]);
  results
    .filter((r) => r.status === 'rejected')
    .forEach((r) => reportFailure(r.reason?.message || String(r.reason)));
  $('input').focus();
}

start();
