// ── Config ────────────────────────────────────────────────
const API   = 'http://127.0.0.1:8000';
const TOP_K = 10;

// ── Health check ─────────────────────────────────────────
async function checkHealth() {
  const dot   = document.getElementById('healthDot');
  const label = document.getElementById('healthLabel');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    dot.className     = 'health-dot ok';
    label.textContent = `${(d.docs || d.count || '—').toLocaleString()} docs`;
  } catch {
    dot.className     = 'health-dot err';
    label.textContent = 'offline';
  }
}
checkHealth();

// ── Helpers ──────────────────────────────────────────────
function setQuery(text) {
  document.getElementById('searchInput').value = text;
  doSearch();
}

function setLoading(on) {
  const spinner = document.getElementById('spinner');
  const icon    = document.getElementById('btnIcon');
  const label   = document.getElementById('btnLabel');
  const btn     = document.getElementById('searchBtn');
  spinner.style.display = on ? 'block' : 'none';
  icon.style.display    = on ? 'none'  : 'block';
  label.textContent     = on ? 'Searching…' : 'Search';
  btn.disabled          = on;
}

function showStatus(msg, isErr = false) {
  const bar     = document.getElementById('statusBar');
  bar.textContent = msg;
  bar.className   = 'status-bar' + (isErr ? ' error' : '');
}

function showSkeletons() {
  const el = document.getElementById('results');
  el.innerHTML = Array.from({ length: 5 }, () => `
    <div class="skeleton">
      <div class="skeleton-line" style="width:30%;margin-bottom:10px;height:10px;"></div>
      <div class="skeleton-line" style="width:60%;height:14px;margin-bottom:14px;"></div>
      <div class="skeleton-line" style="width:100%"></div>
      <div class="skeleton-line" style="width:90%"></div>
      <div class="skeleton-line" style="width:50%"></div>
    </div>
  `).join('');
  document.getElementById('divider').classList.add('visible');
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Parse RAG API response ───────────────────────────────
// Handles the specific format:
// { recommendations:[{rank,title,url,mal_score,why}], all_retrieved:[...], message, rewritten_query }
function parseRagResponse(raw) {
  // ── Format chính của API này ──────────────────────────
  if (Array.isArray(raw?.recommendations) && raw.recommendations.length > 0) {
    return {
      type: 'rag',
      message:         raw.message         || '',
      rewrittenQuery:  raw.rewritten_query || '',
      excludedTitles:  raw.excluded_titles || [],
      recommendations: raw.recommendations,
      allRetrieved:    raw.all_retrieved   || [],
    };
  }

  // ── Fallback: all_retrieved sẵn có nhưng recommendations rỗng ──
  if (Array.isArray(raw?.all_retrieved) && raw.all_retrieved.length > 0) {
    return {
      type: 'rag',
      message:         raw.message         || '',
      rewrittenQuery:  raw.rewritten_query || '',
      excludedTitles:  raw.excluded_titles || [],
      recommendations: raw.all_retrieved.map((r, i) => ({
        rank:      i + 1,
        title:     r.title     || `Result ${i + 1}`,
        url:       r.url       || '',
        mal_score: r.mal_score || 0,
        why:       '',
      })),
      allRetrieved: raw.all_retrieved,
    };
  }

  // ── Generic fallback cho API format khác ──────────────
  let items = [];
  if (Array.isArray(raw))                   items = raw;
  else if (Array.isArray(raw?.results))     items = raw.results;
  else if (Array.isArray(raw?.data))        items = raw.data;
  else if (Array.isArray(raw?.hits))        items = raw.hits;
  else                                      items = [raw];

  return {
    type: 'generic',
    message: '',
    rewrittenQuery: '',
    excludedTitles: [],
    recommendations: items.map((item, i) => {
      const meta = item?.metadata ?? item?.metadatas?.[0] ?? {};
      return {
        rank:      i + 1,
        title:     item?.title ?? meta?.title ?? `Result ${i + 1}`,
        url:       item?.url   ?? meta?.url   ?? '',
        mal_score: item?.score ?? item?.mal_score ?? meta?.score ?? 0,
        why:       item?.synopsis ?? item?.description ?? meta?.synopsis ?? '',
        genres:    item?.genres   ?? meta?.genres ?? '',
        episodes:  item?.episodes ?? meta?.episodes ?? null,
      };
    }),
    allRetrieved: [],
  };
}

// ── Render cards ─────────────────────────────────────────
function renderCards(parsed) {
  const container = document.getElementById('results');
  const items     = parsed.recommendations;

  if (!items || !items.length) {
    container.innerHTML = `
      <div class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
        </svg>
        No results found — try a different description.
      </div>`;
    return;
  }

  // Message bar (nếu có)
  const msgHtml = parsed.message ? `
    <div class="result-message">${escHtml(parsed.message)}</div>
  ` : '';

  // Rewritten query hint
  const rewriteHtml = parsed.rewrittenQuery && parsed.rewrittenQuery !== document.getElementById('searchInput').value.trim() ? `
    <div class="rewrite-hint">
      <span class="rewrite-label">Searched as:</span>
      <span class="rewrite-text">${escHtml(parsed.rewrittenQuery)}</span>
    </div>
  ` : '';

  const cardsHtml = items.map(item => {
    const score = item.mal_score ? parseFloat(item.mal_score) : null;
    const scoreHtml = score && score > 0
      ? `<div class="card-score">${score.toFixed(1)}<small>/10</small></div>`
      : '';

    const whyHtml = item.why
      ? `<div class="card-why">${escHtml(item.why)}</div>`
      : '';

    const urlHtml = item.url
      ? `<a class="card-link" href="${escHtml(item.url)}" target="_blank" rel="noopener">
           MAL ↗
         </a>`
      : '';

    // genres từ all_retrieved nếu có
    const retrieved = parsed.allRetrieved.find(r => r.title === item.title);
    const genreStr  = item.genres || retrieved?.genres || '';
    const genreList = Array.isArray(genreStr)
      ? genreStr
      : String(genreStr).split(/[,|]/).map(g => g.trim()).filter(Boolean);
    const tagsHtml  = genreList.slice(0, 5).map(g =>
      `<span class="tag">${escHtml(g)}</span>`
    ).join('');

    const relevance = retrieved?.relevance;
    const relHtml   = relevance != null
      ? `<div class="card-rel">${(relevance * 100).toFixed(0)}% match</div>`
      : '';

    return `
      <div class="card">
        <div class="card-left">
          <div class="card-rank">#${item.rank}</div>
          <div class="card-title">${escHtml(item.title)}</div>
          ${whyHtml}
          ${tagsHtml ? `<div class="card-tags">${tagsHtml}</div>` : ''}
        </div>
        <div class="card-score-wrap">
          ${scoreHtml}
          ${relHtml}
          ${urlHtml}
        </div>
      </div>`;
  }).join('');

  container.innerHTML = msgHtml + rewriteHtml + cardsHtml;
}

// ── Main search ──────────────────────────────────────────
async function doSearch() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) { showStatus('Type something first.', true); return; }

  setLoading(true);
  showStatus('');
  showSkeletons();
  document.getElementById('divider').classList.add('visible');

  const t0 = performance.now();

  let raw     = null;
  let usedUrl = '';

  try {
    const r = await fetch(`${API}/search`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, top_k: TOP_K }),
    });
    if (r.ok) {
      raw     = await r.json();
      usedUrl = r.url;
    } else {
      showStatus(`API error ${r.status}`, true);
    }
  } catch (err) {
    document.getElementById('results').innerHTML = `
      <div class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
        </svg>
        Could not reach API — make sure <code>uvicorn app:app</code> is running on port 8000.
      </div>`;
    showStatus('API unreachable — check the terminal.', true);
    setLoading(false);
    return;
  }

  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  setLoading(false);

  if (!raw) return;

  const parsed = parseRagResponse(raw);
  renderCards(parsed);

  const n     = parsed.recommendations.length;
  const label = `${n} result${n !== 1 ? 's' : ''} · ${elapsed}s`;
  document.getElementById('dividerLabel').textContent = label;
  showStatus(`↳ ${usedUrl}`);
}

// ── Keyboard shortcut ────────────────────────────────────
document.getElementById('searchInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

// ── Back-to-top ──────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('backTop').classList.toggle('show', window.scrollY > 300);
});
// (styles appended separately in style.css)