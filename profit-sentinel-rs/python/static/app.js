/**
 * Profit Sentinel — Mobile Client
 *
 * Vanilla JS, no framework, no build step.
 * Target: 90-second mobile interaction at 6 AM.
 *
 * Flow: Auth -> Load Digest -> Browse Issues -> Delegate / Call Vendor
 */

'use strict';

// ---------------------------------------------------------------------------
// API Client
// ---------------------------------------------------------------------------

class SentinelAPI {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('sentinel_token') || '';
  }

  setToken(token) {
    this.token = token;
    localStorage.setItem('sentinel_token', token);
  }

  clearToken() {
    this.token = '';
    localStorage.removeItem('sentinel_token');
  }

  get isAuthenticated() {
    return this.token.length > 0;
  }

  async _fetch(path, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...(this.token ? { 'Authorization': `Bearer ${this.token}` } : {}),
      ...options.headers,
    };

    const resp = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    if (resp.status === 401) {
      this.clearToken();
      showAuthGate();
      throw new Error('Authentication required');
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ message: resp.statusText }));
      throw new Error(err.message || `HTTP ${resp.status}`);
    }

    return resp.json();
  }

  // Pipeline
  getDigest(stores, topK = 5) {
    let qs = `?top_k=${topK}`;
    if (stores) qs += `&stores=${stores}`;
    return this._fetch(`/api/v1/digest${qs}`);
  }

  getStoreDigest(storeId, topK = 5) {
    return this._fetch(`/api/v1/digest/${storeId}?top_k=${topK}`);
  }

  // Delegation
  delegate(issueId, assignee, notes) {
    return this._fetch('/api/v1/delegate', {
      method: 'POST',
      body: JSON.stringify({ issue_id: issueId, assignee, notes }),
    });
  }

  getTasks(filters = {}) {
    const params = new URLSearchParams();
    if (filters.store_id) params.set('store_id', filters.store_id);
    if (filters.priority) params.set('priority', filters.priority);
    if (filters.status) params.set('status', filters.status);
    const qs = params.toString();
    return this._fetch(`/api/v1/tasks${qs ? '?' + qs : ''}`);
  }

  getTask(taskId) {
    return this._fetch(`/api/v1/tasks/${taskId}`);
  }

  updateTask(taskId, status, notes) {
    return this._fetch(`/api/v1/tasks/${taskId}`, {
      method: 'PATCH',
      body: JSON.stringify({ status, notes }),
    });
  }

  // Vendor
  getVendorCall(issueId) {
    return this._fetch(`/api/v1/vendor-call/${issueId}`);
  }

  // Co-op
  getCoopReport(storeId) {
    return this._fetch(`/api/v1/coop/${storeId}`);
  }

  // Health
  health() {
    return this._fetch('/health');
  }

  // Diagnostic
  startDiagnostic(items, storeName = 'My Store') {
    return this._fetch('/api/v1/diagnostic/start', {
      method: 'POST',
      body: JSON.stringify({ items, store_name: storeName }),
    });
  }

  getDiagnosticQuestion(sessionId) {
    return this._fetch(`/api/v1/diagnostic/${sessionId}/question`);
  }

  answerDiagnostic(sessionId, classification, note = '') {
    return this._fetch(`/api/v1/diagnostic/${sessionId}/answer`, {
      method: 'POST',
      body: JSON.stringify({ classification, note }),
    });
  }

  getDiagnosticSummary(sessionId) {
    return this._fetch(`/api/v1/diagnostic/${sessionId}/summary`);
  }

  getDiagnosticReport(sessionId) {
    return this._fetch(`/api/v1/diagnostic/${sessionId}/report`);
  }
}

const api = new SentinelAPI();

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatDollars(amount) {
  if (amount < 0) return `-$${Math.abs(amount).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  return `$${amount.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
}

function formatTime() {
  return new Date().toLocaleString('en-US', {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function storeDisplay(storeId) {
  if (storeId.startsWith('store-')) return `Store ${storeId.slice(6)}`;
  return storeId;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

function showAuthGate() {
  document.getElementById('auth-gate').style.display = 'flex';
  document.getElementById('app').style.display = 'none';
}

function showApp() {
  document.getElementById('auth-gate').style.display = 'none';
  document.getElementById('app').style.display = 'block';
}

function initAuth() {
  const submitBtn = document.getElementById('auth-submit');
  const tokenInput = document.getElementById('token-input');

  submitBtn.addEventListener('click', () => {
    const token = tokenInput.value.trim();
    if (token) {
      api.setToken(token);
      showApp();
      loadDigest();
    }
  });

  tokenInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') submitBtn.click();
  });
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

let currentView = 'digest-view';

function initNavigation() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const targetView = tab.dataset.view;
      switchView(targetView);
    });
  });
}

function switchView(viewId) {
  // Update views
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(viewId).classList.add('active');

  // Update tabs
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-view="${viewId}"]`).classList.add('active');

  currentView = viewId;

  // Auto-load data for view
  if (viewId === 'tasks-view') loadTasks();
  if (viewId === 'coop-view') loadCoopReport();
}

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

let lastDigest = null;

async function loadDigest() {
  const loading = document.getElementById('digest-loading');
  const content = document.getElementById('digest-content');
  const empty = document.getElementById('digest-empty');

  loading.style.display = 'block';
  content.style.display = 'none';
  empty.style.display = 'none';

  try {
    const data = await api.getDigest();
    lastDigest = data;

    if (data.issue_count === 0) {
      loading.style.display = 'none';
      empty.style.display = 'block';
      return;
    }

    // Summary
    document.getElementById('total-impact').textContent = formatDollars(data.total_dollar_impact);
    document.getElementById('issue-count').textContent = `${data.issue_count} issues`;
    document.getElementById('store-count').textContent = `${data.store_ids.length} stores`;
    document.getElementById('pipeline-time').textContent = `${data.digest.pipeline_ms}ms`;

    // Issue cards
    renderIssueCards(data.digest.issues);

    loading.style.display = 'none';
    content.style.display = 'block';
  } catch (err) {
    loading.innerHTML = `<div style="color: var(--accent-red);">Error: ${err.message}</div>`;
  }
}

function renderIssueCards(issues) {
  const container = document.getElementById('issue-list');
  container.innerHTML = '';

  issues.forEach((issue, index) => {
    const card = document.createElement('div');
    card.className = 'issue-card';
    card.innerHTML = `
      <div class="issue-header">
        <span class="issue-type">${issue.issue_type.replace(/([A-Z])/g, ' $1').trim()}</span>
        <span class="issue-dollar">${formatDollars(issue.dollar_impact)}</span>
      </div>
      <div class="issue-store">${storeDisplay(issue.store_id)} &middot; ${issue.skus.length} SKUs</div>
      <div class="issue-detail">${issue.context}</div>
      <span class="issue-trend ${issue.trend_direction.toLowerCase()}">${issue.trend_direction}</span>
      <div class="action-row">
        <button class="btn btn-primary" onclick="delegateIssue('${issue.id}')">
          Delegate
        </button>
        <button class="btn" onclick="prepareVendorCall('${issue.id}')">
          Vendor Call
        </button>
      </div>
    `;
    container.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Delegation
// ---------------------------------------------------------------------------

async function delegateIssue(issueId) {
  const assignee = prompt('Assign to (name or role):');
  if (!assignee) return;

  try {
    const result = await api.delegate(issueId, assignee);
    alert(`Task created: ${result.task_id}\nAssigned to: ${assignee}`);
    if (currentView === 'tasks-view') loadTasks();
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

async function loadTasks() {
  const loading = document.getElementById('tasks-loading');
  const content = document.getElementById('tasks-content');
  const empty = document.getElementById('tasks-empty');

  loading.style.display = 'block';
  content.style.display = 'none';
  empty.style.display = 'none';

  try {
    const data = await api.getTasks();

    if (data.total === 0) {
      loading.style.display = 'none';
      empty.style.display = 'block';
      return;
    }

    renderTaskList(data.tasks);
    loading.style.display = 'none';
    content.style.display = 'block';
  } catch (err) {
    loading.innerHTML = `<div style="color: var(--accent-red);">Error: ${err.message}</div>`;
  }
}

function renderTaskList(tasks) {
  const container = document.getElementById('task-list');
  container.innerHTML = '';

  tasks.forEach(taskResp => {
    const task = taskResp.task;
    const status = taskResp.status;
    const item = document.createElement('div');
    item.className = 'task-item';
    item.innerHTML = `
      <div class="priority-badge ${task.priority}"></div>
      <div class="task-info">
        <div class="task-title">${task.title}</div>
        <div class="task-meta">
          ${storeDisplay(task.store_id)} &middot;
          ${task.assignee} &middot;
          ${formatDollars(task.dollar_impact)}
        </div>
      </div>
      <span class="task-status ${status}">${status.replace('_', ' ')}</span>
    `;
    item.addEventListener('click', () => showTaskDetail(task.task_id));
    container.appendChild(item);
  });
}

async function showTaskDetail(taskId) {
  try {
    const data = await api.getTask(taskId);
    const task = data.task;

    const actions = task.action_items.map(a => `  - ${a}`).join('\n');
    const msg = [
      task.title,
      `Priority: ${task.priority.toUpperCase()}`,
      `Assignee: ${task.assignee}`,
      `Impact: ${formatDollars(task.dollar_impact)}`,
      `Status: ${data.status}`,
      '',
      'Action Items:',
      actions,
    ].join('\n');

    const action = prompt(`${msg}\n\nUpdate status? (complete / escalate / cancel):`);
    if (action === 'complete') {
      await api.updateTask(taskId, 'completed');
      loadTasks();
    } else if (action === 'escalate') {
      await api.updateTask(taskId, 'escalated');
      loadTasks();
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Vendor Call
// ---------------------------------------------------------------------------

async function prepareVendorCall(issueId) {
  try {
    const data = await api.getVendorCall(issueId);
    alert(data.rendered_text);
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Co-op Intelligence
// ---------------------------------------------------------------------------

async function loadCoopReport() {
  const loading = document.getElementById('coop-loading');
  const content = document.getElementById('coop-content');
  const empty = document.getElementById('coop-empty');

  loading.style.display = 'block';
  content.style.display = 'none';
  empty.style.display = 'none';

  // Need a store from last digest
  const storeId = lastDigest?.store_ids?.[0];
  if (!storeId) {
    loading.style.display = 'none';
    empty.style.display = 'block';
    return;
  }

  try {
    const data = await api.getCoopReport(storeId);

    document.getElementById('coop-opportunity').textContent =
      formatDollars(data.total_opportunity);

    renderCoopAlerts(data.report.alerts || []);

    if (data.health_summary) {
      document.getElementById('health-text').textContent = data.health_summary;
      document.getElementById('health-summary').style.display = 'block';
    }

    loading.style.display = 'none';
    content.style.display = 'block';
  } catch (err) {
    loading.innerHTML = `<div style="color: var(--accent-red);">Error: ${err.message}</div>`;
  }
}

function renderCoopAlerts(alerts) {
  const container = document.getElementById('coop-alerts');
  container.innerHTML = '';

  if (alerts.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No co-op alerts at this time.</p></div>';
    return;
  }

  alerts.forEach(alert => {
    const el = document.createElement('div');
    el.className = 'coop-alert';
    el.innerHTML = `
      <div>
        <span class="alert-icon">${alert.alert_type === 'PATRONAGE_LEAKAGE' ? '!!' : '!'}</span>
        <span class="alert-title">${alert.title}</span>
        <span class="alert-dollar">${formatDollars(alert.dollar_impact)}/yr</span>
      </div>
      <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 8px;">
        ${alert.detail}
      </div>
      <div class="alert-recommendation">
        ${alert.recommendation}
      </div>
    `;
    container.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// Diagnostic
// ---------------------------------------------------------------------------

let diagSessionId = null;

// CSV column aliases (flexible mapping for various POS exports)
const SKU_ALIASES = ['SKU', 'sku', 'Item Number', 'Item No', 'Item', 'UPC', 'Product Code', 'Part Number', 'ItemNumber', 'item_number'];
const DESC_ALIASES = ['Description', 'Description ', 'Name', 'Item Name', 'Product Name', 'Item Description', 'ProductName', 'description', 'name'];
const QTY_ALIASES = ['In Stock Qty.', 'In Stock Qty', 'Quantity', 'Qty', 'On Hand', 'QOH', 'Stock', 'Stock Qty', 'Inventory', 'quantity', 'qty', 'on_hand', 'stock'];
const COST_ALIASES = ['Cost', 'Unit Cost', 'Avg Cost', 'Average Cost', 'UnitCost', 'AvgCost', 'cost', 'unit_cost', 'avg_cost', 'Price'];

function findCol(row, aliases) {
  for (const alias of aliases) {
    if (alias in row) return row[alias];
  }
  return '';
}

function parseNum(v) {
  if (!v) return 0;
  const cleaned = String(v).replace(/[,$\s]/g, '');
  const n = parseFloat(cleaned);
  return isNaN(n) ? 0 : n;
}

function parseCSV(text) {
  const lines = text.split('\n');
  if (lines.length < 2) return [];

  // Parse header
  const header = lines[0].split(',').map(h => h.replace(/^"|"$/g, '').trim());
  const items = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Simple CSV parse (handles quoted fields)
    const values = [];
    let current = '';
    let inQuotes = false;
    for (const ch of line) {
      if (ch === '"') { inQuotes = !inQuotes; }
      else if (ch === ',' && !inQuotes) { values.push(current.trim()); current = ''; }
      else { current += ch; }
    }
    values.push(current.trim());

    const row = {};
    header.forEach((h, idx) => { row[h] = values[idx] || ''; });

    const sku = findCol(row, SKU_ALIASES);
    if (!sku) continue;

    items.push({
      sku: sku.trim(),
      description: findCol(row, DESC_ALIASES).trim(),
      stock: parseNum(findCol(row, QTY_ALIASES)),
      cost: parseNum(findCol(row, COST_ALIASES)),
    });
  }
  return items;
}

function initDiagnostic() {
  document.getElementById('diag-start-btn').addEventListener('click', startDiagnostic);

  const restartBtn = document.getElementById('diag-restart-btn');
  if (restartBtn) {
    restartBtn.addEventListener('click', () => {
      diagSessionId = null;
      document.getElementById('diag-start').style.display = 'block';
      document.getElementById('diag-question').style.display = 'none';
      document.getElementById('diag-complete').style.display = 'none';
      document.getElementById('diag-file').value = '';
    });
  }
}

async function startDiagnostic() {
  const fileInput = document.getElementById('diag-file');
  if (!fileInput.files.length) {
    alert('Please select a CSV file');
    return;
  }

  const file = fileInput.files[0];
  const text = await file.text();
  const items = parseCSV(text);

  if (items.length === 0) {
    alert('No valid inventory items found in CSV');
    return;
  }

  // Show loading
  document.getElementById('diag-start').style.display = 'none';
  document.getElementById('diag-loading').style.display = 'block';

  try {
    const data = await api.startDiagnostic(items);
    diagSessionId = data.session_id;

    document.getElementById('diag-loading').style.display = 'none';

    if (data.patterns_detected === 0) {
      alert(`Analyzed ${data.total_items} items. No negative stock patterns detected.`);
      document.getElementById('diag-start').style.display = 'block';
      return;
    }

    // Load first question
    await loadDiagQuestion();
  } catch (err) {
    document.getElementById('diag-loading').style.display = 'none';
    document.getElementById('diag-start').style.display = 'block';
    alert(`Error: ${err.message}`);
  }
}

async function loadDiagQuestion() {
  if (!diagSessionId) return;

  try {
    const q = await api.getDiagnosticQuestion(diagSessionId);

    if (!q) {
      // Session complete
      await showDiagComplete();
      return;
    }

    renderDiagQuestion(q);
    document.getElementById('diag-question').style.display = 'block';
  } catch (err) {
    alert(`Error loading question: ${err.message}`);
  }
}

function renderDiagQuestion(q) {
  // Progress
  const pct = (q.progress.current / q.progress.total) * 100;
  document.getElementById('diag-progress-fill').style.width = `${pct}%`;
  document.getElementById('diag-progress-text').textContent =
    `${q.progress.current} / ${q.progress.total}`;

  // Running totals
  const totals = q.running_totals;
  document.getElementById('diag-total-shrinkage').textContent = formatDollars(totals.total_shrinkage);
  document.getElementById('diag-explained').textContent = formatDollars(totals.explained_value);
  document.getElementById('diag-remaining').textContent = formatDollars(totals.unexplained_value);
  document.getElementById('diag-reduction').textContent =
    `${totals.reduction_percent.toFixed(1)}% reduction`;

  // Pattern info
  document.getElementById('diag-pattern-name').textContent = q.pattern_name;
  document.getElementById('diag-pattern-value').textContent = formatDollars(q.total_value);
  document.getElementById('diag-pattern-count').textContent = `${q.item_count} items`;

  // Question text (convert markdown bold to HTML)
  const questionHtml = q.question.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  document.getElementById('diag-question-text').innerHTML = questionHtml;

  // Sample items
  const samplesEl = document.getElementById('diag-samples');
  if (q.sample_items && q.sample_items.length > 0) {
    samplesEl.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px;">Sample items:</div>' +
      q.sample_items.map(s =>
        `<div style="font-size: 0.8rem; padding: 4px 0; border-bottom: 1px solid var(--border-color);">` +
        `${s.sku} - ${s.description} (${formatDollars(s.value)})` +
        `</div>`
      ).join('');
  } else {
    samplesEl.innerHTML = '';
  }

  // Answer buttons
  const answersEl = document.getElementById('diag-answers');
  answersEl.innerHTML = '';
  q.suggested_answers.forEach(([label, classification]) => {
    const btn = document.createElement('button');
    btn.className = 'btn diag-answer-btn';
    if (classification === 'receiving_gap' || classification === 'non_tracked' || classification === 'vendor_managed' || classification === 'expiration') {
      btn.classList.add('btn-primary');
    }
    btn.textContent = label;
    btn.addEventListener('click', () => submitDiagAnswer(classification));
    answersEl.appendChild(btn);
  });
}

async function submitDiagAnswer(classification) {
  if (!diagSessionId) return;

  // Disable buttons during submission
  document.querySelectorAll('.diag-answer-btn').forEach(b => b.disabled = true);

  try {
    const result = await api.answerDiagnostic(diagSessionId, classification);

    if (result.is_complete) {
      document.getElementById('diag-question').style.display = 'none';
      await showDiagComplete();
    } else if (result.next_question) {
      renderDiagQuestion(result.next_question);
    } else {
      await loadDiagQuestion();
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
    document.querySelectorAll('.diag-answer-btn').forEach(b => b.disabled = false);
  }
}

async function showDiagComplete() {
  try {
    const report = await api.getDiagnosticReport(diagSessionId);
    const summary = report.summary;

    document.getElementById('diag-final-reduction').textContent =
      `${summary.reduction_percent.toFixed(1)}%`;
    document.getElementById('diag-final-explained').textContent =
      `${formatDollars(summary.explained_value)} explained`;
    document.getElementById('diag-final-remaining').textContent =
      `${formatDollars(summary.unexplained_value)} remaining`;

    // Journey
    const journeyEl = document.getElementById('diag-journey');
    journeyEl.innerHTML = '<h3 style="font-size: 1rem; margin: 16px 0 8px;">Classification Journey</h3>';
    report.journey.forEach(step => {
      const el = document.createElement('div');
      el.className = 'task-item';
      const classLabel = step.classification.replace(/_/g, ' ');
      el.innerHTML = `
        <div class="task-info">
          <div class="task-title">${step.pattern}</div>
          <div class="task-meta">${step.items} items &middot; ${formatDollars(step.value)}</div>
        </div>
        <span class="task-status ${step.classification === 'investigate' || step.classification === 'theft' ? 'escalated' : 'completed'}">${classLabel}</span>
      `;
      journeyEl.appendChild(el);
    });

    // Items to investigate
    const invEl = document.getElementById('diag-investigate-list');
    const inv = report.items_to_investigate || [];
    if (inv.length > 0) {
      invEl.innerHTML = '<h3 style="font-size: 1rem; margin: 16px 0 8px;">Top Items to Investigate</h3>';
      inv.slice(0, 10).forEach(item => {
        const el = document.createElement('div');
        el.className = 'task-item';
        el.innerHTML = `
          <div class="task-info">
            <div class="task-title">${item.sku}</div>
            <div class="task-meta">${item.description} &middot; ${item.pattern}</div>
          </div>
          <span class="issue-dollar">${formatDollars(item.value)}</span>
        `;
        invEl.appendChild(el);
      });
    } else {
      invEl.innerHTML = '';
    }

    document.getElementById('diag-question').style.display = 'none';
    document.getElementById('diag-complete').style.display = 'block';
  } catch (err) {
    alert(`Error loading report: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Pull to Refresh
// ---------------------------------------------------------------------------

function initPullToRefresh() {
  let startY = 0;
  let pulling = false;
  const indicator = document.getElementById('pull-indicator');

  document.addEventListener('touchstart', (e) => {
    if (window.scrollY === 0) {
      startY = e.touches[0].clientY;
      pulling = true;
    }
  }, { passive: true });

  document.addEventListener('touchmove', (e) => {
    if (!pulling) return;
    const dy = e.touches[0].clientY - startY;
    if (dy > 60) {
      indicator.classList.add('visible');
    }
  }, { passive: true });

  document.addEventListener('touchend', () => {
    if (indicator.classList.contains('visible')) {
      indicator.classList.remove('visible');
      if (currentView === 'digest-view') loadDigest();
      else if (currentView === 'tasks-view') loadTasks();
      else if (currentView === 'coop-view') loadCoopReport();
    }
    pulling = false;
  });
}

// ---------------------------------------------------------------------------
// Auto-refresh
// ---------------------------------------------------------------------------

let refreshInterval = null;

function startAutoRefresh() {
  if (refreshInterval) clearInterval(refreshInterval);
  refreshInterval = setInterval(() => {
    if (currentView === 'digest-view') loadDigest();
  }, 5 * 60 * 1000); // 5 minutes
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('header-time').textContent = formatTime();

  initAuth();
  initNavigation();
  initPullToRefresh();
  initDiagnostic();

  // Check health first to detect dev mode
  api.health().then(health => {
    if (health.dev_mode) {
      // Dev mode: skip auth, auto-load
      api.setToken('dev-mode');
      showApp();
      loadDigest();
      startAutoRefresh();
    } else if (api.isAuthenticated) {
      showApp();
      loadDigest();
      startAutoRefresh();
    } else {
      showAuthGate();
    }
  }).catch(() => {
    // Health failed — maybe server isn't running
    if (api.isAuthenticated) {
      showApp();
      loadDigest();
    } else {
      showAuthGate();
    }
  });
});
