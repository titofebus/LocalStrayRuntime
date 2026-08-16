// DFlash Utility Web Client with real-time WebSockets and implicit transitions
let ws = null;
let reconnectTimer = null;

// DOM Elements
const overallProgressFill = document.getElementById('overall-progress-fill');
const overallProgressPct = document.getElementById('overall-progress-pct');
const metricElapsed = document.getElementById('metric-elapsed');
const metricEta = document.getElementById('metric-eta');
const metricLoss = document.getElementById('metric-loss');
const metricSpeed = document.getElementById('metric-speed');
const metricMemory = document.getElementById('metric-memory');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const actionBanner = document.getElementById('action-banner');
const bannerTitle = document.getElementById('banner-title');
const bannerDesc = document.getElementById('banner-desc');
const stagesList = document.getElementById('stages-list');
const terminalLogs = document.getElementById('terminal-logs');

const btnStart = document.getElementById('btn-start');
const btnPause = document.getElementById('btn-pause');
const btnReset = document.getElementById('btn-reset');

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] Connected to DFlash Orchestrator.');
    clearTimeout(reconnectTimer);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateUI(data);
    } catch (e) {
      console.error('[WS] Parse error:', e);
    }
  };

  ws.onclose = () => {
    console.log('[WS] Disconnected. Reconnecting in 2s...');
    reconnectTimer = setTimeout(connectWebSocket, 2000);
  };
}

function updateUI(state) {
  // 1. Overall Progress
  const pct = state.overall_progress_pct || 0;
  overallProgressFill.style.width = `${pct}%`;
  overallProgressPct.innerText = `${pct.toFixed(1)}%`;

  if (pct >= 100) {
    overallProgressFill.classList.add('success');
  } else {
    overallProgressFill.classList.remove('success');
  }

  // 2. Status Badge
  statusBadge.className = `status-badge ${state.status}`;
  statusText.innerText = state.status.toUpperCase();

  // Button state toggle
  if (state.status === 'running') {
    btnStart.style.display = 'none';
    btnPause.style.display = 'inline-flex';
  } else {
    btnStart.style.display = 'inline-flex';
    btnPause.style.display = 'none';
  }

  // 3. Telemetry Metrics
  metricElapsed.innerText = formatTime(state.elapsed_seconds || 0);
  metricEta.innerText = state.status === 'completed' ? 'Done' : formatTime(state.eta_seconds || 0);
  metricLoss.innerText = state.current_loss !== null ? state.current_loss.toFixed(4) : '--';
  metricSpeed.innerText = state.target_speed_tps !== null ? `${state.target_speed_tps.toFixed(1)} tok/s` : (state.acceptance_rate ? `${state.acceptance_rate}% acc` : '--');
  metricMemory.innerText = `${(state.memory_used_gb || 0).toFixed(1)} GB`;

  // 4. Action Banner
  if (state.status === 'completed') {
    actionBanner.style.display = 'flex';
    actionBanner.className = 'action-banner completed';
    bannerTitle.innerText = 'DFlash Distillation Complete!';
    bannerDesc.innerText = 'Qwen3.8-27B DFlash model weights are compiled and ready for 50+ tok/s speculative inference.';
  } else if (state.status === 'running') {
    actionBanner.style.display = 'flex';
    actionBanner.className = 'action-banner';
    bannerTitle.innerText = 'Training Pipeline Active';
    bannerDesc.innerText = 'Processing on Apple M4 Max GPU. Checkpoints are automatically saved after each stage.';
  } else if (state.status === 'paused') {
    actionBanner.style.display = 'flex';
    actionBanner.className = 'action-banner';
    bannerTitle.innerText = 'Pipeline Paused';
    bannerDesc.innerText = 'State preserved in SSD checkpoint. Click Resume to continue without loss of progress.';
  } else {
    actionBanner.style.display = 'none';
  }

  // 5. Render Stages
  renderStages(state.stages, state.active_stage_id);

  // 6. Logs
  renderLogs(state.recent_logs || []);
}

function renderStages(stages, activeId) {
  if (!stages) return;
  stagesList.innerHTML = '';

  stages.forEach((stage, idx) => {
    const card = document.createElement('div');
    card.className = `stage-card ${stage.status} ${stage.id === activeId ? 'active' : ''}`;

    let badgeIcon = idx + 1;
    if (stage.status === 'completed') {
      badgeIcon = '✓';
    }

    card.innerHTML = `
      <div class="stage-left">
        <div class="stage-badge">${badgeIcon}</div>
        <div class="stage-info">
          <h3>${stage.name}</h3>
          <p>${stage.description}</p>
          ${stage.details ? `<div class="stage-details-text">${stage.details}</div>` : ''}
        </div>
      </div>
      <div class="stage-right">
        <div class="stage-progress-mini">
          <div class="stage-progress-fill" style="width: ${stage.progress_pct}%"></div>
        </div>
        <div class="stage-pct">${stage.progress_pct.toFixed(0)}%</div>
      </div>
    `;

    stagesList.appendChild(card);
  });
}

function renderLogs(logs) {
  terminalLogs.innerHTML = '';
  logs.forEach(line => {
    const el = document.createElement('div');
    el.className = 'log-line';
    el.innerText = line;
    terminalLogs.appendChild(el);
  });
  terminalLogs.scrollTop = terminalLogs.scrollHeight;
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s < 10 ? '0' : ''}${s}s`;
}

// Event Listeners
btnStart.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'start' }));
  } else {
    fetch('/api/start', { method: 'POST' });
  }
});

btnPause.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'pause' }));
  } else {
    fetch('/api/pause', { method: 'POST' });
  }
});

btnReset.addEventListener('click', () => {
  if (confirm('Reset training pipeline to initial state?')) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'reset' }));
    } else {
      fetch('/api/reset', { method: 'POST' });
    }
  }
});

// Request desktop notification permission on launch
if ('Notification' in window && Notification.permission !== 'granted') {
  Notification.requestPermission();
}

// Start WebSocket connection
connectWebSocket();
