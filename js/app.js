import { t, setLang, getLang, detectLang, applyTranslations } from './i18n.js';
import { SoundEngine } from './audio.js';
import { PhoneDetector, HAND_CONNECTIONS, SENSITIVITY_THRESHOLD } from './detector.js';

const STORAGE_KEY = 'focusguard.v1';
const RELEASE_MS = 1500;     // phone must be unseen this long to count as released
const MIN_HITS = 3;          // detections needed before a hold counts (filters one-off false positives)
const PRAISE_EVERY_MIN = 5;

// ---------------------------------------------------------------- settings

const PRESETS = {
  gentle: { grace: 6, sound: 'chime', volume: 40, escalate: false, escalateSec: 20, flash: 'glow', minDuration: 0, penaltyMin: 0, stopWhen: 'release' },
  medium: { grace: 3, sound: 'alarm', volume: 70, escalate: true, escalateSec: 20, flash: 'pulse', minDuration: 2, penaltyMin: 1, stopWhen: 'release' },
  strict: { grace: 1, sound: 'siren', volume: 100, escalate: true, escalateSec: 8, flash: 'flash', minDuration: 5, penaltyMin: 3, stopWhen: 'outOfView' }
};

const DEFAULTS = {
  lang: null,
  preset: 'medium',
  severity: { ...PRESETS.medium },
  adhd: { enabled: false, warnings: 1, pomodoro: false, focusMin: 25, breakMin: 5, praise: true, hidePreview: false, reduceMotion: false },
  detection: { mode: 'hand', sensitivity: 'normal', overlay: true, mirror: true },
  privacy: { remember: true }
};

const ADHD_RECOMMENDED = {
  adhd: { enabled: true, warnings: 2, pomodoro: true, focusMin: 20, breakMin: 5, praise: true, hidePreview: true, reduceMotion: true },
  severity: { flash: 'glow', grace: 4 }
};

function clone(o) { return JSON.parse(JSON.stringify(o)); }

function merge(base, extra) {
  const out = clone(base);
  for (const k of Object.keys(extra || {})) {
    if (out[k] && typeof out[k] === 'object' && !Array.isArray(out[k])) Object.assign(out[k], extra[k]);
    else if (k in out) out[k] = extra[k];
  }
  return out;
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return merge(DEFAULTS, JSON.parse(raw));
  } catch { /* storage blocked or corrupt — use defaults */ }
  return clone(DEFAULTS);
}

let settings = loadSettings();

function saveSettings() {
  try {
    if (settings.privacy.remember) localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    else localStorage.removeItem(STORAGE_KEY);
  } catch { /* ignore */ }
}

const getPath = (path) => path.split('.').reduce((o, k) => o?.[k], settings);
function setPath(path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  keys.reduce((o, k) => o[k], settings)[last] = value;
}

// ---------------------------------------------------------------- dom

const $ = (id) => document.getElementById(id);
const els = {
  video: $('video'), overlay: $('overlay'), stage: $('stage'), placeholder: $('placeholder'),
  previewHidden: $('previewHidden'), pill: $('statusPill'), statusText: $('statusText'),
  countdown: $('countdown'), countdownBar: $('countdownBar'), countdownNum: $('countdownNum'),
  startBtn: $('startBtn'), pauseBtn: $('pauseBtn'), togglePreview: $('togglePreview'),
  task: $('taskInput'), timerPhase: $('timerPhase'), timerValue: $('timerValue'), timerProgress: $('timerProgress'),
  alarm: $('alarm'), alarmTitle: $('alarmTitle'), alarmText: $('alarmText'), toasts: $('toasts'),
  drawer: $('drawer'), backdrop: $('drawerBackdrop'), presetSummary: $('presetSummary'),
  stats: { streak: $('statStreak'), best: $('statBest'), violations: $('statViolations'), warnings: $('statWarnings'), phoneTime: $('statPhoneTime') }
};
const ctx2d = els.overlay.getContext('2d');
const sound = new SoundEngine();

// ---------------------------------------------------------------- session state

const S = {
  running: false,
  paused: false,
  stream: null,
  detector: null,
  loop: null,
  previewHidden: false,

  holdSince: null,      // first detection of current hold
  lastHit: 0,
  hits: 0,
  violation: null,      // { kind: 'alarm'|'warning', start }
  warningsUsed: 0,

  sessionStart: 0,
  phoneFreeSince: 0,
  bestStreak: 0,
  violations: 0,
  warnings: 0,
  phoneMs: 0,
  lastTick: 0,
  lastPraiseMin: 0,

  phase: 'session',     // 'session' | 'focus' | 'break'
  phaseEnd: 0,
  phaseLen: 0,
  pausedAt: 0
};

// ---------------------------------------------------------------- helpers

function fmt(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}

function toast(text, kind = 'info', ms = 4200) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = text;
  els.toasts.appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 400); }, ms);
}

function setI18nText(el, key) {
  el.dataset.i18n = key;
  el.textContent = t(key);
}

function setStatus(state, vars) {
  els.pill.dataset.state = state;
  els.statusText.removeAttribute('data-i18n');
  els.statusText.textContent = t(`status.${state}`, vars);
}

// ---------------------------------------------------------------- settings UI

function renderOutputs() {
  document.querySelectorAll('output[data-for]').forEach((out) => {
    const input = $(out.dataset.for);
    const path = input.dataset.setting;
    const v = Number(input.value);
    if (path === 'severity.volume') out.textContent = `${v}%`;
    else if (/Min$/.test(path)) out.textContent = v === 0 && path === 'severity.penaltyMin' ? t('off') : t('minutes', { n: v });
    else if (path === 'adhd.warnings') out.textContent = v === 0 ? t('off') : String(v);
    else out.textContent = v === 0 ? t('off') : t('seconds', { n: v });
  });
}

function syncSettingsUI() {
  document.querySelectorAll('input[data-setting]').forEach((input) => {
    const v = getPath(input.dataset.setting);
    if (input.type === 'checkbox') input.checked = !!v;
    else input.value = v;
  });
  document.querySelectorAll('.segmented[data-setting]').forEach((group) => {
    const v = getPath(group.dataset.setting);
    group.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.value === v));
  });
  document.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('active', b.dataset.preset === settings.preset));
  document.querySelectorAll('[data-show-if]').forEach((el) => { el.hidden = !getPath(el.dataset.showIf); });
  document.querySelectorAll('.lang-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.lang === getLang()));
  renderOutputs();
  renderPresetSummary();
  applyBodyModes();
}

function renderPresetSummary() {
  const s = settings.severity;
  els.presetSummary.textContent = t('presetSummary', {
    grace: s.grace ? t('seconds', { n: s.grace }) : '0 s',
    sound: t(`sound.${s.sound}`),
    volume: s.volume,
    escalate: s.escalate ? t('summaryEscalate') : '',
    penalty: s.penaltyMin ? t('summaryPenalty', { n: s.penaltyMin }) : ''
  });
}

function applyBodyModes() {
  const b = document.body.classList;
  b.toggle('adhd', settings.adhd.enabled);
  b.toggle('reduce-motion', settings.adhd.reduceMotion);
  b.toggle('mirror', settings.detection.mirror);
}

function onSettingChanged(path) {
  if (path.startsWith('severity.')) settings.preset = 'custom';
  if (path === 'detection.sensitivity') S.detector?.setThreshold(thresholdFor());
  if (path === 'adhd.pomodoro' && S.running) resetPhase();
  if (path === 'adhd.enabled' && settings.adhd.enabled && settings.severity.flash === 'flash') {
    settings.severity.flash = 'pulse'; // no intense flashing in ADHD mode
  }
  saveSettings();
  syncSettingsUI();
}

function applyPreset(name) {
  settings.preset = name;
  if (PRESETS[name]) Object.assign(settings.severity, PRESETS[name]);
  if (settings.adhd.enabled && settings.severity.flash === 'flash') settings.severity.flash = 'pulse';
  saveSettings();
  syncSettingsUI();
}

function bindSettings() {
  document.querySelectorAll('input[data-setting]').forEach((input) => {
    input.addEventListener('input', () => {
      const path = input.dataset.setting;
      setPath(path, input.type === 'checkbox' ? input.checked : Number(input.value));
      if (input.type === 'range') { renderOutputs(); }
      onSettingChanged(path);
    });
  });
  document.querySelectorAll('.segmented[data-setting]').forEach((group) => {
    group.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-value]');
      if (!btn) return;
      setPath(group.dataset.setting, btn.dataset.value);
      onSettingChanged(group.dataset.setting);
    });
  });
  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => applyPreset(b.dataset.preset)));

  $('testSound').addEventListener('click', () => {
    sound.start({ ...settings.severity, escalate: false });
    setTimeout(() => { if (!S.violation || S.violation.kind !== 'alarm') sound.stop(); }, 2000);
  });

  $('adhdPreset').addEventListener('click', () => {
    Object.assign(settings.adhd, ADHD_RECOMMENDED.adhd);
    Object.assign(settings.severity, ADHD_RECOMMENDED.severity);
    settings.preset = 'custom';
    saveSettings();
    syncSettingsUI();
    if (S.running) { resetPhase(); setPreviewHidden(settings.adhd.hidePreview); }
    toast(t('adhdApplied'), 'good');
  });

  $('forgetAll').addEventListener('click', () => {
    try { localStorage.clear(); sessionStorage.clear(); } catch { /* ignore */ }
    settings = clone(DEFAULTS);
    settings.privacy.remember = false;
    setLang(detectLang());
    syncSettingsUI();
    toast(t('forgotten'), 'good');
  });

  // drawer & tabs
  const open = () => { els.drawer.classList.add('open'); els.drawer.setAttribute('aria-hidden', 'false'); els.backdrop.hidden = false; };
  const close = () => { els.drawer.classList.remove('open'); els.drawer.setAttribute('aria-hidden', 'true'); els.backdrop.hidden = true; };
  $('openSettings').addEventListener('click', open);
  $('closeSettings').addEventListener('click', close);
  els.backdrop.addEventListener('click', close);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  $('tabs').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-tab]');
    if (!btn) return;
    document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === btn.dataset.tab));
  });

  document.querySelectorAll('.lang-toggle button').forEach((b) => b.addEventListener('click', () => {
    settings.lang = b.dataset.lang;
    setLang(settings.lang);
    saveSettings();
    syncSettingsUI();
    refreshDynamicText();
  }));
}

function refreshDynamicText() {
  if (S.violation) showAlarm(S.violation.kind);
  if (!S.running) setStatus('idle');
  else if (S.paused) setStatus('paused');
  else if (S.phase === 'break') setStatus('break');
  else if (!S.violation && !S.holdSince) setStatus('watching');
  updateTimer(performance.now());
}

// ---------------------------------------------------------------- session control

function thresholdFor() {
  return SENSITIVITY_THRESHOLD[settings.detection.sensitivity] ?? SENSITIVITY_THRESHOLD.normal;
}

async function startSession() {
  sound.unlock();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    toast(t('errInsecure'), 'bad', 8000);
    return;
  }
  els.startBtn.disabled = true;
  try {
    setStatus('camera');
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
      audio: false
    });
  } catch (e) {
    console.error(e);
    toast(t('errCamera'), 'bad', 8000);
    setStatus('idle');
    els.startBtn.disabled = false;
    return;
  }
  els.video.srcObject = S.stream;
  await els.video.play();
  els.placeholder.hidden = true;

  if (!S.detector) {
    setStatus('loading');
    try {
      const det = new PhoneDetector();
      await det.init(thresholdFor());
      S.detector = det;
    } catch (e) {
      console.error(e);
      toast(t('errModels'), 'bad', 8000);
      stopSession();
      return;
    }
  }

  const now = performance.now();
  Object.assign(S, {
    running: true, paused: false, holdSince: null, hits: 0, violation: null, warningsUsed: 0,
    sessionStart: now, phoneFreeSince: now, bestStreak: 0, violations: 0, warnings: 0,
    phoneMs: 0, lastTick: now, lastPraiseMin: 0
  });
  resetPhase();
  setPreviewHidden(settings.adhd.hidePreview);
  els.startBtn.disabled = false;
  els.startBtn.classList.add('running');
  setI18nText(els.startBtn.firstElementChild, 'stop');
  els.pauseBtn.disabled = false;
  els.togglePreview.disabled = false;
  setStatus('watching');
  scheduleLoop();
}

function stopSession() {
  S.running = false;
  endViolation(true);
  cancelLoop();
  S.stream?.getTracks().forEach((tr) => tr.stop());
  S.stream = null;
  els.video.srcObject = null;
  els.placeholder.hidden = false;
  els.previewHidden.hidden = true;
  els.video.hidden = false;
  els.countdown.hidden = true;
  ctx2d.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.startBtn.disabled = false;
  els.startBtn.classList.remove('running');
  setI18nText(els.startBtn.firstElementChild, 'start');
  els.pauseBtn.disabled = true;
  setI18nText(els.pauseBtn, 'pause');
  els.togglePreview.disabled = true;
  S.paused = false;
  setStatus('idle');
}

function togglePause() {
  if (!S.running) return;
  const now = performance.now();
  S.paused = !S.paused;
  if (S.paused) {
    S.pausedAt = now;
    endViolation(true);
    resetHold();
    setStatus('paused');
    setI18nText(els.pauseBtn, 'resume');
  } else {
    const gap = now - S.pausedAt;
    S.phaseEnd += gap;
    S.sessionStart += gap;
    S.phoneFreeSince += gap;
    S.lastTick = now;
    setStatus(S.phase === 'break' ? 'break' : 'watching');
    setI18nText(els.pauseBtn, 'pause');
  }
}

function setPreviewHidden(hidden) {
  S.previewHidden = hidden;
  els.video.hidden = hidden;
  els.overlay.hidden = hidden;
  els.previewHidden.hidden = !hidden || !S.running;
  setI18nText(els.togglePreview, hidden ? 'showPreview' : 'hidePreview');
}

// ---------------------------------------------------------------- focus timer

function resetPhase() {
  const now = performance.now();
  S.warningsUsed = 0;
  if (settings.adhd.pomodoro) {
    S.phase = 'focus';
    S.phaseLen = settings.adhd.focusMin * 60000;
    S.phaseEnd = now + S.phaseLen;
  } else {
    S.phase = 'session';
    S.phaseLen = 0;
  }
  updateTimer(now);
}

function updateTimer(now) {
  if (!S.running) {
    els.timerValue.textContent = '0:00';
    els.timerProgress.style.width = '0%';
    setI18nText(els.timerPhase, 'phase.session');
    return;
  }
  const ref = S.paused ? S.pausedAt : now;
  setI18nText(els.timerPhase, `phase.${S.phase}`);
  els.timerPhase.dataset.phase = S.phase;
  if (S.phase === 'session') {
    els.timerValue.textContent = fmt(ref - S.sessionStart);
    els.timerProgress.style.width = '0%';
    return;
  }
  const left = S.phaseEnd - ref;
  els.timerValue.textContent = fmt(left);
  els.timerProgress.style.width = `${Math.min(100, 100 * (1 - left / S.phaseLen))}%`;
  if (left <= 0 && !S.paused) advancePhase(now);
}

function advancePhase(now) {
  if (S.phase === 'focus') {
    endViolation(true);
    resetHold();
    S.phase = 'break';
    S.phaseLen = settings.adhd.breakMin * 60000;
    S.phaseEnd = now + S.phaseLen;
    sound.chime(60, [523.25, 659.25, 783.99]);
    toast(t('breakStart', { n: settings.adhd.breakMin }), 'good', 7000);
    setStatus('break');
  } else {
    S.phase = 'focus';
    S.phaseLen = settings.adhd.focusMin * 60000;
    S.phaseEnd = now + S.phaseLen;
    S.warningsUsed = 0;
    S.phoneFreeSince = now;
    sound.chime(60, [783.99, 659.25, 523.25]);
    toast(t('breakEnd'), 'info', 7000);
    setStatus('watching');
  }
}

// ---------------------------------------------------------------- detection loop

function scheduleLoop() {
  cancelLoop();
  const step = () => {
    tick();
    if (!S.running) return;
    // rAF stops in background tabs; fall back to a slower timer there.
    S.loop = document.hidden
      ? { id: setTimeout(step, 250), t: 'to' }
      : { id: requestAnimationFrame(step), t: 'raf' };
  };
  step();
}

function cancelLoop() {
  if (!S.loop) return;
  if (S.loop.t === 'raf') cancelAnimationFrame(S.loop.id); else clearTimeout(S.loop.id);
  S.loop = null;
}

document.addEventListener('visibilitychange', () => { if (S.running) scheduleLoop(); });

function resetHold() {
  S.holdSince = null;
  S.hits = 0;
  els.countdown.hidden = true;
}

function tick() {
  const now = performance.now();
  const dt = now - S.lastTick;
  S.lastTick = now;
  updateTimer(now);
  if (!S.running) return;

  if (S.paused || S.phase === 'break') {
    drawOverlay(null);
    updateStats(now);
    return;
  }

  let res = null;
  try { res = S.detector.detect(els.video); } catch (e) { console.error(e); }
  drawOverlay(res);
  if (!res) return;

  // What counts as "the phone is still there" depends on settings and on
  // whether we are already punishing (stricter "out of sight" stop rule).
  let hit = settings.detection.mode === 'any' ? res.phoneVisible : res.inHand;
  if (S.violation && settings.severity.stopWhen === 'outOfView') hit = res.phoneVisible;

  if (hit) {
    if (!S.holdSince) { S.holdSince = now; S.hits = 0; }
    S.hits++;
    S.lastHit = now;
  }
  const holding = S.holdSince && now - S.lastHit < RELEASE_MS;
  if (holding && S.hits >= MIN_HITS) S.phoneMs += dt;

  if (!holding && S.holdSince) {
    resetHold();
    if (!S.violation) setStatus('watching');
  }

  if (holding && S.hits >= MIN_HITS && !S.violation) {
    const graceMs = settings.severity.grace * 1000;
    const left = graceMs - (now - S.holdSince);
    if (left <= 0) startViolation(now);
    else showCountdown(left, graceMs);
  }

  // Ignoring a warning and keeping the phone escalates to the real siren.
  if (S.violation?.kind === 'warning' && holding &&
      now - S.violation.start >= Math.max(3, settings.severity.grace) * 1000) {
    startAlarm(now);
  }

  if (S.violation && !holding) {
    const minMs = S.violation.kind === 'alarm' ? settings.severity.minDuration * 1000 : 0;
    if (now - S.violation.start >= minMs) endViolation(false, now);
  }

  updateStats(now);
}

function showCountdown(left, total) {
  els.countdown.hidden = false;
  const secs = Math.ceil(left / 1000);
  els.countdownNum.textContent = secs;
  const c = 2 * Math.PI * 44;
  els.countdownBar.style.strokeDasharray = c;
  els.countdownBar.style.strokeDashoffset = c * (1 - left / total);
  setStatus('pending', { s: t('seconds', { n: secs }) });
}

function startViolation(now) {
  els.countdown.hidden = true;
  const warnLeft = settings.adhd.enabled ? settings.adhd.warnings - S.warningsUsed : 0;
  const streak = now - S.phoneFreeSince;
  S.bestStreak = Math.max(S.bestStreak, streak);

  if (warnLeft > 0) {
    S.warningsUsed++;
    S.warnings++;
    S.violation = { kind: 'warning', start: now };
    sound.chime(Math.min(60, settings.severity.volume), [659.25, 523.25]);
    showAlarm('warning');
    setStatus('warning');
    return;
  }
  startAlarm(now);
}

function startAlarm(now) {
  S.violations++;
  S.violation = { kind: 'alarm', start: now };
  sound.start(settings.severity);
  showAlarm('alarm');
  setStatus('alarm');
  if (S.phase === 'focus' && settings.severity.penaltyMin > 0) {
    S.phaseEnd += settings.severity.penaltyMin * 60000;
    S.phaseLen += settings.severity.penaltyMin * 60000;
    toast(t('penaltyAdded', { n: settings.severity.penaltyMin }), 'bad');
  }
}

function endViolation(silent, now = performance.now()) {
  if (!S.violation) return;
  const { kind, start } = S.violation;
  S.violation = null;
  sound.stop();
  els.alarm.hidden = true;
  els.alarm.className = 'alarm';
  document.body.classList.remove('alarming');
  S.phoneFreeSince = now;
  S.lastPraiseMin = 0;
  if (silent) return;
  setStatus('watching');
  if (settings.adhd.enabled && settings.adhd.praise) {
    const task = els.task.value.trim();
    const secs = Math.max(1, Math.round((now - start) / 1000));
    if (task) toast(t('praiseBack', { task }), 'good');
    else if (kind === 'alarm' && secs <= 5) toast(t('praiseQuick', { s: secs }), 'good');
  }
}

function showAlarm(kind) {
  const adhd = settings.adhd.enabled;
  const task = els.task.value.trim();
  els.alarm.hidden = false;
  if (kind === 'warning') {
    els.alarm.className = 'alarm warning';
    els.alarmTitle.textContent = t('warningTitle');
    els.alarmText.textContent = t('warningText', { n: settings.adhd.warnings - S.warningsUsed });
    return;
  }
  let flash = settings.severity.flash;
  if (adhd && flash === 'flash') flash = 'pulse';
  els.alarm.className = `alarm fx-${flash}`;
  document.body.classList.add('alarming');
  els.alarmTitle.textContent = t(adhd ? 'alarmTitleAdhd' : 'alarmTitle');
  const how = t(settings.severity.stopWhen === 'outOfView' ? 'alarmTextOut' : 'alarmText');
  els.alarmText.textContent = task ? `${how} ${t('alarmTextTask', { task })}` : how;
}

function updateStats(now) {
  const streak = S.violation || S.paused ? 0 : now - S.phoneFreeSince;
  if (!S.violation) S.bestStreak = Math.max(S.bestStreak, streak);
  els.stats.streak.textContent = fmt(streak);
  els.stats.best.textContent = fmt(S.bestStreak);
  els.stats.violations.textContent = S.violations;
  els.stats.warnings.textContent = S.warnings;
  els.stats.phoneTime.textContent = fmt(S.phoneMs);

  if (settings.adhd.enabled && settings.adhd.praise && S.phase !== 'break' && !S.paused) {
    const mins = Math.floor(streak / 60000);
    if (mins >= PRAISE_EVERY_MIN && mins % PRAISE_EVERY_MIN === 0 && mins !== S.lastPraiseMin) {
      S.lastPraiseMin = mins;
      toast(t('praiseStreak', { n: mins }), 'good');
      sound.chime(30, [783.99, 1046.5]);
    }
  }
}

// ---------------------------------------------------------------- overlay drawing

function drawOverlay(res) {
  const c = els.overlay;
  const w = c.clientWidth;
  const h = c.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx2d.clearRect(0, 0, w, h);
  if (!res || !settings.detection.overlay || S.previewHidden) return;

  // Map normalised video coords to the element, matching object-fit: cover.
  const vw = els.video.videoWidth;
  const vh = els.video.videoHeight;
  const scale = Math.max(w / vw, h / vh);
  const ox = (w - vw * scale) / 2;
  const oy = (h - vh * scale) / 2;
  const X = (x) => ox + x * vw * scale;
  const Y = (y) => oy + y * vh * scale;

  const styles = getComputedStyle(document.body);
  const accent = styles.getPropertyValue('--accent').trim() || '#5eead4';
  const danger = styles.getPropertyValue('--danger').trim() || '#f43f5e';

  for (const p of res.phones) {
    ctx2d.strokeStyle = res.inHand ? danger : '#facc15';
    ctx2d.lineWidth = 3;
    ctx2d.setLineDash([8, 6]);
    roundRect(X(p.x), Y(p.y), p.w * vw * scale, p.h * vh * scale, 10);
    ctx2d.stroke();
    ctx2d.setLineDash([]);
  }

  ctx2d.lineWidth = 2.5;
  ctx2d.strokeStyle = accent;
  ctx2d.fillStyle = accent;
  for (const hand of res.hands) {
    ctx2d.globalAlpha = 0.85;
    ctx2d.beginPath();
    for (const { start, end } of HAND_CONNECTIONS) {
      ctx2d.moveTo(X(hand[start].x), Y(hand[start].y));
      ctx2d.lineTo(X(hand[end].x), Y(hand[end].y));
    }
    ctx2d.stroke();
    for (const p of hand) {
      ctx2d.beginPath();
      ctx2d.arc(X(p.x), Y(p.y), 3.2, 0, Math.PI * 2);
      ctx2d.fill();
    }
    ctx2d.globalAlpha = 1;
  }
}

function roundRect(x, y, w, h, r) {
  ctx2d.beginPath();
  ctx2d.moveTo(x + r, y);
  ctx2d.arcTo(x + w, y, x + w, y + h, r);
  ctx2d.arcTo(x + w, y + h, x, y + h, r);
  ctx2d.arcTo(x, y + h, x, y, r);
  ctx2d.arcTo(x, y, x + w, y, r);
  ctx2d.closePath();
}

// ---------------------------------------------------------------- boot

els.startBtn.addEventListener('click', () => (S.running ? stopSession() : startSession()));
els.pauseBtn.addEventListener('click', togglePause);
els.togglePreview.addEventListener('click', () => setPreviewHidden(!S.previewHidden));
setInterval(() => { if (S.running && (S.paused || document.hidden)) updateTimer(performance.now()); }, 1000);

setLang(settings.lang || detectLang());
applyTranslations();
bindSettings();
syncSettingsUI();
setStatus('idle');
updateTimer(performance.now());
