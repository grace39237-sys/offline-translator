// app.js — 오프라인 번역기 UI + 흐름.
import * as ort from 'onnxruntime-web';
import { loadASR, transcribe, ASR_MODEL_KO, ASR_MODEL_EN } from './asr.js';
import { loadMT, translate, MT_MODEL_KO_EN, MT_MODEL_EN_KO } from './mt.js';
import { VoiceCapture } from './vad.js';

// 오프라인 동작: ONNX WASM 파일은 저장소에서 직접 가져온다.
// transformers.js는 백엔드를 첫 모델 로드 때 늦게 초기화해서
// env.backends.onnx가 처음엔 비어 있다. 그래서 ORT env에 직접 지정한다.
// (import map 덕분에 transformers.js가 쓰는 것과 같은 ort 인스턴스다.)
// GitHub Pages는 COOP/COEP 헤더가 없어 SharedArrayBuffer가 안 되므로
// 스레드는 1개로 고정한다.
// NOTE: wasmPaths에 상대 경로('./vendor/ort/')를 쓰면 ORT 번들 자신의
// 위치(vendor/ort/)를 기준으로 다시 해석돼서
// vendor/ort/vendor/ort/ort-wasm-....mjs 로 이중 경로가 된다
// (라이브 브라우저 테스트에서 404로 확인됨). 그래서 절대 URL로 지정한다.
ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(() => {});
}

// 저장한 모델을 브라우저가 용량 부족 이유로 지우지 않게 영구 저장소로 요청한다.
// (안 되면 무시 — 저장 자체는 그대로 동작한다.)
if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});

const DIRECTIONS = {
  'ko-en': { from: '한국어', to: 'English', asr: ASR_MODEL_KO, asrLang: 'korean', mt: MT_MODEL_KO_EN, size: '약 190MB', sizeBytes: 190 * 1048576, asrLabel: 'base-ko' },
  'en-ko': { from: 'English', to: '한국어', asr: ASR_MODEL_EN, asrLang: 'english', mt: MT_MODEL_EN_KO, size: '약 300MB', sizeBytes: 300 * 1048576, asrLabel: 'tiny-en' },
};
const APP_VERSION = 'v9';
const TARGET_LANG = { 'ko-en': 'en-US', 'en-ko': 'ko-KR' };

const $ = (id) => document.getElementById(id);
const els = {
  statusDot: $('statusDot'), statusText: $('statusText'),
  langFrom: $('langFrom'), langTo: $('langTo'), swapBtn: $('swapBtn'),
  modelBox: $('modelBox'), modelInfo: $('modelInfo'),
  downloadBtn: $('downloadBtn'), downloadProgress: $('downloadProgress'),
  downloadBar: $('downloadBar'), downloadLabel: $('downloadLabel'),
  inputText: $('inputText'), micBtn: $('micBtn'), convBtn: $('convBtn'),
  modeTranslateBtn: $('modeTranslateBtn'), modeDictateBtn: $('modeDictateBtn'),
  recognizedLabel: $('recognizedLabel'),
  translateBtn: $('translateBtn'), listening: $('listening'),
  listeningText: $('listeningText'), stopListenBtn: $('stopListenBtn'),
  micHint: $('micHint'),
  recognizedWrap: $('recognizedWrap'), recognizedText: $('recognizedText'),
  retranslateBtn: $('retranslateBtn'),
  resultWrap: $('resultWrap'), resultText: $('resultText'), resultSrc: $('resultSrc'),
  speakBtn: $('speakBtn'), copyBtn: $('copyBtn'),
  convWrap: $('convWrap'), convList: $('convList'), convClearBtn: $('convClearBtn'),
  historyList: $('historyList'), historyEmpty: $('historyEmpty'),
  historyClearBtn: $('historyClearBtn'),
  modelLine: $('modelLine'),
};

let direction = 'ko-en';
let inputMode = 'translate'; // 'translate' | 'dictate' — 받아쓰기 모드에서는 인식만 하고 번역은 안 한다.
const loaded = {}; // dir -> { asr, mt }
const progressFiles = new Map();
let capture = null;
let captureMode = null; // 'single' | 'conv'
let convOn = false;

function setStatus(kind, text) {
  els.statusDot.className = `status-dot ${kind}`;
  els.statusText.textContent = text;
}

function fmtMB(n) {
  return n > 0 ? `${(n / 1048576).toFixed(0)}MB` : '';
}

let progressExpected = 0; // 현재 받는 방향의 전체 예상 용량(바이트)

function onProgress(info) {
  if (info.status === 'progress' && info.total) {
    progressFiles.set(info.file, { loaded: info.loaded, total: info.total });
  } else if (info.status === 'done' && info.file) {
    const prev = progressFiles.get(info.file);
    progressFiles.set(info.file, { loaded: prev?.total || 0, total: prev?.total || 0 });
  }
  // 파일별 합계가 아니라 방향 전체 예상 용량 기준으로 보여준다.
  // (파일별 합계는 첫 파일이 끝나면 100%로 보여서 오해가 생긴다.)
  let loadedBytes = 0;
  for (const f of progressFiles.values()) { loadedBytes += f.loaded; }
  if (progressExpected > 0) {
    const pct = Math.min(100, Math.round((loadedBytes / progressExpected) * 100));
    els.downloadBar.style.width = `${pct}%`;
    els.downloadLabel.textContent = `${fmtMB(loadedBytes)} / 약 ${fmtMB(progressExpected)} (${pct}%)`;
  }
}

function updateModelBox() {
  const d = DIRECTIONS[direction];
  els.langFrom.textContent = d.from;
  els.langTo.textContent = d.to;
  // 화면 아래에 쓰는 버전·모델 표시: 폰에 새 버전이 올라왔는지 확인용.
  if (els.modelLine) els.modelLine.textContent = `음성인식: ${d.asrLabel} · 앱 ${APP_VERSION}`;
  if (loaded[direction]) {
    els.modelBox.hidden = true;
    setStatus('ready', '오프라인 준비 완료');
    els.micBtn.disabled = false;
    els.convBtn.disabled = false;
  } else {
    els.modelBox.hidden = false;
    els.modelInfo.textContent = `이 방향(${d.from} → ${d.to})으로 쓰려면 모델을 받아야 합니다. 처음 한 번만 인터넷이 필요합니다. 크기 ${d.size}.`;
    els.downloadBtn.textContent = `모델 받기 (${d.size})`;
    els.downloadBtn.disabled = false;
    els.downloadProgress.hidden = true;
    els.downloadLabel.hidden = true;
    setStatus('off', '모델이 필요합니다');
    els.micBtn.disabled = true;
    els.convBtn.disabled = true;
  }
}

// ---- 번역 / 받아쓰기 모드 ----
function setMode(next) {
  if (inputMode === next) return;
  // 듣는 중이면 멈추고 바꾼다. 대화 모드가 켜져 있었으면 끈다.
  if (capture) stopCapture();
  if (convOn) {
    convOn = false;
    els.convBtn.classList.remove('active');
    els.convBtn.textContent = '대화 모드';
  }
  inputMode = next;
  els.modeTranslateBtn.setAttribute('aria-pressed', String(next === 'translate'));
  els.modeDictateBtn.setAttribute('aria-pressed', String(next === 'dictate'));
  // 대화 모드는 번역 대화 기능이라 받아쓰기 모드에서는 안 보인다.
  els.convBtn.hidden = next === 'dictate';
  els.recognizedLabel.textContent = next === 'dictate'
    ? '받아쓴 글자 — 말할 때마다 이어붙습니다. 틀린 부분은 고치세요'
    : '인식된 글자 — 번역이 바로 나옵니다. 틀린 부분은 고치고 "번역하기"를 누르세요';
  els.micHint.textContent = next === 'dictate'
    ? '마이크: 말하면 0.9초 침묵 뒤 자동으로 인식해서 아래에 이어붙습니다. 그만 들으려면 마이크를 다시 누르세요. (최대 30초)'
    : '마이크: 말하면 0.9초 침묵 뒤 자동으로 인식합니다. (최대 30초)';
}

async function ensureModels(dir) {
  if (loaded[dir]) return loaded[dir];
  const d = DIRECTIONS[dir];
  progressFiles.clear();
  progressExpected = d.sizeBytes;
  els.downloadBtn.disabled = true;
  els.downloadProgress.hidden = false;
  els.downloadLabel.hidden = false;
  els.downloadBar.style.width = '0%';
  els.downloadLabel.textContent = '받는 중…';
  setStatus('busy', '모델 받는 중…');
  try {
    const asr = await loadASR(d.asr, onProgress);
    const mt = await loadMT(d.mt, onProgress);
    loaded[dir] = { asr, mt };
    updateModelBox();
    return loaded[dir];
  } catch (e) {
    console.error(e);
    setStatus('off', '모델 받기 실패 — 다시 시도해 주세요');
    els.downloadBtn.disabled = false;
    els.downloadLabel.textContent = `받기 실패: ${e.message}`;
    throw e;
  }
}

function saveHistory(src, dst, dir) {
  try {
    const list = JSON.parse(localStorage.getItem('ot-history') || '[]');
    const item = { src, dst, dir: dir || direction, ts: Date.now() };
    const next = [item, ...list.filter((h) => h.src !== src)].slice(0, 30);
    localStorage.setItem('ot-history', JSON.stringify(next));
    renderHistory();
  } catch (_) {}
}

function renderHistory() {
  let list = [];
  try { list = JSON.parse(localStorage.getItem('ot-history') || '[]'); } catch (_) {}
  els.historyList.innerHTML = '';
  els.historyEmpty.hidden = list.length > 0;
  for (const h of list) {
    const li = document.createElement('li');
    const d = DIRECTIONS[h.dir] || DIRECTIONS['ko-en'];
    li.innerHTML = `<div class="src"></div><div class="dst"></div><div class="meta">${d.from} → ${d.to}</div>`;
    li.querySelector('.src').textContent = h.src;
    li.querySelector('.dst').textContent = h.dst;
    li.addEventListener('click', () => {
      els.inputText.value = h.src;
      els.resultText.textContent = h.dst;
      els.resultSrc.textContent = h.src;
      els.resultSrc.hidden = false;
      els.resultWrap.hidden = false;
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    els.historyList.appendChild(li);
  }
}

function showResult(text, src) {
  els.resultText.textContent = text;
  // 원문도 결과 카드에 같이 보여준다 (친 문장·말한 문장과 번역문을 한눈에).
  els.resultSrc.textContent = src || '';
  els.resultSrc.hidden = !src;
  els.resultWrap.hidden = false;
}

async function doTranslate(srcText) {
  const src = (srcText ?? els.inputText.value).trim();
  if (!src) return null;
  let models;
  try {
    models = await ensureModels(direction);
  } catch (e) {
    // 모델이 없으면 번역이 안 된다. 조용히 넘어가지 말고 알려준다.
    showResult('모델을 먼저 받아야 합니다. 위의 "모델 받기" 버튼을 눌러 주세요.', '');
    return null;
  }
  setStatus('busy', '번역 중…');
  try {
    const dst = await translate(models.mt, src, direction);
    showResult(dst, src);
    saveHistory(src, dst);
    setStatus('ready', '오프라인 준비 완료');
    return dst;
  } catch (e) {
    console.error(e);
    setStatus('ready', '오프라인 준비 완료');
    els.resultText.textContent = '';
    alert(`번역 실패: ${e.message}`);
    return null;
  }
}

// ---- 음성 합성 (폰 자체 TTS) ----
let voices = [];
function refreshVoices() { voices = speechSynthesis.getVoices(); }
if ('speechSynthesis' in window) {
  refreshVoices();
  speechSynthesis.onvoiceschanged = refreshVoices;
}
function speak(text) {
  if (!('speechSynthesis' in window)) { alert('이 폰에서 읽어주기를 쓸 수 없습니다.'); return; }
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  const lang = TARGET_LANG[direction];
  u.lang = lang;
  const prefix = lang.slice(0, 2);
  const v = voices.find((x) => x.lang && x.lang.toLowerCase().startsWith(prefix));
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}

// ---- 마이크 / 대화 모드 ----
function setListeningUI(on, text) {
  els.listening.hidden = !on;
  if (text) els.listeningText.textContent = text;
  els.micBtn.classList.toggle('recording', on && captureMode === 'single');
  els.convBtn.classList.toggle('active', on && captureMode === 'conv');
}

function setBars(level) {
  const bars = document.querySelectorAll('#listening .bars i');
  bars.forEach((b, i) => {
    const h = 4 + Math.max(0, level - i * 0.2) * 16;
    b.style.height = `${Math.min(20, h)}px`;
  });
}

async function startCapture(mode) {
  const models = await ensureModels(direction);
  capture = new VoiceCapture({
    onLevel: (l) => setBars(l),
    onUtterance: async (audio) => {
      try {
        const dirAtSpeech = direction;
        const m = await ensureModels(dirAtSpeech);
        const lang = DIRECTIONS[dirAtSpeech].asrLang;
        const text = await transcribe(m.asr, audio, lang);
        if (!text) return;
        if (mode === 'single') {
          if (inputMode === 'dictate') {
            // 받아쓰기 모드: 인식된 글자를 이어붙이고 계속 듣는다. 번역은 안 한다.
            const prev = els.recognizedText.value.trim();
            els.recognizedText.value = prev ? `${prev} ${text}` : text;
            els.recognizedWrap.hidden = false;
          } else {
            stopCapture();
            els.recognizedText.value = text;
            els.recognizedWrap.hidden = false;
            // 인식된 글자를 보여주면서 바로 번역도 한다.
            // (틀렸으면 고치고 아래 "번역하기"를 누르면 다시 번역된다.)
            doTranslate(text);
          }
        } else {
          addConvTurn(text, dirAtSpeech);
        }
      } catch (e) {
        console.error(e);
        setListeningUI(true, `인식 실패: ${e.message}`);
      }
    },
  });
  try {
    await capture.start();
  } catch (e) {
    console.error(e);
    els.micHint.textContent = '마이크 권한을 허용해 주세요. (브라우저 주소창의 자물쇠 → 마이크 허용)';
    capture = null;
    return;
  }
  captureMode = mode;
  setListeningUI(true, mode === 'conv' ? '대화 모드 — 듣는 중…' : '듣는 중… 말을 마치면 자동으로 인식합니다');
}

function stopCapture() {
  capture?.stop();
  capture = null;
  captureMode = null;
  setListeningUI(false);
  setBars(0);
}

function addConvTurn(srcText, dirAtSpeech) {
  els.convWrap.hidden = false;
  const li = document.createElement('li');
  li.dataset.dir = dirAtSpeech;
  li.innerHTML = `<div class="src"></div><div class="dst">번역 중…</div><div class="meta"></div>`;
  li.querySelector('.src').textContent = srcText;
  const d = DIRECTIONS[dirAtSpeech];
  li.querySelector('.meta').textContent = `${d.from} → ${d.to} · 탭해서 고치기`;
  els.convList.prepend(li);
  (async () => {
    try {
      const m = await ensureModels(dirAtSpeech);
      const dst = await translate(m.mt, srcText, dirAtSpeech);
      li.querySelector('.dst').textContent = dst;
      saveHistory(srcText, dst, dirAtSpeech);
    } catch (e) {
      li.querySelector('.dst').textContent = `번역 실패: ${e.message}`;
    }
  })();
}

// 대화 말풍선 고치기: 원문을 탭하면 수정 가능, 벗어나면 다시 번역
els.convList.addEventListener('click', (e) => {
  const srcEl = e.target.closest('.src');
  if (!srcEl || srcEl.isContentEditable) return;
  srcEl.contentEditable = 'true';
  srcEl.focus();
});
els.convList.addEventListener('focusout', async (e) => {
  const srcEl = e.target.closest('.src');
  if (!srcEl || !srcEl.isContentEditable) return;
  srcEl.contentEditable = 'false';
  const li = srcEl.closest('li');
  const newSrc = srcEl.textContent.trim();
  if (!newSrc || newSrc === srcEl.dataset.orig) return;
  srcEl.dataset.orig = newSrc;
  const dirAtSpeech = li.dataset.dir;
  li.querySelector('.dst').textContent = '번역 중…';
  try {
    const m = await ensureModels(dirAtSpeech);
    const dst = await translate(m.mt, newSrc, dirAtSpeech);
    li.querySelector('.dst').textContent = dst;
    saveHistory(newSrc, dst, dirAtSpeech);
  } catch (err) {
    li.querySelector('.dst').textContent = `번역 실패: ${err.message}`;
  }
});
els.convList.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.isContentEditable) {
    e.preventDefault();
    e.target.blur();
  }
});

// ---- 버튼 연결 ----
els.modeTranslateBtn.addEventListener('click', () => setMode('translate'));
els.modeDictateBtn.addEventListener('click', () => setMode('dictate'));
els.downloadBtn.addEventListener('click', () => ensureModels(direction).catch(() => {}));
els.swapBtn.addEventListener('click', () => {
  direction = direction === 'ko-en' ? 'en-ko' : 'ko-en';
  updateModelBox();
});
els.translateBtn.addEventListener('click', () => doTranslate());
els.retranslateBtn.addEventListener('click', () => doTranslate(els.recognizedText.value));
els.micBtn.addEventListener('click', () => {
  if (capture) { stopCapture(); return; }
  startCapture('single');
});
els.convBtn.addEventListener('click', () => {
  if (convOn) {
    convOn = false;
    stopCapture();
    els.convBtn.classList.remove('active');
    els.convBtn.textContent = '대화 모드';
  } else {
    convOn = true;
    els.convBtn.textContent = '대화 모드 켜짐';
    startCapture('conv');
  }
});
els.stopListenBtn.addEventListener('click', () => {
  if (convOn) {
    convOn = false;
    els.convBtn.classList.remove('active');
    els.convBtn.textContent = '대화 모드';
  }
  stopCapture();
});
els.speakBtn.addEventListener('click', () => speak(els.resultText.textContent));
els.copyBtn.addEventListener('click', async () => {
  const text = els.resultText.textContent;
  const original = els.copyBtn.innerHTML;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }
  els.copyBtn.innerHTML = '복사됨';
  setTimeout(() => { els.copyBtn.innerHTML = original; }, 1500);
});
els.convClearBtn.addEventListener('click', () => {
  els.convList.innerHTML = '';
  els.convWrap.hidden = true;
});
els.historyClearBtn.addEventListener('click', () => {
  try { localStorage.removeItem('ot-history'); } catch (_) {}
  renderHistory();
});

// ---- 저장된 모델 자동 불러오기 ----
// transformers.js는 모델 파일을 브라우저 저장소(Cache Storage의
// 'transformers-cache')에 둔다. 앱을 새로 열 때마다 "모델 받기"를
// 다시 누르게 하지 않고, 저장된 모델이 있으면 알아서 불러온다.
// (저장된 게 없으면 조용히 넘어가고 버튼이 그대로 보인다.)
async function hasCachedModel(modelId) {
  try {
    const cache = await caches.open('transformers-cache');
    const keys = await cache.keys();
    return keys.some((r) => r.url.includes(modelId));
  } catch (_) {
    return false;
  }
}

async function autoLoadSavedModels() {
  const d = DIRECTIONS[direction];
  const asrCached = await hasCachedModel(d.asr);
  const mtCached = await hasCachedModel(d.mt);
  console.log('[auto-load]', d.asr, asrCached, d.mt, mtCached);
  if (!asrCached || !mtCached) {
    // 저장된 모델이 없으면 버튼 표시로 폴백. 진단용으로 박스에 표시한다.
    els.modelInfo.textContent += ' (폰에 저장된 모델이 없습니다)';
    return;
  }
  setStatus('busy', '저장된 모델 불러오는 중…');
  try {
    await ensureModels(direction);
  } catch (_) {
    // 저장소가 비었거나 깨졌으면 버튼 표시로 폴백 (ensureModels가 처리).
  }
}

window.addEventListener('pagehide', () => stopCapture());

renderHistory();
updateModelBox();
autoLoadSavedModels();
