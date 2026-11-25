// app.js — 오프라인 번역기 UI + 흐름.
import * as ort from 'onnxruntime-web';
import { loadASR, transcribe, ASR_MODEL_KO, ASR_MODEL_EN } from './asr.js';
import { loadMT, translate, MT_MODEL_KO_EN, MT_MODEL_EN_KO } from './mt.js';
import { createPool } from './pool.js';
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
const APP_VERSION = 'v22';
const TARGET_LANG = { 'ko-en': 'en-US', 'en-ko': 'ko-KR' };

// 듣기 감도: 마이크가 말을 얼마나 쉽게 듣는지. 화면 표시는 소음 정도다:
// 낮게(조용한 곳)=민감(작은 소리도 듣기, 임계값 = 소음 기준 ×2),
// 보통=×4, 높게(시끄러운 곳)=둔감(시끄러운 곳에서 잡음 무시, ×7).
// 기본 보통. 선택은 폰에 저장한다.
const SENSITIVITY = { high: 2, mid: 4, low: 7 };
let sensitivity = 'mid';
try { sensitivity = localStorage.getItem('ot-sensitivity') || 'mid'; } catch (_) {}
if (!SENSITIVITY[sensitivity]) sensitivity = 'mid';

// 듣기 방식: 'auto' (말 끝나면 0.9초 침묵 뒤 자동 인식) / 'ptt' (버튼을
// 누른 채로만 녹음, 떼면 바로 인식). 선택은 폰에 저장한다.
let listenMode = 'auto';
try { listenMode = localStorage.getItem('ot-listen-mode') || 'auto'; } catch (_) {}
if (listenMode !== 'ptt') listenMode = 'auto';

const $ = (id) => document.getElementById(id);
const els = {
  statusDot: $('statusDot'), statusText: $('statusText'),
  langFrom: $('langFrom'), langTo: $('langTo'), swapBtn: $('swapBtn'),
  modelBox: $('modelBox'), modelInfo: $('modelInfo'),
  downloadBtn: $('downloadBtn'), downloadProgress: $('downloadProgress'),
  downloadBar: $('downloadBar'), downloadLabel: $('downloadLabel'),
  inputText: $('inputText'), micBtn: $('micBtn'), convBtn: $('convBtn'),
  convMicRow: $('convMicRow'), convMicKoBtn: $('convMicKoBtn'), convMicEnBtn: $('convMicEnBtn'),
  enhancedChk: $('enhancedChk'),
  inputModeBtn: $('inputModeBtn'), listenModeBtn: $('listenModeBtn'),
  recognizedLabel: $('recognizedLabel'),
  translateBtn: $('translateBtn'), listening: $('listening'),
  listeningText: $('listeningText'), sendBtn: $('sendBtn'),
  micHint: $('micHint'),
  sensHighBtn: $('sensHighBtn'), sensMidBtn: $('sensMidBtn'), sensLowBtn: $('sensLowBtn'),
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

// 향상모드: 대화 모드에서 인식기를 2개로 복제해서 번갈아 쓴다.
// (1번 차례는 1번 세션, 2번 차례는 2번 세션, 3번 차례는 다시 1번…)
// 켜져 있으면 연속으로 말해도 차례가 버려지지 않고, 꺼져 있으면
// 한 번에 하나씩 순서대로 처리한다. 선택은 폰에 저장한다 (기본 켜짐).
let enhanced = true;
try { enhanced = localStorage.getItem('ot-enhanced') !== '0'; } catch (_) {}
let convDir = null; // 대화 모드에서 지금 듣는 방향 (null = 안 듣는 중)
const convPools = {}; // `${dir}|${workers}` -> pool
const convAsr2 = {}; // dir -> 두 번째 인식 세션 (향상모드용)
// 번역은 방향마다 한 번에 하나씩 (같은 번역 세션에 두 번 동시 generate()를
// 시키면 인식 세션처럼 서로 방해한다).
const mtChains = { 'ko-en': Promise.resolve(), 'en-ko': Promise.resolve() };
function translateSerialized(mt, text, dir) {
  const p = mtChains[dir].then(() => translate(mt, text, dir));
  mtChains[dir] = p.catch(() => {});
  return p;
}

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
    convDir = null;
    updateConvUI();
  }
  inputMode = next;
  // 각 버튼에 지금 모드만 보이고, 누르면 다른 모드로 바뀐다.
  // (큰 글씨만 바꾸고, 밑의 작은 글씨는 그대로 둔다.)
  els.inputModeBtn.querySelector('.mode-main').textContent = next === 'translate' ? '번역' : '받아쓰기';
  // 대화 모드는 번역 대화 기능이라 받아쓰기 모드에서는 안 보인다.
  // 누를 때만 모드에서도 안 보인다 (떼면 바로 번역되니까).
  els.convBtn.hidden = next === 'dictate' || listenMode === 'ptt';
  els.recognizedLabel.textContent = next === 'dictate'
    ? '받아쓴 글자 — 말할 때마다 이어붙습니다. 틀린 부분은 고치세요'
    : '인식된 글자 — 번역이 바로 나옵니다. 틀린 부분은 고치고 "번역하기"를 누르세요';
  // 힌트는 듣기 방식이 정한다 (누를 때만 모드에서는 침묵·보내기 얘기가 안 맞는다).
  els.micHint.textContent = listenMode === 'ptt' ? LISTEN_HINT.ptt
    : next === 'dictate'
      ? '마이크: 말하고 보내기를 누르면 아래에 이어붙습니다. 0.9초 침묵 뒤에도 자동으로 인식돼요. (최대 30초)'
      : '마이크: 말하고 보내기를 누르면 바로 인식합니다. 0.9초 침묵 뒤에도 자동으로 인식돼요. (최대 30초)';
}

async function ensureModels(dir) {
  if (loaded[dir]) return loaded[dir];
  const d = DIRECTIONS[dir];
  progressFiles.clear();
  progressExpected = d.sizeBytes;
  els.downloadBtn.disabled = true;
  // 저장된 모델이 있으면 폰 안에서 불러오는 거라 받는 중이 아니다.
  // "모델 받는 중…"으로 뜨면 새로 받는 것처럼 보이니까, 이 경우엔
  // 불러오는 중이라고 표시하고 다운로드 진행 막대는 안 보여준다.
  const fromCache = (await hasCachedModel(d.asr)) && (await hasCachedModel(d.mt));
  if (fromCache) {
    els.downloadProgress.hidden = true;
    els.downloadLabel.hidden = true;
    setStatus('busy', '저장된 모델 불러오는 중…');
  } else {
    els.downloadProgress.hidden = false;
    els.downloadLabel.hidden = false;
    els.downloadBar.style.width = '0%';
    els.downloadLabel.textContent = '받는 중…';
    setStatus('busy', '모델 받는 중…');
  }
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
    els.downloadLabel.hidden = false;
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
    const dst = await translateSerialized(models.mt, src, direction);
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
// 한 문장씩 듣기: 인식·번역은 한 번에 하나만 돈다. 같은 모델을 두 번 동시에
// 돌리면 둘 다 죽는다 (2026-11-24 실폰 보고: 7초 걸리는 인식 중에 다시
// 말하면 앞 작업이 중단됨). 처리 중 새 발화가 들어오면 1개만 대기열에 둔다
// (더 새 발화가 오면 갈아끼운다). 마이크를 눌러 새 작업을 시작하면 runId가
// 올라가서 진행 중이던 옛날 작업의 결과는 버려진다.
// 대화 모드: 풀(submitConvTurn)로 돈다. 차례마다 인식 세션을 번갈아 배정하고
// (향상모드 켜짐: 2개, 꺼짐: 1개), 작업자가 바쁘면 기다리기만 하고 버리지 않는다.
let processing = false;
let queued = null;
let runId = 0;

// 한 문장씩 듣기 전용. 대화 모드는 풀(submitConvTurn)로 따로 돈다.
async function runTask(audio) {
  const myRun = runId;
  try {
    // 발화 받자마자 마이크를 끈다. 소리는 이미 받아뒀으니 인식에는 지장 없고,
    // 인식 중에 새 말이 들어와 앞 작업을 죽이는 일을 막는다.
    stopCapture();
    setListeningUI(true, '인식 중…');
    const dirAtSpeech = direction;
    const m = await ensureModels(dirAtSpeech);
    const lang = DIRECTIONS[dirAtSpeech].asrLang;
    const text = await transcribe(m.asr, audio, lang);
    // 그 사이 마이크를 눌러 새 작업이 시작됐으면 이 결과는 버린다.
    if (myRun !== runId) return;
    // 인식 결과가 비어 나왔으면 조용히 버리지 않고 알려준다.
    // (안 그러면 "인식 중…"만 뜨고 아무것도 안 뜨는 것처럼 보인다.)
    if (!text) {
      els.micHint.textContent = '잘 안 들렸어요. 다시 말해주세요. (마이크 가까이 말해보세요)';
      return;
    }
    // 인식 결과가 그 방향의 말이 아니면(예: 한국어 모드인데 한글이 하나도
    // 없으면) 잘못 들은 거라 번역하면 헛소리가 나온다. 번역하지 않고
    // 인식된 글자만 보여주고 다시 말하게 한다.
    const looksRight = dirAtSpeech === 'ko-en' ? /[가-힣]/.test(text) : /[a-zA-Z]/.test(text);
    if (!looksRight) {
      if (inputMode !== 'dictate') {
        els.recognizedText.value = text;
        els.recognizedWrap.hidden = false;
      }
      els.micHint.textContent = '잘 안 들렸어요. 다시 말해주세요. (주변 소음이 있으면 조용한 곳에서 해보세요)';
      return;
    }
    if (inputMode === 'dictate') {
      // 받아쓰기 모드: 인식된 글자를 이어붙이고 계속 듣는다. 번역은 안 한다.
      const prev = els.recognizedText.value.trim();
      els.recognizedText.value = prev ? `${prev} ${text}` : text;
      els.recognizedWrap.hidden = false;
    } else {
      els.recognizedText.value = text;
      els.recognizedWrap.hidden = false;
      // 인식된 글자를 보여주면서 바로 번역도 한다.
      // (틀렸으면 고치고 아래 "번역하기"를 누르면 다시 번역된다.)
      await doTranslate(text);
    }
  } catch (e) {
    console.error(e);
    if (myRun === runId) setListeningUI(true, `인식 실패: ${e.message}`);
  } finally {
    if (myRun === runId) setListeningUI(false);
  }
}

function setListeningUI(on, text) {
  els.listening.hidden = !on;
  if (text) els.listeningText.textContent = text;
  els.micBtn.classList.toggle('recording', on && captureMode === 'single');
  // 대화 모드에서는 지금 듣는 언어 버튼이 빨갛게 보인다.
  els.convMicKoBtn.classList.toggle('recording', on && captureMode === 'conv' && convDir === 'ko-en');
  els.convMicEnBtn.classList.toggle('recording', on && captureMode === 'conv' && convDir === 'en-ko');
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
  const cap = new VoiceCapture({
    onLevel: (l) => setBars(l),
    sensitivity: SENSITIVITY[sensitivity],
    // 누를 때만 모드에서는 버튼을 누른 채로만 녹음한다 (대화 모드는 자동 그대로).
    ptt: mode === 'single' && listenMode === 'ptt',
    // 말이 들리면 화면에 바로 표시한다 — 들었는지 안 들었는지 보이게.
    onSpeechStart: () => setListeningUI(true, '듣고 있어요…'),
    onSpeechEnd: () => setListeningUI(true, mode === 'conv' ? '대화 모드 — 듣는 중… 보내기를 누르면 그 차례를 바로 번역합니다' : '듣는 중… 보내기를 누르면 바로 인식합니다'),
    onUtterance: async (audio) => {
      // 대화 모드: 풀에 넘긴다. 차례마다 작업자를 번갈아 배정하고,
      // 작업자가 바쁘면 기다리기만 하고 버려지지 않는다.
      if (mode === 'conv') { submitConvTurn(audio, direction); return; }
      // 한 문장씩: 처리 중이면 1개만 대기열에 둔다 (더 새 발화가 오면 갈아끼운다).
      if (processing) { queued = { audio }; return; }
      processing = true;
      let current = { audio };
      while (current) {
        await runTask(current.audio);
        current = queued;
        queued = null;
      }
      processing = false;
    },
  });
  capture = cap;
  try {
    await cap.start();
  } catch (e) {
    console.error(e);
    els.micHint.textContent = '마이크 권한을 허용해 주세요. (브라우저 주소창의 자물쇠 → 마이크 허용)';
    // 그 사이에 새 마이크가 켜졌으면 그건 건드리지 않는다.
    if (capture === cap) capture = null;
    return;
  }
  // 시작 도중에 멈췄으면(짧게 누르고 뗐으면) 이 마이크는 이미 정리됐다.
  // 새로 켠 것처럼 굴지 않는다.
  if (capture !== cap) return;
  captureMode = mode;
  // 마이크를 새로 켜면 진행 중이던 옛날 작업의 결과는 버린다 (새 작업이 우선).
  runId++;
  setListeningUI(true, mode === 'conv' ? '대화 모드 — 듣는 중… 보내기를 누르면 그 차례를 바로 번역합니다' : '듣는 중… 보내기를 누르면 바로 인식합니다');
}

function stopCapture() {
  capture?.stop();
  capture = null;
  captureMode = null;
  setListeningUI(false);
  setBars(0);
}

// ---- 대화 모드: 차례 카드 + 향상모드 풀 ----
// 대화 차례 카드는 발화가 들어온 순서대로 만든다 (인식 전, "인식 중…" 표시).
// 인식·번역이 끝나면 그 카드에 채운다. 나중에 온 차례가 먼저 끝나도
// 카드 순서는 말한 순서 그대로다 (목록은 새 차례가 위로 간다).
function createConvTurnCard(dirAtSpeech) {
  els.convWrap.hidden = false;
  const li = document.createElement('li');
  li.dataset.dir = dirAtSpeech;
  li.innerHTML = `<div class="src">인식 중…</div><div class="dst">번역 중…</div><div class="meta"></div>`;
  const d = DIRECTIONS[dirAtSpeech];
  li.querySelector('.meta').textContent = `${d.from} → ${d.to} · 탭해서 고치기`;
  els.convList.prepend(li);
  return li;
}

// 향상모드용 두 번째 인식 세션 (방향마다 하나). 다운로드는 한 번만 받고,
// 세션만 두 개로 나눠 쓴다. 꺼져 있으면 0번 세션만 쓴다.
async function convASR(dir, workerIdx) {
  const m = await ensureModels(dir);
  if (workerIdx === 0) return m.asr;
  if (!convAsr2[dir]) convAsr2[dir] = await loadASR(DIRECTIONS[dir].asr, onProgress, 1);
  return convAsr2[dir];
}

function convPoolFor(dir) {
  const workers = enhanced ? 2 : 1;
  const key = `${dir}|${workers}`;
  if (!convPools[key]) convPools[key] = createPool(workers);
  return convPools[key];
}

// 한 차례 처리: 인식(풀의 작업자 세션) → 번역(방향마다 한 번에 하나씩).
// 결과는 이 차례의 카드에 채운다. 실패해도 카드에 이유를 적고 끝낸다.
async function processConvTurn(audio, dir, card, workerIdx) {
  const srcEl = card.querySelector('.src');
  const dstEl = card.querySelector('.dst');
  try {
    const asr = await convASR(dir, workerIdx);
    const text = await transcribe(asr, audio, DIRECTIONS[dir].asrLang);
    if (!text) {
      srcEl.textContent = '잘 안 들렸어요. 다시 말해주세요.';
      dstEl.textContent = '';
      return;
    }
    const looksRight = dir === 'ko-en' ? /[가-힣]/.test(text) : /[a-zA-Z]/.test(text);
    if (!looksRight) {
      srcEl.textContent = text;
      dstEl.textContent = '잘 안 들렸어요. 다시 말해주세요. (주변 소음이 있으면 조용한 곳에서 해보세요)';
      return;
    }
    srcEl.textContent = text;
    const m = await ensureModels(dir);
    const dst = await translateSerialized(m.mt, text, dir);
    dstEl.textContent = dst;
    saveHistory(text, dst, dir);
  } catch (e) {
    console.error(e);
    dstEl.textContent = `인식 실패: ${e.message}`;
  }
}

function submitConvTurn(audio, dir) {
  const card = createConvTurnCard(dir);
  convPoolFor(dir).submit((workerIdx) => processConvTurn(audio, dir, card, workerIdx));
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
    const dst = await translateSerialized(m.mt, newSrc, dirAtSpeech);
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

// ---- 듣기 감도 ----
function updateSensitivityUI() {
  els.sensHighBtn.setAttribute('aria-pressed', String(sensitivity === 'high'));
  els.sensMidBtn.setAttribute('aria-pressed', String(sensitivity === 'mid'));
  els.sensLowBtn.setAttribute('aria-pressed', String(sensitivity === 'low'));
}
function setSensitivity(next) {
  if (!SENSITIVITY[next]) return;
  sensitivity = next;
  try { localStorage.setItem('ot-sensitivity', next); } catch (_) {}
  updateSensitivityUI();
  // 듣는 중이면 임계값과 EQ 강도를 바로 함께 바꾼다.
  if (capture) capture.setSensitivity(SENSITIVITY[next]);
}

// ---- 듣기 방식 (자동 / 누를 때만) ----
const LISTEN_HINT = {
  auto: '마이크: 말하고 보내기를 누르면 바로 인식합니다. 0.9초 침묵 뒤에도 자동으로 인식돼요. (최대 30초)',
  ptt: '마이크: 버튼을 누른 채로 말하고, 떼면 바로 인식합니다. (최대 30초)',
};
function updateListenModeUI() {
  // 각 버튼에 지금 모드만 보이고, 누르면 다른 모드로 바뀐다.
  // (큰 글씨만 바꾸고, 밑의 작은 글씨는 그대로 둔다.)
  els.listenModeBtn.querySelector('.mode-main').textContent = listenMode === 'auto' ? '자동' : '누를 때만';
  els.micHint.textContent = LISTEN_HINT[listenMode];
  // 누를 때만 모드에서는 대화 모드·번역하기 버튼이 필요 없다.
  // 떼면 바로 번역되니까, 마이크와 감도만 남긴다.
  els.convBtn.hidden = listenMode === 'ptt' || inputMode === 'dictate';
  els.translateBtn.hidden = listenMode === 'ptt';
  // 누를 때만 모드에서는 친 텍스트 입력 칸도 안 보인다 (말로만 쓴다).
  els.inputText.hidden = listenMode === 'ptt';
  // 누를 때만 모드에서는 마이크 버튼을 더 크게, 누르기 편하게.
  els.micBtn.classList.toggle('ptt-big', listenMode === 'ptt');
}
function setListenMode(next) {
  if (next !== 'auto' && next !== 'ptt') return;
  if (listenMode === next) return;
  listenMode = next;
  try { localStorage.setItem('ot-listen-mode', next); } catch (_) {}
  // 듣는 중에 방식을 바꾸면 헷갈리니까 녹음을 끝낸다.
  if (capture) stopCapture();
  updateListenModeUI();
}
els.sensHighBtn.addEventListener('click', () => setSensitivity('high'));
els.sensMidBtn.addEventListener('click', () => setSensitivity('mid'));
els.sensLowBtn.addEventListener('click', () => setSensitivity('low'));

// 모드 버튼은 2개만 있다 — 각 버튼에 지금 모드만 보이고, 누르면 바뀐다.
els.inputModeBtn.addEventListener('click', () => setMode(inputMode === 'translate' ? 'dictate' : 'translate'));
els.listenModeBtn.addEventListener('click', () => setListenMode(listenMode === 'auto' ? 'ptt' : 'auto'));
els.downloadBtn.addEventListener('click', () => ensureModels(direction).catch(() => {}));
els.swapBtn.addEventListener('click', () => {
  direction = direction === 'ko-en' ? 'en-ko' : 'ko-en';
  updateModelBox();
});
els.translateBtn.addEventListener('click', () => doTranslate());
els.retranslateBtn.addEventListener('click', () => doTranslate(els.recognizedText.value));
els.micBtn.addEventListener('click', () => {
  // 누를 때만 모드에서는 클릭이 아니라 누르고 있기(pointer)로 동작한다.
  // (pointerup 뒤에 따라오는 click이 마이크를 다시 켜는 것을 막는다.)
  if (listenMode === 'ptt') return;
  if (capture) { stopCapture(); return; }
  startCapture('single');
});
// 누를 때만 모드: 누르고 있는 동안만 녹음, 떼면 바로 인식한다.
els.micBtn.addEventListener('pointerdown', (e) => {
  if (listenMode !== 'ptt') return;
  e.preventDefault();
  if (capture) return;
  // 손가락을 버튼에 묶어둔다. 누른 채로 손가락이 버튼에서 벗어나도
  // 떼기 전까지 녹음이 계속된다 (누르기 편하게).
  try { els.micBtn.setPointerCapture(e.pointerId); } catch (_) {}
  els.micBtn.classList.add('holding');
  startCapture('single');
});
function endPtt() {
  els.micBtn.classList.remove('holding');
  if (!capture) return;
  const wasProcessing = processing;
  const had = capture.flush();
  // 떼면 녹음은 끝. 인식이 대기 중이면 마이크는 지금 끈다.
  if (had && wasProcessing) stopCapture();
  if (!had) {
    stopCapture();
    els.micHint.textContent = '너무 짧아요. 버튼을 누른 채로 말해주세요.';
  }
}
els.micBtn.addEventListener('pointerup', () => { if (listenMode === 'ptt') endPtt(); });
els.micBtn.addEventListener('pointercancel', () => { if (listenMode === 'ptt') endPtt(); });
els.micBtn.addEventListener('pointerleave', () => { if (listenMode === 'ptt') endPtt(); });
els.micBtn.addEventListener('contextmenu', (e) => e.preventDefault());
// ---- 대화 모드 UI (마이크 버튼 2개 + 향상모드) ----
function updateConvUI() {
  els.convBtn.textContent = convOn ? '대화 모드 켜짐' : '대화 모드';
  els.convBtn.classList.toggle('active', convOn);
  // 대화 모드에서는 마이크 버튼 2개(한국어/English)가 마이크다.
  // 한 문장씩 하는 마이크·방향 바꾸기·모드 버튼은 안 보인다.
  els.micBtn.hidden = convOn;
  els.convMicRow.hidden = !convOn;
  els.swapBtn.hidden = convOn;
  els.inputModeBtn.hidden = convOn;
  els.listenModeBtn.hidden = convOn;
}

// 대화 모드에 들어갈 때 양쪽 방향 모델이 있는지 확인하고,
// 없는 방향이 있으면 받으라고 안내한다.
async function checkConvModels() {
  const missing = [];
  for (const dir of ['ko-en', 'en-ko']) {
    const d = DIRECTIONS[dir];
    const ok = (await hasCachedModel(d.asr)) && (await hasCachedModel(d.mt));
    if (!ok) missing.push(d);
  }
  if (missing.length) {
    const names = missing.map((d) => `${d.from} → ${d.to}`).join(', ');
    els.micHint.textContent = `대화 모드: ${names} 모델이 없습니다. 해당 언어 마이크 버튼을 누르면 받아요. (처음 한 번만 인터넷 필요)`;
  } else {
    els.micHint.textContent = '대화 모드: 한국어 버튼은 한국어로 듣고 영어로, English 버튼은 영어로 듣고 한국어로 번역해요.';
  }
}

// 대화 모드 언어 마이크: 같은 버튼을 다시 누르면 듣기를 멈추고,
// 다른 버튼을 누르면 그 언어로 바꿔서 계속 듣는다.
async function toggleConvMic(dir) {
  if (capture && captureMode === 'conv' && convDir === dir) {
    stopCapture();
    convDir = null;
    setListeningUI(false);
    return;
  }
  if (capture) stopCapture();
  direction = dir;
  convDir = dir;
  updateModelBox(); // 언어 바·모델 박스를 이 방향으로 맞춘다.
  // 모델이 없으면 받으라고 안내하고, 받으면 듣기를 시작한다.
  if (!loaded[dir]) {
    els.micHint.textContent = `${DIRECTIONS[dir].from} → ${DIRECTIONS[dir].to} 모델을 받는 중이에요. 받으면 바로 듣기 시작해요.`;
    try { await ensureModels(dir); } catch (_) { return; }
  }
  startCapture('conv');
}
els.convMicKoBtn.addEventListener('click', () => toggleConvMic('ko-en'));
els.convMicEnBtn.addEventListener('click', () => toggleConvMic('en-ko'));
els.enhancedChk.addEventListener('change', () => {
  enhanced = els.enhancedChk.checked;
  try { localStorage.setItem('ot-enhanced', enhanced ? '1' : '0'); } catch (_) {}
  // 다음 차례부터 새 풀이 쓰인다. 진행 중인 차례는 옛날 풀에서 끝까지 간다.
  for (const k of Object.keys(convPools)) delete convPools[k];
});
els.convBtn.addEventListener('click', async () => {
  if (convOn) {
    convOn = false;
    stopCapture();
    convDir = null;
    updateConvUI();
    setListeningUI(false);
    els.micHint.textContent = LISTEN_HINT[listenMode];
  } else {
    convOn = true;
    updateConvUI();
    await checkConvModels();
    // 듣기는 언어 버튼을 눌러서 시작한다 (방향을 골라야 하니까).
  }
});
els.sendBtn.addEventListener('click', () => {
  // 인식·번역 중에는 보내기를 눌러도 새 작업이 안 돈다. 기다리라고 알린다.
  if (processing) {
    setListeningUI(true, '인식 중이에요. 잠시만요.');
    return;
  }
  // 보내기: 지금까지 들은 말을 침묵을 기다리지 않고 바로 인식·번역한다.
  // 대화 모드에서는 그 차례만 끝내고 계속 듣는다.
  if (!capture?.flush()) {
    setListeningUI(true, '아직 들은 말이 없어요. 말하고 보내기를 누르세요');
  }
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
  try {
    await ensureModels(direction);
  } catch (_) {
    // 저장소가 비었거나 깨졌으면 버튼 표시로 폴백 (ensureModels가 처리).
  }
}

window.addEventListener('pagehide', () => stopCapture());

renderHistory();
updateModelBox();
updateSensitivityUI();
updateListenModeUI();
els.enhancedChk.checked = enhanced;
updateConvUI();
autoLoadSavedModels();
