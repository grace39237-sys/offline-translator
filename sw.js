// sw.js — 앱 껍데기(HTML/CSS/JS/벤더 파일)를 캐시해서 오프라인에서 연다.
// 모델 파일은 Hugging Face에서 받으며 transformers.js가 브라우저 캐시에
// 따로 저장하므로 서비스 워커가 가로채지 않는다.
const CACHE = 'offline-translator-v19';
const SHELL = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './asr.js',
  './mt.js',
  './vad.js',
  './vad-processor.js',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './vendor/transformers.web.min.js',
  './vendor/ort/ort.bundle.min.mjs',
  './vendor/ort/ort-wasm-simd-threaded.wasm',
  './vendor/ort/ort-wasm-simd-threaded.mjs',
  './vendor/ort/ort-wasm-simd-threaded.jsep.wasm',
  './vendor/ort/ort-wasm-simd-threaded.jsep.mjs',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      // 이 앱의 셸 캐시(old 버전)만 지운다. transformers.js가 모델을
      // 저장하는 'transformers-cache'는 절대 지우지 않는다 — v6까지는
      // 이름이 다른 캐시를 전부 지워서, 저장된 모델이 SW 활성화 때마다
      // 날아가고 매번 다시 받아야 했다.
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith('offline-translator-') && k !== CACHE)
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // 같은 출처 파일만 캐시한다. Hugging Face 모델은 그대로 둔다.
  if (url.origin !== self.location.origin) return;
  e.respondWith(
    caches.match(e.request).then(
      (cached) =>
        cached ||
        fetch(e.request).then((resp) => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put(e.request, copy));
          }
          return resp;
        }),
    ),
  );
});
