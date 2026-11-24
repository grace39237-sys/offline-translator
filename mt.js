// mt.js — 오프라인 번역 모델.
//
// en-ko는 Opus-MT(teamwon/opus-mt-en-ko-onnx)를 쓴다. 이 모델은 sepvoc
// (영어 원문 사전 / 한국어 목표 사전이 따로)이라서 pipeline('translation')은
// 목표어를 영어 사전으로 디코딩해서 엉뚱한 말이 나온다. 그래서 모델·
// 토크나이저를 직접 불러 generate()로 만든 뒤, 목표어 토크나이저
// (tokenizer_target.json)로 디코딩한다. Node 테스트로 검증됨 (2026-11-24):
// 5문장 전부 정확한 한국어, 문장당 ~0.5초.
import {
  pipeline,
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  MarianTokenizer,
} from './vendor/transformers.web.min.js';

export const MT_MODEL_KO_EN = 'Xenova/opus-mt-ko-en';
export const MT_MODEL_EN_KO = 'teamwon/opus-mt-en-ko-onnx';

// generate() 결과의 첫 토큰(디코더 시작 토큰). 목표어 사전에 없어서
// 그대로 디코딩하면 "<unk>"로 나온다. 디코딩 전에 떼어낸다.
const DECODER_START_ID = 32000;

const cache = new Map();

// tokenizer_target.json은 transformers.js가 자동으로 안 받아서
// transformers-cache에 직접 넣어두고 쓴다. (처음 한 번만 인터넷 필요.
// 그 뒤로는 오프라인에서도 캐시에서 읽는다.)
async function loadTargetTokenizer(modelId, onProgress) {
  const targetUrl = `https://huggingface.co/${modelId}/resolve/main/tokenizer_target.json`;
  const configUrl = `https://huggingface.co/${modelId}/resolve/main/tokenizer_config.json`;
  const cacheStore = await caches.open('transformers-cache');

  let resp = await cacheStore.match(targetUrl);
  if (!resp) {
    resp = await fetch(targetUrl);
    if (!resp.ok) throw new Error(`tokenizer_target.json 받기 실패 (${resp.status})`);
    const total = Number(resp.headers.get('content-length') || 2422693);
    if (onProgress) onProgress({ status: 'progress', file: 'tokenizer_target.json', loaded: 0, total });
    await cacheStore.put(targetUrl, resp.clone());
    if (onProgress) onProgress({ status: 'progress', file: 'tokenizer_target.json', loaded: total, total });
  }
  const targetJSON = await resp.json();

  // tokenizer_config.json은 AutoTokenizer가 받을 때 transformers-cache에
  // 이미 넣어두므로, 보통은 캐시에서 바로 읽힌다.
  let configResp = await cacheStore.match(configUrl);
  if (!configResp) {
    configResp = await fetch(configUrl);
    if (!configResp.ok) throw new Error(`tokenizer_config.json 받기 실패 (${configResp.status})`);
    await cacheStore.put(configUrl, configResp.clone());
  }
  const targetConfig = await configResp.json();

  return new MarianTokenizer(targetJSON, targetConfig);
}

export async function loadMT(modelId, onProgress) {
  if (cache.has(modelId)) return cache.get(modelId);
  let mt;
  if (modelId === MT_MODEL_EN_KO) {
    const [model, tokenizer] = await Promise.all([
      AutoModelForSeq2SeqLM.from_pretrained(modelId, { dtype: 'q8', progress_callback: onProgress }),
      AutoTokenizer.from_pretrained(modelId, { progress_callback: onProgress }),
    ]);
    // 목표어 토크나이저는 tokenizer_config.json이 캐시에 들어간 뒤에
    // 불러야 해서 AutoTokenizer 다음에 순서대로 불러온다.
    const targetTokenizer = await loadTargetTokenizer(modelId, onProgress);
    mt = { model, tokenizer, targetTokenizer };
  } else {
    mt = await pipeline('translation', modelId, { dtype: 'q8', progress_callback: onProgress });
  }
  cache.set(modelId, mt);
  return mt;
}

// dir: 'ko-en' | 'en-ko'
export async function translate(mt, text, dir) {
  if (dir === 'ko-en') {
    const out = await mt(text);
    return out[0].translation_text;
  }
  // en-ko (Opus-MT, sepvoc): 수동 generate + 목표어 토크나이저 디코딩.
  const enc = mt.tokenizer(text);
  const out = await mt.model.generate({ ...enc, max_new_tokens: 128 });
  // tolist()는 BigInt를 돌려줘서 Number로 바꾼다.
  const ids = out.tolist()[0].map((x) => Number(x));
  const clean = ids[0] === DECODER_START_ID ? ids.slice(1) : ids;
  return mt.targetTokenizer.decode(clean, { skip_special_tokens: true }).trim();
}
