// mt.js — 오프라인 번역 모델.
import { pipeline } from './vendor/transformers.web.min.js';

export const MT_MODEL_KO_EN = 'Xenova/opus-mt-ko-en';
export const MT_MODEL_EN_KO = 'Xenova/nllb-200-distilled-600M';

const cache = new Map();

export async function loadMT(modelId, onProgress) {
  if (cache.has(modelId)) return cache.get(modelId);
  const mt = await pipeline('translation', modelId, { dtype: 'q8', progress_callback: onProgress });
  cache.set(modelId, mt);
  return mt;
}

// dir: 'ko-en' | 'en-ko'
export async function translate(mt, text, dir) {
  if (dir === 'ko-en') {
    const out = await mt(text);
    return out[0].translation_text;
  }
  const out = await mt(text, { src_lang: 'eng_Latn', tgt_lang: 'kor_Hang' });
  return out[0].translation_text;
}
