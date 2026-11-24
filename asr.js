// asr.js — Moonshine 음성인식 (transformers.js).
//
// NOTE: pipeline('automatic-speech-recognition')은 -ko 저장소에 쓸 수 없다.
// onnx-community/moonshine-*-ko-ONNX의 preprocessor_config.json에
// "processor_class"가 없어서 AutoProcessor가 토큰 없는 기본 Processor를
// 돌려주고("Unable to decode without a tokenizer") 인식이 실패한다.
// 그래서 모델·토크나이저·특징추출기를 직접 불러 MoonshineProcessor를
// 손으로 조립한다. transformers.js의 _call_moonshine과 같은 흐름이다.
import {
  AutoModelForSpeechSeq2Seq,
  AutoTokenizer,
  AutoFeatureExtractor,
  MoonshineProcessor,
} from './vendor/transformers.web.min.js';
import { collapseRepeats } from './textutil.js';

// 한국어 모델을 바꾸려면 이 상수 하나만 바꾸면 된다.
// base-ko (기본값, ~67MB, Node 테스트에서 한국어 완벽 인식)
// tiny-ko (~32MB)는 실폰 테스트에서 인식률이 너무 안 좋다고 보고됨 (2026-11-24).
export const ASR_MODEL_KO = 'onnx-community/moonshine-base-ko-ONNX';
export const ASR_MODEL_EN = 'onnx-community/moonshine-tiny-ONNX';

// 초당 최대 생성 토큰: 논문 휴리스틱 6/s는 한국어에 너무 적다.
// 5.5초 문장이 30토큰에 잘렸고, 12/s도 60토큰에 잘렸다.
// Node 테스트에서 16/s로 완전한 문장이 나왔다. 실폰에서는 반복 출력이
// 보고됨 (2026-11-24) — transcribe()에서 collapseRepeats로 한 번으로 줄인다.
const TOKENS_PER_SEC = 16;

const cache = new Map();

export async function loadASR(modelId, onProgress) {
  if (cache.has(modelId)) return cache.get(modelId);
  const [model, tokenizer, featureExtractor] = await Promise.all([
    AutoModelForSpeechSeq2Seq.from_pretrained(modelId, { dtype: 'q8', progress_callback: onProgress }),
    AutoTokenizer.from_pretrained(modelId, { progress_callback: onProgress }),
    AutoFeatureExtractor.from_pretrained(modelId, { progress_callback: onProgress }),
  ]);
  const processor = new MoonshineProcessor({}, { tokenizer, feature_extractor: featureExtractor });
  const asr = { model, processor };
  cache.set(modelId, asr);
  return asr;
}

// audio16k: Float32Array, 16kHz 모노. language: 'korean' | 'english'.
export async function transcribe(asr, audio16k, language) {
  const inputs = await asr.processor(audio16k);
  const max_new_tokens = Math.max(32, Math.floor((audio16k.length / 16000) * TOKENS_PER_SEC));
  const outputs = await asr.model.generate({
    max_new_tokens,
    language,
    task: 'transcribe',
    ...inputs,
  });
  return collapseRepeats(asr.processor.batch_decode(outputs, { skip_special_tokens: true })[0].trim());
}
