// textutil.js — 음성인식 결과 정리 (순수 함수, 의존성 없음).
//
// Moonshine 같은 작은 ASR 모델은 소음이 섞이거나 침묵이 길면 같은 문장을
// 반복해서 내놓을 때가 있다. 인식 결과를 그대로 번역하면 그 반복이
// 번역에도 그대로 들어간다. 그래서 번역 전에 반복을 한 번으로 줄인다.
export function collapseRepeats(text) {
  if (!text) return text;
  // 1) 문장 단위: 연속으로 같은 문장이 반복되면 한 번만 남긴다.
  const parts = text.split(/(?<=[.!?。！？])\s+/);
  const deduped = [];
  for (const p of parts) {
    if (p && !(deduped.length && deduped[deduped.length - 1] === p)) deduped.push(p);
  }
  let out = deduped.join(' ');
  // 2) 문장 부호 없이 같은 구절이 통째로 반복되면 한 번만 남긴다.
  //    "안녕 안녕 안녕" / "hello hello hello"
  const words = out.split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    for (let n = 1; n <= Math.floor(words.length / 2); n++) {
      if (words.length % n !== 0) continue;
      const unit = words.slice(0, n).join(' ');
      if (words.join(' ') === Array(words.length / n).fill(unit).join(' ')) {
        out = unit;
        break;
      }
    }
  }
  return out;
}
