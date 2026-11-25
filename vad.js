// vad.js — 에너지 기반 음성 구간 감지(VAD) + 16kHz PCM 캡처.
//
// 흐름: 마이크 → AudioWorklet(20ms 단위) → 16kHz로 리샘플 → 100ms 프레임
// 에너지 계산 → 방 소음에 맞춘 적응 임계값 → 말 시작/끝 판단.
// 말 시작이 잘리지 않게 0.5초 프리롤 버퍼를 항상 들고 있다.
// 말이 끝나면(약 0.9초 침묵) 발화 구간을 onUtterance로 넘긴다.
// 동시통역이 아니라 문장 단위 처리다.
const FRAME = 1600; // 100ms @ 16kHz
const SILENCE_FRAMES_TO_END = 9; // 0.9초 침묵이면 발화 끝
const PRE_ROLL = 8000; // 0.5초 프리롤
const MAX_UTTERANCE_SEC = 30; // 한 발화 최대 30초
// 소음 기준 측정: 조용한 프레임만 모은다. 말처럼 큰 소리는 제외한다.
// (처음 1초에 말이 섞이면 기준이 말 소리만큼 부풀어서 그 뒤 말이
// 임계값을 못 넘고 앱이 귀머거리가 된다 — 2026-11-24 실폰 보고.)
const QUIET_ENERGY = 0.0002; // 이보다 큰 프레임은 말이 섞인 것으로 보고 제외
const MIN_THRESHOLD = 0.0001; // 임계값 하한. 조용한 곳에서 소음 기준이
// 낮아져도 이 아래로는 안 내려간다 — 이보다 작은 목소리는 인식이 못 듣는다
// (2026-11-25 실폰 보고: 조용한 곳에서도 인식이 비어 나옴).
const NOISE_FRAMES_NEEDED = 10; // 조용한 프레임 10개(1초)로 기준을 잡는다
const NOISE_WAIT_MAX_FRAMES = 30; // 3초 안에 안 모이면 모은 것으로 기준을 잡는다

// EQ 강도: 감도에 맞춰 함께 바꾼다. 단, 두 손잡이는 반대로 움직인다.
// - 저음 자르기(highpass): 시끄러울수록 세게 — 저음 웅웅거림(팬·차·바람)을
//   더 많이 잘라낸다. 낮게 80Hz / 보통 120Hz / 높게 200Hz.
// - 말소리 대역 올리기(2.5kHz presence): 조용할수록 세게 — 작은 목소리를
//   선명하게 한다. 시끄러운 곳에서 올리면 잡음까지 커져서 안 된다.
//   낮게 +6dB / 보통 +3dB / 높게 +0dB.
export function eqProfileForSensitivity(sensitivity) {
  if (sensitivity <= 2) return { hp: 80, peakGain: 6 };
  if (sensitivity >= 7) return { hp: 200, peakGain: 0 };
  return { hp: 120, peakGain: 3 };
}

export class VoiceCapture {
  constructor({ onUtterance, onLevel, onSpeechStart, onSpeechEnd, sensitivity, ptt }) {
    this.onUtterance = onUtterance;
    this.onLevel = onLevel || (() => {});
    this.onSpeechStart = onSpeechStart || (() => {});
    this.onSpeechEnd = onSpeechEnd || (() => {});
    // 감도: 임계값 = 소음 기준 × 감도. 작을수록 작은 소리도 듣는다.
    // 낮게(민감) 2 / 보통 4 / 높게(둔감) 7.
    this.sensitivity = sensitivity || 6;
    // ptt (누를 때만): 버튼을 누른 채로만 녹음한다. 소음 기준을 재지 않고
    // 침묵으로 자동 종료하지도 않는다 — 떼면 flush()로 바로 넘긴다.
    this.ptt = !!ptt;
    this.running = false;
  }

  async start() {
    if (this.running) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    this.stream = stream;
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule('./vad-processor.js');
    const src = this.ctx.createMediaStreamSource(stream);
    // EQ: 인식기에 들어가기 전에 보이스를 강화한다. 저음 웅웅거림을 자르고
    // 말소리가 제일 선명한 대역(2.5kHz)을 올린다. 강도는 감도(상/중/하)와 함께 간다.
    const eq = eqProfileForSensitivity(this.sensitivity);
    this.hpFilter = this.ctx.createBiquadFilter();
    this.hpFilter.type = 'highpass';
    this.hpFilter.frequency.value = eq.hp;
    this.peakFilter = this.ctx.createBiquadFilter();
    this.peakFilter.type = 'peaking';
    this.peakFilter.frequency.value = 2500;
    this.peakFilter.Q.value = 1;
    this.peakFilter.gain.value = eq.peakGain;
    this.node = new AudioWorkletNode(this.ctx, 'vad-processor');
    src.connect(this.hpFilter);
    this.hpFilter.connect(this.peakFilter);
    this.peakFilter.connect(this.node);
    this.node.port.onmessage = (e) => this._onChunk(e.data.data);
    this.sampleRate = this.ctx.sampleRate;
    this.resampleStep = this.sampleRate / 16000;
    this.preRoll = [];
    this.frames = [];
    this.noiseFloor = null;
    // 누를 때만 모드: 소음 기준을 재지 않는다 (임계값을 안 쓰니까).
    // 첫 프레임부터 바로 말을 받는다.
    if (this.ptt) this.noiseFloor = QUIET_ENERGY;
    this.noiseSum = 0;
    this.noiseFrames = 0;
    this.noiseMin = Infinity;
    this.noiseWait = 0;
    this.speaking = false;
    this.silenceFrames = 0;
    this.speechFrames = 0;
    this.speechStart = 0;
    this.speechEnd = 0;
    this.utterance = [];
    this.running = true;
  }

  // 감도를 바꾸면 임계값 배수와 EQ 강도를 함께 바꾼다 (듣는 중에도 바로 적용).
  setSensitivity(mult) {
    this.sensitivity = mult;
    const eq = eqProfileForSensitivity(mult);
    if (this.hpFilter) this.hpFilter.frequency.value = eq.hp;
    if (this.peakFilter) this.peakFilter.gain.value = eq.peakGain;
  }

  stop() {
    this.running = false;
    try { this.node?.disconnect(); } catch (_) {}
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch (_) {}
    if (this.ctx) this.ctx.close().catch(() => {});
    this.node = null;
    this.hpFilter = null;
    this.peakFilter = null;
    this.ctx = null;
  }

  _onChunk(chunk) {
    if (!this.running) return;
    // 48kHz(보통) → 16kHz: 박스 평균으로 다운샘플
    const step = this.resampleStep;
    let acc = 0;
    let n = 0;
    let next = step;
    for (let i = 0; i < chunk.length; i++) {
      acc += chunk[i];
      n++;
      if (i + 1 >= next) {
        this._onSample(acc / n);
        acc = 0;
        n = 0;
        next += step;
      }
    }
  }

  _onSample(v) {
    this.preRoll.push(v);
    if (this.preRoll.length > PRE_ROLL) this.preRoll.shift();
    this.frames.push(v);
    if (this.frames.length < FRAME) return;
    const frame = this.frames.splice(0, FRAME);
    let energy = 0;
    for (let i = 0; i < frame.length; i++) energy += frame[i] * frame[i];
    energy /= frame.length;
    this.onLevel(Math.min(1, Math.sqrt(Math.max(energy, 0)) / 0.08));

    if (this.noiseFloor === null) {
      // 소음 기준 측정: 조용한 프레임만 모아서 평균을 낸다.
      // 말처럼 큰 소리는 제외한다. 3초 안에 10개가 안 모이면
      // 모은 것의 평균(없으면 지금까지 본 가장 조용한 값)으로 잡는다.
      this.noiseWait++;
      if (energy < QUIET_ENERGY) {
        this.noiseSum += energy;
        this.noiseFrames++;
      }
      this.noiseMin = Math.min(this.noiseMin, energy);
      if (this.noiseFrames >= NOISE_FRAMES_NEEDED || this.noiseWait >= NOISE_WAIT_MAX_FRAMES) {
        this.noiseFloor = this.noiseFrames > 0 ? this.noiseSum / this.noiseFrames : this.noiseMin;
      }
      return;
    }
    const threshold = Math.max(this.noiseFloor * this.sensitivity, MIN_THRESHOLD);
    // 누를 때만 모드에서는 모든 프레임을 말로 본다. 침묵이어도 녹음은
    // 계속되고, 떼기 전까지는 자동 종료가 안 된다.
    const isSpeech = this.ptt ? true : energy > threshold;
    if (!isSpeech && !this.speaking) {
      // 조용할 때 소음 기준을 천천히 따라간다
      this.noiseFloor = this.noiseFloor * 0.98 + energy * 0.02;
    }
    if (isSpeech && !this.speaking) {
      this.speaking = true;
      this.silenceFrames = 0;
      this.speechFrames = 0;
      this.utterance = [...this.preRoll];
      // 말 시작 직전 0.15초만 남겨서 첫 음절이 잘리지 않게 한다.
      this.speechStart = Math.max(0, this.utterance.length - 2400);
      this.speechEnd = this.utterance.length;
      this.onSpeechStart();
    }
    if (this.speaking) {
      this.utterance.push(...frame);
      if (isSpeech) {
        this.silenceFrames = 0;
        this.speechFrames++;
        this.speechEnd = this.utterance.length;
      } else if (++this.silenceFrames >= SILENCE_FRAMES_TO_END) {
        this._emit();
        return;
      }
      if (this.utterance.length > MAX_UTTERANCE_SEC * 16000) this._emit();
    }
  }

  _emit() {
    // 발화가 끝났음을 알린다 (인식 결과가 버려져도 — 화면 표시용).
    this.onSpeechEnd();
    // 말 앞뒤 침묵은 잘라내고 진짜 말한 구간만 인식에 넘긴다.
    // (프리롤 0.5초 + 끝 0.9초 침묵이 그대로 들어가면 모델이 헛소리·반복을 낸다.)
    const out = new Float32Array(this.utterance.slice(this.speechStart ?? 0, this.speechEnd ?? this.utterance.length));
    const speechFrames = this.speechFrames;
    this.speaking = false;
    this.silenceFrames = 0;
    this.speechFrames = 0;
    this.speechStart = 0;
    this.speechEnd = 0;
    this.utterance = [];
    // 0.4초 미만, 또는 진짜 말(임계값 넘는 프레임)이 0.5초 미만이면 잡음으로 버린다.
    // 짧은 소음 spikes가 발화로 인식돼서 음성인식이 헛소리를 내는 것을 막는다.
    if (out.length > 16000 * 0.4 && speechFrames >= 5) this.onUtterance(out);
  }

  // 보내기 버튼: 지금 듣고 있는 발화를 침묵을 기다리지 않고 바로 넘긴다.
  // 보낼 말이 없으면 false (계속 듣는다).
  flush() {
    if (!this.running || !this.speaking) return false;
    const hadSpeech = this.speechFrames >= 5 && this.utterance.length > 16000 * 0.4;
    this._emit();
    return hadSpeech;
  }
}
