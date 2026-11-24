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

export class VoiceCapture {
  constructor({ onUtterance, onLevel }) {
    this.onUtterance = onUtterance;
    this.onLevel = onLevel || (() => {});
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
    this.node = new AudioWorkletNode(this.ctx, 'vad-processor');
    src.connect(this.node);
    this.node.port.onmessage = (e) => this._onChunk(e.data.data);
    this.sampleRate = this.ctx.sampleRate;
    this.resampleStep = this.sampleRate / 16000;
    this.preRoll = [];
    this.frames = [];
    this.noiseFloor = null;
    this.noiseFrames = 0;
    this.speaking = false;
    this.silenceFrames = 0;
    this.speechFrames = 0;
    this.speechStart = 0;
    this.speechEnd = 0;
    this.utterance = [];
    this.running = true;
  }

  stop() {
    this.running = false;
    try { this.node?.disconnect(); } catch (_) {}
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch (_) {}
    if (this.ctx) this.ctx.close().catch(() => {});
    this.node = null;
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
      // 처음 1초로 방 소음 기준을 잡는다
      this.noiseFrames++;
      this.noiseFloor = this.noiseFloor === null ? energy : this.noiseFloor + energy;
      if (this.noiseFrames >= 10) this.noiseFloor /= 10;
      return;
    }
    const threshold = Math.max(this.noiseFloor * 6, 0.0002);
    const isSpeech = energy > threshold;
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
