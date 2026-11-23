// vad-processor.js — AudioWorklet: 마이크 입력을 20ms 단위로 메인 스레드에 보낸다.
class VadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = [];
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const ch = input[0];
      for (let i = 0; i < ch.length; i++) this.buf.push(ch[i]);
      if (this.buf.length >= 960) {
        this.port.postMessage({ type: 'chunk', data: new Float32Array(this.buf) });
        this.buf = [];
      }
    }
    return true;
  }
}
registerProcessor('vad-processor', VadProcessor);
