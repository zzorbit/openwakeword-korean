"use strict";

// Match numpy.rint(samples * 32768): nearest integer, with half ties to even.
function pcm16Value(value) {
  const scaled = value * 32768;
  const floor = Math.floor(scaled);
  const fraction = scaled - floor;
  const rounded = fraction < 0.5 ? floor : fraction > 0.5 ? floor + 1 : floor % 2 === 0 ? floor : floor + 1;
  return Math.max(-32768, Math.min(32767, rounded));
}

class KowwMicrophoneProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSamples = 1280;
    this.buffer = new ArrayBuffer(this.frameSamples * 2);
    this.view = new DataView(this.buffer);
    this.offset = 0;
    this.failed = false;
  }

  fail(message) {
    if (!this.failed) this.port.postMessage({ type: "error", message });
    this.failed = true;
  }

  process(inputs, outputs) {
    // The worklet stays connected to a muted output; microphone audio is never played.
    for (const channels of outputs) for (const channel of channels) channel.fill(0);
    if (this.failed) return true;
    const channels = inputs[0];
    if (!channels || channels.length === 0) return true;
    const count = channels[0].length;
    if (channels.some(channel => channel.length !== count)) {
      this.fail("마이크 채널의 길이가 달라 시험을 중지했습니다.");
      return true;
    }
    for (let index = 0; index < count; index++) {
      let value = 0;
      for (const channel of channels) value += channel[index];
      value /= channels.length;
      if (!Number.isFinite(value) || Math.abs(value) > 1) {
        this.fail("마이크 입력에 유효하지 않은 값이 있어 시험을 중지했습니다.");
        return true;
      }
      this.view.setInt16(this.offset * 2, pcm16Value(value), true);
      this.offset++;
      if (this.offset === this.frameSamples) {
        this.port.postMessage({ type: "frame", pcm: this.buffer }, [this.buffer]);
        this.buffer = new ArrayBuffer(this.frameSamples * 2);
        this.view = new DataView(this.buffer);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("koww-microphone", KowwMicrophoneProcessor);
