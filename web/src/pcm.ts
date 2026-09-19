const UPLINK_SR = 16000;
const DOWNLINK_SR = 24000;
const UPLINK_FRAME_SAMPLES = 320; // 20ms @ 16kHz, Volcengine duplex heartbeat size

export function floatToPcm16(input: Float32Array, inRate: number, outRate = UPLINK_SR): ArrayBuffer {
  const ratio = inRate / outRate;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const view = new DataView(new ArrayBuffer(outLen * 2));
  for (let i = 0; i < outLen; i += 1) {
    const sample = input[Math.min(input.length - 1, Math.floor(i * ratio))] || 0;
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(i * 2, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
  }
  return view.buffer;
}

export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private next = 0;

  async ensure(): Promise<AudioContext> {
    if (!this.ctx) {
      this.ctx = new AudioContext();
    }
    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }
    return this.ctx;
  }

  async push(pcm: ArrayBuffer): Promise<void> {
    const ctx = await this.ensure();
    const samples = new Int16Array(pcm);
    if (!samples.length) return;
    const data = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      data[i] = samples[i] / 32768;
    }
    const buffer = ctx.createBuffer(1, data.length, DOWNLINK_SR);
    buffer.copyToChannel(data, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.02, this.next);
    src.start(startAt);
    this.next = startAt + buffer.duration;
  }

  close(): void {
    this.next = 0;
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
    }
  }
}

export async function startMicCapture(
  onChunk: (pcm: ArrayBuffer) => void,
): Promise<{ stop: () => void }> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(1024, 1, 1);
  let pending = new Int16Array(0);
  processor.onaudioprocess = (event) => {
    const converted = new Int16Array(
      floatToPcm16(event.inputBuffer.getChannelData(0), ctx.sampleRate, UPLINK_SR),
    );
    const merged = new Int16Array(pending.length + converted.length);
    merged.set(pending);
    merged.set(converted, pending.length);
    let offset = 0;
    while (offset + UPLINK_FRAME_SAMPLES <= merged.length) {
      onChunk(merged.slice(offset, offset + UPLINK_FRAME_SAMPLES).buffer);
      offset += UPLINK_FRAME_SAMPLES;
    }
    pending = merged.slice(offset);
  };
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(ctx.destination);
  return {
    stop() {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      void ctx.close();
    },
  };
}
