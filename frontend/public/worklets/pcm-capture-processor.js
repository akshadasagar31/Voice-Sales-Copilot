// ============================================================================
// AUDIO WORKLET PROCESSOR: pcm-capture-processor.js
// Runs on the dedicated browser audio rendering thread.
// Replaces deprecated ScriptProcessorNode without blocking the main UI thread.
// ============================================================================

class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions?.bufferSize || 2048;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    const len = channel.length;
    for (let i = 0; i < len; i++) {
      this.buffer[this.bufferIndex++] = channel[i];
      if (this.bufferIndex >= this.bufferSize) {
        const chunk = new Float32Array(this.buffer);
        let sumSq = 0;
        for (let j = 0; j < this.bufferSize; j++) {
          sumSq += chunk[j] * chunk[j];
        }
        const rms = Math.sqrt(sumSq / this.bufferSize);

        this.port.postMessage({
          type: "audio_data",
          buffer: chunk,
          rms: rms,
        });
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);

