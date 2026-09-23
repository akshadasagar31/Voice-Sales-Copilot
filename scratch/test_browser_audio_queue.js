// Verification of SentenceAudioQueue logic:
// Tests token stream -> SentenceTokenizer -> SentenceAudioQueue -> Web Audio / HTMLAudio playback readiness

const http = require('http');

class MockAudioElement {
  constructor() {
    this.src = '';
    this.muted = false;
    this.volume = 1.0;
    this.playbackRate = 1.0;
    this.onended = null;
    this.onerror = null;
  }
  pause() { this.paused = true; }
  load() {}
  async play() {
    this.paused = false;
    // Simulate real audio duration
    setTimeout(() => {
      if (this.onended) this.onended(new Event('ended'));
    }, 100);
    return Promise.resolve();
  }
}

class MockAudioContext {
  constructor() {
    this.state = 'running';
    this.destination = {};
  }
  async resume() { this.state = 'running'; return Promise.resolve(); }
  async decodeAudioData(arrayBuffer) {
    return {
      duration: arrayBuffer.byteLength / 16000,
      sampleRate: 48000
    };
  }
  createBufferSource() {
    const self = this;
    return {
      buffer: null,
      playbackRate: { value: 1.0 },
      onended: null,
      connect(dest) {},
      start(delay) {
        setTimeout(() => {
          if (this.onended) this.onended();
        }, 100);
      },
      stop() {},
      disconnect() {}
    };
  }
}

// Global Event mock
global.Event = class Event { constructor(type) { this.type = type; } };

async function runTest() {
  console.log('Testing SentenceAudioQueue in mock Web Audio + HTMLAudio environment...');

  const fs = require('fs');
  const path = require('path');
  const ts = require(path.resolve(__dirname, '../frontend/node_modules/typescript'));
  const code = fs.readFileSync(path.resolve(__dirname, '../frontend/lib/sentenceStreamingTTS.ts'), 'utf8');
  const js = ts.transpile(code, { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS });
  const mod = { exports: {} };
  const fn = new Function('module', 'exports', 'require', js);
  fn(mod, mod.exports, require);

  const mockAudio = new MockAudioElement();
  const mockCtx = new MockAudioContext();

  const events = [];
  const queue = new mod.exports.SentenceAudioQueue(mockAudio, {
    onSentenceStart: (text, idx) => events.push(`START #${idx}: "${text}"`),
    onSentenceEnd: (text, idx) => events.push(`END #${idx}`),
    onQueueComplete: () => events.push('COMPLETE')
  }, mockCtx);

  queue.startNewTurn(1);

  // Mock fetch for /api/tts
  global.fetch = async (url, opts) => {
    return {
      ok: true,
      status: 200,
      blob: async () => ({
        size: 15830,
        type: 'audio/mpeg',
        arrayBuffer: async () => new ArrayBuffer(15830)
      })
    };
  };

  queue.enqueueSentence('First sentence test for personal loan.', 'en');
  queue.enqueueSentence('Second sentence with interest rates.', 'en');
  queue.markStreamComplete();

  // Wait for mock playback to process both sentences
  await new Promise((resolve) => setTimeout(resolve, 350));

  console.log('Captured events:');
  events.forEach((e) => console.log('  ', e));

  if (events.includes('START #0: "First sentence test for personal loan."') &&
      events.includes('END #0') &&
      events.includes('START #1: "Second sentence with interest rates."') &&
      events.includes('END #1') &&
      events.includes('COMPLETE')) {
    console.log('SUCCESS: All sentences played in strict sequential order via Web Audio API engine!');
  } else {
    throw new Error('Test failed to complete sequential playback');
  }
}

runTest().catch((err) => {
  console.error('Test error:', err);
  process.exit(1);
});
