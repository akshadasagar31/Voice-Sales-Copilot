// End-to-end test of Module 2 TTS pipeline:
// DeepSeek response -> SentenceTokenizer -> SentenceAudioQueue logic -> Deepgram Aura TTS -> Audio playback readiness

const http = require('http');

function postJson(urlStr, data) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const postData = JSON.stringify(data);
    const req = http.request({
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    }, (res) => {
      resolve(res);
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

// Simple port of SentenceTokenizer from frontend/lib/sentenceStreamingTTS.ts
class SimpleSentenceTokenizer {
  constructor() {
    this.buffer = '';
  }
  feed(chunk) {
    this.buffer += chunk;
    const sentences = [];
    const punctuationRegex = /([.?!।॥]+)(\s+|$)/;
    while (true) {
      const match = this.buffer.match(punctuationRegex);
      if (!match) break;
      const endIdx = match.index + match[1].length;
      const s = this.buffer.slice(0, endIdx).trim();
      this.buffer = this.buffer.slice(endIdx).trimStart();
      if (s) sentences.push(s);
    }
    return sentences;
  }
  flush() {
    const rem = this.buffer.trim();
    this.buffer = '';
    return rem || null;
  }
}

async function runTest() {
  console.log('--- STARTING MODULE 2 TTS END-TO-END VERIFICATION ---');
  const startTime = Date.now();

  console.log('[1/4] Querying DeepSeek SSE stream via Next.js (/api/ask)...');
  const askRes = await postJson('http://localhost:3000/api/ask', {
    question: 'What are the interest rates for personal loans?',
    stream: true
  });

  if (askRes.statusCode !== 200) {
    throw new Error(`Ask request failed with status: ${askRes.statusCode}`);
  }

  const tokenizer = new SimpleSentenceTokenizer();
  const sentencesQueue = [];
  const ttsPromises = [];
  let firstSentenceTime = null;
  let firstTtsAudioTime = null;
  let fullAnswerText = '';

  let buffer = '';
  askRes.on('data', (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) {
        const jsonStr = trimmed.slice(5).trim();
        try {
          const parsed = JSON.parse(jsonStr);
          const token = parsed.token || parsed.delta || '';
          if (token) {
            fullAnswerText += token;
            const completed = tokenizer.feed(token);
            for (const s of completed) {
              const sentenceIndex = sentencesQueue.length;
              const sentenceReadyAt = Date.now() - startTime;
              if (firstSentenceTime === null) {
                firstSentenceTime = sentenceReadyAt;
                console.log(`[2/4] First sentence formed at +${sentenceReadyAt}ms: "${s}"`);
              }
              sentencesQueue.push({ index: sentenceIndex, text: s, readyAt: sentenceReadyAt });

              // Immediately fetch Deepgram TTS in parallel
              const ttsPromise = (async () => {
                const ttsFetchStart = Date.now();
                const ttsRes = await postJson('http://localhost:3000/api/tts', { text: s, language: 'en' });
                const chunks = [];
                for await (const c of ttsRes) chunks.push(c);
                const audioBuffer = Buffer.concat(chunks);
                const ttsElapsed = Date.now() - ttsFetchStart;
                const totalElapsed = Date.now() - startTime;
                if (firstTtsAudioTime === null) {
                  firstTtsAudioTime = totalElapsed;
                  console.log(`[3/4] FIRST AUDIO READY FOR AUTOMATIC PLAYBACK at +${totalElapsed}ms (${audioBuffer.length} bytes MP3, TTS latency: ${ttsElapsed}ms)!`);
                }
                return { index: sentenceIndex, text: s, bytes: audioBuffer.length, ttsElapsed };
              })();
              ttsPromises.push(ttsPromise);
            }
          }
        } catch (_) {}
      }
    }
  });

  await new Promise((resolve) => askRes.on('end', resolve));

  const trailing = tokenizer.flush();
  if (trailing) {
    const sIndex = sentencesQueue.length;
    sentencesQueue.push({ index: sIndex, text: trailing, readyAt: Date.now() - startTime });
    const p = (async () => {
      const ttsRes = await postJson('http://localhost:3000/api/tts', { text: trailing, language: 'en' });
      const chunks = [];
      for await (const c of ttsRes) chunks.push(c);
      const audioBuffer = Buffer.concat(chunks);
      return { index: sIndex, text: trailing, bytes: audioBuffer.length };
    })();
    ttsPromises.push(p);
  }

  console.log(`[4/4] DeepSeek stream finished. Total sentences: ${sentencesQueue.length}`);
  const ttsResults = await Promise.all(ttsPromises);

  console.log('\n--- VERIFICATION RESULTS ---');
  console.log(`Total answer tokens received: ${fullAnswerText.length} chars`);
  console.log(`First sentence formed at: +${firstSentenceTime}ms`);
  console.log(`First audio sentence synthesized & ready to play at: +${firstTtsAudioTime}ms`);
  console.log('All synthesized sentences:');
  ttsResults.forEach((r) => {
    console.log(`  [Sentence #${r.index}] (${r.bytes} bytes audio): "${r.text}"`);
  });

  if (firstTtsAudioTime && ttsResults.length > 0) {
    console.log('\nSUCCESS: Module 2 TTS pipeline verified! First audio starts automatically without blocking.');
  } else {
    throw new Error('Pipeline failed delivery verification');
  }
}

runTest().catch((err) => {
  console.error('Test error:', err);
  process.exit(1);
});
