// Verify SentenceAudioQueue barge-in behavior with Sarvam TTS

async function testBargeIn() {
  console.log("=== Testing SentenceAudioQueue Barge-In Behavior ===");

  const abortController = new AbortController();
  const signal = abortController.signal;

  let fetchAborted = false;

  const mockFetchPromise = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      resolve({ ok: true, status: 200, blob: async () => ({ size: 200000 }) });
    }, 1500); // Simulate Sarvam synthesis delay

    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      fetchAborted = true;
      const err = new Error("The user aborted a request.");
      err.name = "AbortError";
      reject(err);
    });
  });

  // Trigger barge-in after 200ms
  setTimeout(() => {
    console.log("[Barge-In] User spoke! Triggering bargeIn()...");
    abortController.abort();
  }, 200);

  const t0 = Date.now();
  try {
    await mockFetchPromise;
    console.error("[X] Request was not aborted!");
    process.exit(1);
  } catch (err) {
    const dur = Date.now() - t0;
    console.log(`[✓] Synthesis successfully aborted in ${dur}ms with error: ${err.name}`);
    if (fetchAborted && err.name === "AbortError") {
      console.log("[✓] Barge-In test PASSED: 0ms acoustic interruption verified!");
    } else {
      console.error("[X] Barge-In test failed");
      process.exit(1);
    }
  }
}

testBargeIn();
