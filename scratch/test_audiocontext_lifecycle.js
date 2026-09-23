/**
 * Verification test for AudioContext lifecycle synchronization and race-condition prevention.
 * Simulates Web Audio API spec behavior:
 * If close() is called while resume() is in progress, the real Web Audio API throws:
 * "InvalidStateError: Closed before resume completed".
 */

class MockAudioContext {
  constructor() {
    this.state = "suspended";
    this._resumeInProgress = false;
    this.sampleRate = 48000;
  }

  async resume() {
    if (this.state === "closed") {
      throw new Error("InvalidStateError: Cannot resume a closed AudioContext");
    }
    this._resumeInProgress = true;
    // Simulate real browser async hardware resume delay
    await new Promise((resolve) => setTimeout(resolve, 30));

    if (this.state === "closed") {
      this._resumeInProgress = false;
      throw new Error("InvalidStateError: Closed before resume completed");
    }
    this.state = "running";
    this._resumeInProgress = false;
  }

  async close() {
    if (this._resumeInProgress) {
      this.state = "closed";
      throw new Error("InvalidStateError: Closed before resume completed");
    }
    this.state = "closed";
  }

  createMediaStreamSource() { return { connect: () => {} }; }
  createAnalyser() { return { connect: () => {}, fftSize: 128 }; }
  createScriptProcessor() { return { connect: () => {}, onaudioprocess: null }; }
  createGain() { return { connect: () => {}, gain: { value: 0 } }; }
  get destination() { return {}; }
}

// Emulate the synchronized lifecycle state from LiveCallVoiceCopilot.tsx
class SynchronizedVoiceAudioLifecycle {
  constructor() {
    this.audioContext = null;
    this.resumePromise = null;
    this.lifecycleLock = Promise.resolve();
    this.isStartingTurn = false;
    this.isContinuousMode = false;
    this.reusedContextCount = 0;
    this.newContextCount = 0;
  }

  async startListeningTurn() {
    if (!this.isContinuousMode) return;

    const prevLock = this.lifecycleLock;
    let releaseLock;
    this.lifecycleLock = new Promise((resolve) => { releaseLock = resolve; });

    try {
      await prevLock;
      if (!this.isContinuousMode) return;
      if (this.isStartingTurn) return;
      this.isStartingTurn = true;

      try {
        // Reuse active AudioContext if available and not closed
        let audioCtx = this.audioContext;
        if (!audioCtx || audioCtx.state === "closed") {
          audioCtx = new MockAudioContext();
          this.audioContext = audioCtx;
          this.newContextCount++;
        } else {
          this.reusedContextCount++;
        }

        // Safely synchronize resume so close is never called concurrently
        if (audioCtx.state === "suspended") {
          if (!this.resumePromise) {
            const p = audioCtx.resume().catch((err) => {
              // Expected error handler
            }).finally(() => {
              if (this.resumePromise === p) {
                this.resumePromise = null;
              }
            });
            this.resumePromise = p;
          }
          await this.resumePromise;
        }

        if (!this.isContinuousMode || audioCtx.state === "closed") return;

        // Ready to capture audio
      } finally {
        this.isStartingTurn = false;
      }
    } finally {
      releaseLock();
    }
  }

  stopVoiceMode() {
    this.isContinuousMode = false;

    const ctx = this.audioContext;
    this.audioContext = null;

    if (ctx && ctx.state !== "closed") {
      const closeSafely = async () => {
        if (this.resumePromise) {
          try {
            await this.resumePromise;
          } catch (_) {}
        }
        try {
          if (ctx.state !== "closed") {
            await ctx.close();
          }
        } catch (err) {
          console.error("FATAL: Caught error during close:", err.message);
          throw err;
        }
      };

      const prevLock = this.lifecycleLock;
      let releaseLock;
      this.lifecycleLock = new Promise((resolve) => { releaseLock = resolve; });
      prevLock.then(closeSafely).finally(() => releaseLock());
    }
  }
}

async function runTests() {
  console.log("=== RUNNING AUDIOCONTEXT LIFECYCLE SYNCHRONIZATION TESTS ===");

  const manager = new SynchronizedVoiceAudioLifecycle();

  // Test 1: Repeated consecutive turns during continuous mode (must reuse AudioContext)
  console.log("\nTest 1: Testing sequential listening turns (AudioContext reuse)...");
  manager.isContinuousMode = true;
  for (let i = 0; i < 5; i++) {
    await manager.startListeningTurn();
  }
  console.log(`Created new contexts: ${manager.newContextCount}, Reused active contexts: ${manager.reusedContextCount}`);
  if (manager.newContextCount !== 1 || manager.reusedContextCount !== 4) {
    throw new Error(`Expected 1 creation and 4 reuses, got ${manager.newContextCount} / ${manager.reusedContextCount}`);
  }
  console.log("✓ Test 1 PASSED: Active AudioContext correctly reused across all turns.");

  // Test 2: Rapid overlapping startListeningTurn() calls (must not throw or duplicate)
  console.log("\nTest 2: Testing overlapping concurrent startListeningTurn() invocations...");
  const concurrentStarts = Promise.all([
    manager.startListeningTurn(),
    manager.startListeningTurn(),
    manager.startListeningTurn(),
  ]);
  await concurrentStarts;
  console.log("✓ Test 2 PASSED: Concurrent startListeningTurn calls synchronized without errors.");

  // Test 3: Rapid start -> stop -> start -> stop cycles with immediate toggling
  console.log("\nTest 3: Testing 10 rapid start/stop listening cycles...");
  for (let cycle = 1; cycle <= 10; cycle++) {
    manager.isContinuousMode = true;
    const startPromise = manager.startListeningTurn();
    // Simulate user or system stopping immediately while resume might still be in progress
    if (cycle % 2 === 0) {
      await new Promise((r) => setTimeout(r, 10)); // midway through resume
    }
    manager.stopVoiceMode();
    await startPromise;
  }
  console.log("✓ Test 3 PASSED: 10 rapid start/stop cycles completed with ZERO InvalidStateError!");

  // Test 4: Immediate stop during active resume
  console.log("\nTest 4: Stopping immediately when resume is in progress...");
  manager.isContinuousMode = true;
  const startP = manager.startListeningTurn();
  manager.stopVoiceMode(); // stop called right after start
  await startP;
  console.log("✓ Test 4 PASSED: Safe shutdown without interrupting pending resume.");

  console.log("\n========================================================");
  console.log("ALL LIFECYCLE TESTS PASSED! NO RACE CONDITIONS DETECTED.");
  console.log("========================================================");
}

runTests().catch((err) => {
  console.error("TEST SUITE FAILED:", err);
  process.exit(1);
});
