# Module 1 VAD Verification Plan

## Objective
Verify that the adaptive dB-based VAD in `LiveCallVoiceCopilot.tsx` is **actively gating microphone audio BEFORE STT** and log per-frame decision metrics for:
- Primary user voice → ACCEPT → STT
- Nearby/background voice → REJECT → no STT
- Background noise → REJECT
- Natural pauses → preserved

---

## Current VAD Flow Analysis

### VAD Gate Location
**File:** `frontend/app/components/LiveCallVoiceCopilot.tsx`
**Function:** `processAudioChunk` (lines 1650-1846)
**Trigger:** AudioWorkletNode (`pcm-capture-processor.js`) or ScriptProcessorNode `onaudioprocess` callback

### Gate Points
| Stage | Function | Lines | Decision |
|-------|----------|-------|----------|
| **Pre-roll / Onset** | `processAudioChunk` | 1779-1798 | 3-frame qualifying → `startStreamingSTT()` |
| **Active Streaming** | `processAudioChunk` | 1799-1802 | `sendStreamingChunk()` per frame |
| **Hysteresis Continuation** | `processAudioChunk` | 1816-1842 | Relaxed threshold (`floor+4.5dB`, `vocalEnergy>=16`) |
| **Pre-STT Finalization** | `submitCurrentSpeechTurn` | 1374-1418 | Duration>300ms, `voicedDensity>=0.20`, energy checks |

### VAD IS Active (Not Bypassed)
- ✅ Microphone audio flows through `processAudioChunk` **before** any STT call
- ✅ `startStreamingSTT()` only called after `consecutiveSpeechFrames >= 3`
- ✅ `submitCurrentSpeechTurn()` validates segment before sending to Deepgram
- ✅ No code path sends audio to STT without VAD acceptance

---

## Verification Strategy

### 1. Add Diagnostic Logging to Existing VAD (No Code Changes to Logic)

**Target:** `frontend/app/components/LiveCallVoiceCopilot.tsx` → `processAudioChunk` function

**Add per-frame console logs** (guarded by `process.env.NODE_ENV !== 'production'`):

```typescript
// At line ~1779 (after isSpeechCandidate calculation)
if (process.env.NODE_ENV !== 'production') {
  console.log(`[VAD:TELEMETRY] dBFS=${currentDb.toFixed(1)} floor=${ambientNoiseFloorDbRef.current.toFixed(1)} SNR=${(currentDb - ambientNoiseFloorDbRef.current).toFixed(1)} onsetThresh=${dynamicDbOnsetThreshold.toFixed(1)} vocalE=${vocalEnergy.toFixed(1)} highN=${highNoise.toFixed(1)} crest=${crestFactor.toFixed(2)} zcr=${zcr.toFixed(3)} candidate=${isSpeechCandidate} frames=${consecutiveSpeechFramesRef.current} state=${voiceStateRef.current} action=${isSpeechCandidate ? (isSpeechTurnActiveRef.current ? 'STREAM' : consecutiveSpeechFramesRef.current >= 3 ? 'ONSET' : 'QUALIFY') : isSpeechTurnActiveRef.current ? 'HYSTERESIS' : 'REJECT'}`);
}
```

**Add finalization logs** in `submitCurrentSpeechTurn` (lines 1407-1418):
```typescript
if (process.env.NODE_ENV !== 'production') {
  console.log(`[VAD:FINALIZE] avgDb=${segmentAvgDb.toFixed(1)} peakDb=${peakDb.toFixed(1)} density=${(voicedDensity*100).toFixed(0)}% duration=${(duration*1000).toFixed(0)}ms accepted=${!(segmentAvgDb < ambientNoiseFloorDbRef.current + 3.0 || peakDb < -40.0 || voicedDensity < 0.20)}`);
}
```

### 2. Automated Test Scenarios (Run via `backend/tests/diagnose_vad.py`)

**Scenarios to validate** (already implemented in `diagnose_vad.py`):

| # | Scenario | Expected | Key Metrics |
|---|----------|----------|-------------|
| 1 | Primary user (~15cm, -24 dBFS) | ACCEPT | SNR>20dB, vocalE>70, onset 3-frame |
| 2 | Nearby speaker (~1m, -33 dBFS) | **CURRENTLY ACCEPTS (BUG)** | SNR~17dB, vocalE~46, passes all 5 onset checks |
| 3 | Mid-field speaker (~2m, -40 dBFS) | REJECT or ACCEPT? | SNR~10dB, vocalE~31, borderline |
| 4 | Distant TV (>3.5m, -48 dBFS) | REJECT | SNR~2dB, vocalE<28, fails onset |
| 5 | Steady fan/HVAC (-42 dBFS) | REJECT | vocalE<28, highNoise~vocalE, fails formant check |
| 6 | Keyboard clicks | REJECT | crest>5.8, highNoise>>vocalE, fails crest/ZCR |

**Run command:**
```bash
cd backend && python tests/diagnose_vad.py
```

### 3. Live Browser Verification Steps

1. Open `http://localhost:3000` (or deployed URL)
2. Open DevTools Console
3. Click **"Voice Assistant"** tab → **"Start Voice Mode"**
4. Observe console for `[VAD:TELEMETRY]` logs per frame (~42ms/frame)
5. Test scenarios:
   - **You speak** → Watch for `ONSET` at frame 3 → `STREAM` → `FINALIZE accepted=true`
   - **Colleague speaks nearby** → Watch for `ONSET`/`STREAM` (confirms bug)
   - **Silence/typing/fan** → Watch for `REJECT` / `AMBIENT_REJECT`
   - **You pause 400ms** → Watch for `HYSTERESIS_CONTINUATION` (may leak background)

### 4. Identify VAD Bypass Points (Audit)

**Check these paths for audio reaching STT without VAD:**

| Path | Function | Status |
|------|----------|--------|
| WebSocket streaming | `sendStreamingChunk` → `ws.send()` | ✅ Only called from `processAudioChunk` when `isSpeechTurnActive` |
| HTTP fallback | `processSpeechTurn` → `fetch('/api/voice-entry')` | ✅ Only called from `submitCurrentSpeechTurn` after validation |
| Manual button | `onAudioRecorded` prop | ⚠️ Check if any UI bypasses VAD |
| Simulation mode | `runSimulationTurn` | ✅ Bypasses VAD intentionally (test presets) |

**No bypass found in continuous voice mode.**

---

## Root Cause of Nearby Voice Acceptance

**Location:** `processAudioChunk` lines 1779-1784

```typescript
const isSpeechCandidate =
  currentDb >= dynamicDbOnsetThreshold &&      // -33 dBFS >= -42 dBFS ✅
  vocalEnergy >= 28 &&                          // 46 >= 28 ✅
  vocalEnergy > highNoise * 1.30 &&             // 46 > 15*1.3=19.5 ✅
  (crestFactor <= 5.8 || vocalEnergy >= 50) &&  // 2.4 <= 5.8 ✅
  (zcr <= 0.22 || vocalEnergy >= 40);           // 0.08 <= 0.22 ✅
```

**All 5 conditions pass for nearby speaker.** The acoustic threshold **cannot physically distinguish** a nearby secondary speaker from a soft primary speaker.

**Hysteresis leak:** During pause, continuation threshold drops to `-48 dBFS` / `vocalEnergy >= 16`, allowing background murmurs to hold the turn open.

---

## Files to Modify for Verification (Diagnostic Only)

| File | Change Type | Purpose |
|------|-------------|---------|
| `frontend/app/components/LiveCallVoiceCopilot.tsx` | Add `console.log` in `processAudioChunk` + `submitCurrentSpeechTurn` | Per-frame + finalization telemetry |
| `backend/tests/diagnose_vad.py` | Run as-is | Automated scenario validation |
| `backend/tests/test_vad_acceptance_analysis.py` | Run as-is | Unit test validation |

---

## Validation Checklist

- [ ] VAD telemetry logs appear in browser console during live test
- [ ] Primary user speech: `ONSET` at frame 3 → `ACCEPT` at finalize
- [ ] Nearby speaker: **Currently shows `ONSET`/`ACCEPT`** (confirms root cause)
- [ ] Background noise/fan: `REJECT` at all frames
- [ ] Keyboard clicks: `REJECT` (crest factor / highNoise gate)
- [ ] Natural pause (400ms): `HYSTERESIS_CONTINUATION` holds turn
- [ ] No audio reaches `/api/voice-entry` or WebSocket without VAD acceptance

---

## Next Steps After Verification

Once verified, the fix requires **speaker discrimination** (not in scope for verification):
1. Spectral centroid / rolloff proximity cue (nearby = brighter)
2. Speaker embedding enrollment + cosine similarity gate
3. Syllabic modulation spectrum (3-8 Hz) check
4. Multi-mic beamforming (if hardware supports)

---

## Plan Ready for Execution

This plan is **implementation-ready** for a verification agent. No logic changes—only diagnostic logging additions to existing VAD functions.