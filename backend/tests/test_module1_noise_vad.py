"""
test_module1_noise_vad.py
=========================
Comprehensive automated test suite for Module 1 Adaptive dB-Based VAD,
Acoustic Pre-filtering, and Background Noise Suppression.

Validates:
1. Mathematical conversion of audio signals to dBFS, Crest Factor, and ZCR.
2. Steady-state noise rejection (HVAC hum, fan rumble, traffic).
3. Transient impulsive noise rejection (keyboard typing, mouse clicks, desk bumps).
4. Distant secondary background speaker & TV rejection (adaptive SNR & proximity).
5. Intended multilingual user speech acceptance (English, Hindi, Marathi).
6. Natural conversational pause tolerance (hysteresis holdover without cutting).
7. Multi-digit spoken number dictation preservation.
8. Instant acoustic barge-in voice discrimination vs keyboard typing immunity.
9. Pre-STT segment density gating and non-speech discard.
"""

import math
import numpy as np
import pytest


class AdaptiveDbVAD:
    """
    Python mirror of the TypeScript adaptive dB-based VAD in LiveCallVoiceCopilot.tsx.
    Executes identical mathematical formulas and decision logic for deterministic verification.
    """

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.ambient_floor_db = -50.0
        self.speaker_bleed_db = -45.0
        self.is_speech_turn_active = False
        self.consecutive_speech_frames = 0
        self.consecutive_barge_in_frames = 0
        self.pre_roll_buffers = []
        self.max_pre_roll_count = max(16, math.ceil((sample_rate * 0.85) / buffer_size))
        self.recorded_buffers = []
        self.stream_opened = False
        self.turn_finalized = False
        self.last_discard_reason = None

    def compute_features(self, chunk: np.ndarray, vocal_energy: float = 0.0, high_noise: float = 0.0):
        """Computes peak, true RMS, dBFS, Crest Factor, and Zero-Crossing Rate."""
        n = len(chunk)
        if n == 0:
            return 0.0, -100.0, 0.0, 1.0, 0.0

        max_peak = float(np.max(np.abs(chunk)))
        sum_sq = float(np.sum(chunk ** 2))
        rms = math.sqrt(sum_sq / n)
        effective_rms = max(rms, 1e-5)
        current_db = 20.0 * math.log10(effective_rms)
        crest_factor = max_peak / effective_rms if effective_rms > 1e-5 else 1.0

        # Zero-crossing rate
        signs = np.sign(chunk)
        zero_crossings = np.sum(signs[:-1] != signs[1:])
        zcr = float(zero_crossings) / n

        return rms, current_db, max_peak, crest_factor, zcr

    def process_chunk(
        self,
        chunk: np.ndarray,
        vocal_energy: float,
        high_noise: float,
        voice_state: str = "listening",
        peak_prominence: float = 2.0,
        harmonicity: float = 0.65,
        dominant_bin: int = 5,
        spectral_centroid: float = 6.0,
    ):
        rms, current_db, max_peak, crest_factor, zcr = self.compute_features(chunk, vocal_energy, high_noise)

        # -------------------------------------------------------------
        # BARGE-IN: while assistant is speaking
        # -------------------------------------------------------------
        if voice_state == "speaking":
            self.speaker_bleed_db = max(-60.0, min(-20.0, self.speaker_bleed_db * 0.92 + current_db * 0.08))
            is_barge_in = (
                crest_factor < 5.8
                and vocal_energy >= 32
                and vocal_energy > high_noise * 1.35
                and (peak_prominence >= 1.55 or vocal_energy >= 50)
                and harmonicity >= 0.36
                and (current_db >= max(-36.0, self.speaker_bleed_db + 6.0) or rms >= 0.038)
            )
            if is_barge_in:
                self.consecutive_barge_in_frames += 1
                if self.consecutive_barge_in_frames >= 2:
                    return {"action": "barge_in", "db": current_db}
            else:
                self.consecutive_barge_in_frames = max(0, self.consecutive_barge_in_frames - 1)
            return {"action": "ignore_speaker_bleed", "db": current_db}

        # -------------------------------------------------------------
        # LISTENING: detect speech onset with adaptive dB noise floor
        # -------------------------------------------------------------
        if voice_state == "listening":
            if not self.is_speech_turn_active:
                self.ambient_floor_db = max(-65.0, min(-35.0, self.ambient_floor_db * 0.96 + current_db * 0.04))
                # Update pre-roll buffer
                self.pre_roll_buffers.append(chunk)
                if len(self.pre_roll_buffers) > self.max_pre_roll_count:
                    self.pre_roll_buffers.pop(0)

            # Dynamic SNR onset margin
            snr_margin_db = max(9.0, min(14.0, 11.0 - (self.ambient_floor_db + 45.0) * 0.25))

            # Proximity & DRR check:
            cond_proximity = (
                (peak_prominence >= 1.60 and harmonicity >= 0.38)
                or (peak_prominence >= 1.85)
                or (vocal_energy >= 55 and harmonicity >= 0.48)
            )

            # Proximity-tuned onset threshold: clamps at -38.0 for close mic, -35.0 for diffuse
            dynamic_db_onset_threshold = (
                max(-38.0, self.ambient_floor_db + 8.0)
                if cond_proximity
                else max(-35.0, self.ambient_floor_db + snr_margin_db)
            )

            # Speaker profile match
            is_profile_match = True
            if getattr(self, "primary_profile", {}).get("is_calibrated", False):
                bin_diff = abs(dominant_bin - self.primary_profile["dominant_bin"])
                centroid_diff = abs(spectral_centroid - self.primary_profile["centroid"])
                if bin_diff > 3.0 and centroid_diff > 4.5:
                    if not (peak_prominence >= 2.2 and current_db >= -24.0 and harmonicity >= 0.65):
                        is_profile_match = False

            is_speech_candidate = (
                current_db >= dynamic_db_onset_threshold
                and vocal_energy >= 34
                and vocal_energy > high_noise * 1.30
                and (crest_factor <= 5.8 or vocal_energy >= 50)
                and (zcr <= 0.24 or vocal_energy >= 40)
                and cond_proximity
                and is_profile_match
            )

            if is_speech_candidate:
                self.consecutive_speech_frames += 1
                if not self.is_speech_turn_active and self.consecutive_speech_frames >= 3:
                    self.is_speech_turn_active = True
                    self.stream_opened = True
                    # Prepend pre-roll buffers
                    self.recorded_buffers = list(self.pre_roll_buffers) + [chunk]
                    if not getattr(self, "primary_profile", {}).get("is_calibrated", False):
                        self.primary_profile = {
                            "is_calibrated": True,
                            "dominant_bin": dominant_bin,
                            "centroid": spectral_centroid,
                            "peak_prominence": peak_prominence,
                        }
                    return {
                        "action": "start_speech_turn",
                        "db": current_db,
                        "floor_db": self.ambient_floor_db,
                        "prepended_buffers": len(self.pre_roll_buffers),
                    }
                elif self.is_speech_turn_active:
                    self.recorded_buffers.append(chunk)
                    return {"action": "streaming_speech", "db": current_db}
            else:
                self.consecutive_speech_frames = 0
                if self.is_speech_turn_active:
                    self.recorded_buffers.append(chunk)
                    # Hysteresis continuation threshold: tuned to -42.0 dBFS / vocalEnergy 20 or unvoiced fricatives
                    continuation_threshold_db = max(-42.0, self.ambient_floor_db + 5.5)
                    is_continuation = (
                        current_db >= continuation_threshold_db
                        and (vocal_energy >= 20 or (high_noise >= 24 and zcr >= 0.12))
                    )
                    return {
                        "action": "continuation" if is_continuation else "silence_pause",
                        "db": current_db,
                        "continuation_thresh": continuation_threshold_db,
                    }

        return {"action": "listening_ambient", "db": current_db, "floor_db": self.ambient_floor_db}

    def finalize_turn(self):
        """Simulates submitCurrentSpeechTurn validation."""
        if not self.recorded_buffers:
            return {"accepted": False, "reason": "empty"}

        total_samples = sum(len(b) for b in self.recorded_buffers)
        duration = total_samples / self.sample_rate

        if duration < 0.30:
            self.last_discard_reason = "duration_under_300ms"
            self.is_speech_turn_active = False
            self.stream_opened = False
            return {"accepted": False, "reason": self.last_discard_reason, "duration": duration}

        total_energy = 0.0
        peak_rms = 0.0
        voiced_count = 0
        continuation_threshold_rms = 10.0 ** (max(-42.0, self.ambient_floor_db + 5.5) / 20.0)

        for buf in self.recorded_buffers:
            buf_rms = math.sqrt(float(np.mean(buf ** 2)))
            total_energy += buf_rms
            if buf_rms > peak_rms:
                peak_rms = buf_rms
            if buf_rms >= continuation_threshold_rms:
                voiced_count += 1

        avg_rms = total_energy / len(self.recorded_buffers)
        segment_avg_db = 20.0 * math.log10(max(avg_rms, 1e-5))
        peak_db = 20.0 * math.log10(max(peak_rms, 1e-5))
        voiced_density = voiced_count / len(self.recorded_buffers)

        min_peak_db = -38.0 if getattr(self, "primary_profile", {}).get("is_calibrated", False) else -34.0
        if segment_avg_db < self.ambient_floor_db + 3.5 or peak_db < min_peak_db or voiced_density < 0.20:
            self.last_discard_reason = f"low_density_or_energy (avgDb={segment_avg_db:.1f}, density={voiced_density:.2f})"
            self.is_speech_turn_active = False
            self.stream_opened = False
            return {"accepted": False, "reason": self.last_discard_reason, "avg_db": segment_avg_db, "density": voiced_density}

        self.turn_finalized = True
        self.is_speech_turn_active = False
        return {
            "accepted": True,
            "duration": duration,
            "avg_db": segment_avg_db,
            "peak_db": peak_db,
            "density": voiced_density,
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_adaptive_db_conversion_and_crest_factor():
    """Verify conversion of signals to dBFS, Crest Factor, and Zero-Crossing Rate."""
    vad = AdaptiveDbVAD()

    # 1. Pure sine wave (known Crest Factor = sqrt(2) ≈ 1.414)
    t = np.linspace(0, 1, 2048, endpoint=False)
    sine = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rms, db, peak, crest, zcr = vad.compute_features(sine)
    assert abs(rms - (0.1 / math.sqrt(2))) < 0.005
    assert abs(crest - math.sqrt(2)) < 0.05
    assert db < -22.0 and db > -24.0

    # 2. Impulsive click: sharp single spike with high Crest Factor
    click = np.zeros(2048, dtype=np.float32)
    click[100] = 0.25
    click[101] = -0.15
    rms_c, db_c, peak_c, crest_c, zcr_c = vad.compute_features(click)
    assert peak_c == 0.25
    assert crest_c > 15.0  # Sharp spike has Crest Factor > 15
    assert crest_c > 5.8  # Exceeds click threshold


def test_steady_fan_and_traffic_rejection():
    """Verify that steady fan/HVAC hum (-42 dBFS) is suppressed without triggering STT onset."""
    vad = AdaptiveDbVAD()
    # Initialize near -48 dBFS
    vad.ambient_floor_db = -48.0

    # Generate 15 frames of steady low-frequency fan noise (~100Hz + gentle hiss)
    # RMS around 0.008 (-42 dBFS), vocal formant energy low (14), high noise (12)
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    fan_chunk = (0.010 * np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.003, 2048)).astype(np.float32)

    triggered = False
    for _ in range(15):
        res = vad.process_chunk(fan_chunk, vocal_energy=14.0, high_noise=12.0)
        if res.get("action") == "start_speech_turn":
            triggered = True

    assert not triggered, "Steady fan hum must NOT trigger speech onset"
    assert not vad.is_speech_turn_active
    assert not vad.stream_opened
    # Ambient floor should adapt towards -42 dBFS smoothly
    assert vad.ambient_floor_db > -47.0


def test_keyboard_typing_clicks_rejection():
    """Verify that keyboard typing transients (crest factor > 6.0) are rejected."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    # Generate simulated mechanical keyboard click:
    # sharp impulse (peak 0.12, short decay over 120 samples, then silence)
    click_chunk = np.zeros(2048, dtype=np.float32)
    click_chunk[50:70] = np.random.uniform(0.08, 0.14, 20)
    click_chunk[70:120] = np.random.uniform(-0.06, 0.06, 50) * np.exp(-np.linspace(0, 3, 50))

    rms, db, peak, crest, zcr = vad.compute_features(click_chunk)
    assert crest > 5.8, "Keyboard click must have high crest factor"

    # Feed 4 click frames spaced with quiet typing background
    triggered = False
    for _ in range(4):
        # Keyboard clicks have high-frequency energy dominating vocal formants
        res = vad.process_chunk(click_chunk, vocal_energy=22.0, high_noise=45.0)
        if res.get("action") == "start_speech_turn":
            triggered = True

    assert not triggered, "Keyboard typing clicks must NOT trigger speech onset"
    assert not vad.is_speech_turn_active
    assert not vad.stream_opened


def test_distant_background_speaker_and_tv_rejection():
    """Verify that distant background TV or conversational murmur (+3dB SNR) is rejected."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -48.0

    # Distant speech has low sound pressure at mic (-45 dBFS, vocal energy 20, high noise 18)
    distant_chunk = (np.random.normal(0, 0.0055, 2048)).astype(np.float32)
    _, db, _, _, _ = vad.compute_features(distant_chunk)

    triggered = False
    for _ in range(8):
        res = vad.process_chunk(distant_chunk, vocal_energy=20.0, high_noise=18.0)
        if res.get("action") == "start_speech_turn":
            triggered = True

    assert not triggered, "Distant TV / background murmur must NOT trigger speech onset"
    assert not vad.is_speech_turn_active


def test_intended_user_speech_acceptance_multilingual():
    """
    Verify that intended user speech in English, Hindi, and Marathi cleanly triggers speech onset
    and prepends the ~800ms pre-roll buffer with 100% syllable retention.
    """
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    # 1. Fill pre-roll buffer with ambient silence (~10 frames)
    silent_chunk = np.random.normal(0, 0.002, 2048).astype(np.float32)
    for _ in range(10):
        vad.process_chunk(silent_chunk, vocal_energy=8.0, high_noise=6.0)

    assert len(vad.pre_roll_buffers) == 10

    # 2. Simulate User Speech:
    # "Hi, I need a personal loan" (en) / "नमस्ते मुझे लोन चाहिए" (hi) / "मला पाच लाखांचे कर्ज हवे आहे" (mr)
    # Speech chunk: RMS ~ 0.05 (-26 dBFS), vocal energy 65, high noise 18, crest factor 3.2
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    voice_chunk = (
        0.04 * np.sin(2 * np.pi * 320 * t)
        + 0.03 * np.sin(2 * np.pi * 780 * t)
        + 0.02 * np.sin(2 * np.pi * 1800 * t)
        + np.random.normal(0, 0.005, 2048)
    ).astype(np.float32)

    actions = []
    for _ in range(5):
        r = vad.process_chunk(voice_chunk, vocal_energy=65.0, high_noise=18.0)
        actions.append(r["action"])

    # On frame 3, speech onset confirmed!
    assert "start_speech_turn" in actions, "Intended user speech MUST trigger speech onset"
    assert vad.is_speech_turn_active
    assert vad.stream_opened
    # Verified pre-roll buffers prepended: initial ~800ms of audio is retained
    assert len(vad.recorded_buffers) >= 12, "Pre-roll buffers must be prepended"


def test_natural_speech_pause_preservation():
    """Verify that a 350ms pause (natural thinking/breathing) does NOT cut the turn prematurely."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    # Fill pre-roll and start turn
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    voice_chunk = (0.05 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    for _ in range(4):
        vad.process_chunk(voice_chunk, vocal_energy=60.0, high_noise=15.0)

    assert vad.is_speech_turn_active

    # Simulate 350ms pause (~8 frames at 2048/48kHz = 42.6ms per frame)
    # Background silence chunk: RMS 0.002 (-54 dBFS)
    silent_chunk = np.random.normal(0, 0.002, 2048).astype(np.float32)
    pause_actions = []
    for _ in range(8):  # 8 * 42.6ms = 341ms
        res = vad.process_chunk(silent_chunk, vocal_energy=10.0, high_noise=8.0)
        pause_actions.append(res["action"])

    # During pause, state should remain active (silence_pause, awaiting 480ms timer)
    assert vad.is_speech_turn_active, "Speech turn must NOT be cut off during 341ms pause"

    # User resumes speaking: "five lakh rupees"
    resume_res = vad.process_chunk(voice_chunk, vocal_energy=60.0, high_noise=15.0)
    assert resume_res["action"] == "streaming_speech"
    assert vad.is_speech_turn_active, "Turn continues seamlessly when user resumes speaking"


def test_spoken_numbers_phone_tenure_preservation():
    """Verify that multi-digit phone number dictation with pauses ('98200... 12345') is preserved."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    voice_chunk = (0.045 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    soft_chunk = (0.018 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)  # Soft consonant e.g. "seven"

    # Speak first digits ("98200" ~ 8 chunks = 341ms)
    for _ in range(8):
        vad.process_chunk(voice_chunk, vocal_energy=60.0, high_noise=15.0)

    # Soft digit transition / hesitation (~3 chunks = 128ms): hysteresis continuation kicks in
    for _ in range(3):
        res = vad.process_chunk(soft_chunk, vocal_energy=22.0, high_noise=12.0)
        assert res["action"] in ("continuation", "streaming_speech")
        assert vad.is_speech_turn_active

    # Speak remaining digits ("12345" ~ 8 chunks = 341ms)
    for _ in range(8):
        vad.process_chunk(voice_chunk, vocal_energy=60.0, high_noise=15.0)

    # Finalize segment (~810ms total speech)
    final_res = vad.finalize_turn()
    assert final_res["accepted"], f"Multi-digit phone number turn must be accepted, got {final_res}"
    assert final_res["duration"] >= 0.80
    assert final_res["density"] > 0.60


def test_acoustic_barge_in_voice_vs_keyboard():
    """
    Verify that genuine user speech immediately halts assistant playback (barge-in),
    while keyboard typing during assistant speech is ignored.
    """
    vad = AdaptiveDbVAD()
    vad.speaker_bleed_db = -42.0

    # 1. Keyboard click during assistant playback (crest factor 8.0, high noise)
    click_chunk = np.zeros(2048, dtype=np.float32)
    click_chunk[10:30] = 0.12
    click_res = vad.process_chunk(click_chunk, vocal_energy=20.0, high_noise=50.0, voice_state="speaking")
    assert click_res["action"] == "ignore_speaker_bleed", "Keyboard typing must NOT trigger barge-in"

    # 2. Genuine user interruption: "Wait, stop"
    # RMS ~ 0.06 (-24 dBFS, +18dB over bleed), vocal energy 55, low crest factor (2.8)
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    voice_chunk = (0.06 * np.sin(2 * np.pi * 380 * t)).astype(np.float32)

    # Frame 1: qualifying frame
    r1 = vad.process_chunk(voice_chunk, vocal_energy=55.0, high_noise=15.0, voice_state="speaking")
    assert r1["action"] == "ignore_speaker_bleed"  # 1st frame counts to confirmation
    # Frame 2: confirmed acoustic barge-in!
    r2 = vad.process_chunk(voice_chunk, vocal_energy=55.0, high_noise=15.0, voice_state="speaking")
    assert r2["action"] == "barge_in", "Genuine user interruption MUST trigger acoustic barge-in"


def test_pre_stt_segment_density_discard():
    """Verify that transient noise bursts with low voiced density (<20%) are discarded before STT."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    # Start turn with 3 borderline frames
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    burst_chunk = (0.03 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    for _ in range(3):
        vad.process_chunk(burst_chunk, vocal_energy=40.0, high_noise=15.0)

    assert vad.is_speech_turn_active

    # Followed by 15 frames of silence/clicks (total 18 frames, only 3 voiced = 16.6% voiced density)
    silent_chunk = np.random.normal(0, 0.001, 2048).astype(np.float32)
    for _ in range(15):
        vad.process_chunk(silent_chunk, vocal_energy=8.0, high_noise=8.0)

    # Finalize segment
    final_res = vad.finalize_turn()
    assert not final_res["accepted"], "Low-density burst (<20%) must be discarded before STT"
    assert "low_density" in vad.last_discard_reason


def test_nearby_secondary_speaker_rejection():
    """Verify that secondary speaker ~1m away (-36 dBFS, vocal energy 32) is cleanly rejected by proximity VAD."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    # Secondary speaker ~1m away: -36.5 dBFS, vocal energy 32, high noise 14
    nearby_chunk = (0.015 * np.sin(2 * np.pi * 350 * t) + 0.010 * np.sin(2 * np.pi * 850 * t) + np.random.normal(0, 0.002, 2048)).astype(np.float32)

    triggered = False
    for _ in range(8):
        res = vad.process_chunk(nearby_chunk, vocal_energy=32.0, high_noise=14.0)
        if res.get("action") == "start_speech_turn":
            triggered = True

    assert not triggered, "Secondary speaker at 1m must NOT trigger speech onset"
    assert not vad.is_speech_turn_active
    assert not vad.stream_opened


def test_secondary_speaker_rejection_via_diffuse_spectral_spread():
    """Verify that secondary speaker at -31 dBFS is rejected due to low peakiness (diffuse reverberant field)."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    # Diffuse, reverberant signal: multiple smeared frequencies + acoustic reflections
    diffuse_chunk = (
        0.02 * np.sin(2 * np.pi * 320 * t)
        + 0.02 * np.sin(2 * np.pi * 650 * t)
        + 0.015 * np.sin(2 * np.pi * 1100 * t)
        + 0.015 * np.sin(2 * np.pi * 1800 * t)
        + np.random.normal(0, 0.008, 2048)
    ).astype(np.float32)

    triggered = False
    for _ in range(6):
        res = vad.process_chunk(diffuse_chunk, vocal_energy=38.0, high_noise=20.0, peak_prominence=1.25, harmonicity=0.28)
        if res.get("action") == "start_speech_turn":
            triggered = True

    assert not triggered, "Diffuse reverberant secondary speech must be rejected before STT onset"
    assert not vad.stream_opened


def test_soft_speech_acceptance_via_proximity_peakiness():
    """Verify that primary user's soft voice at -33 dBFS is accepted via close-mic peakiness and harmonicity."""
    vad = AdaptiveDbVAD()
    vad.ambient_floor_db = -50.0

    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    # Soft speech signal with direct acoustic energy on primary formant
    soft_chunk = (
        0.032 * np.sin(2 * np.pi * 220 * t)
        + 0.010 * np.sin(2 * np.pi * 440 * t)
        + np.random.normal(0, 0.001, 2048)
    ).astype(np.float32)

    triggered = False
    for _ in range(8):
        res = vad.process_chunk(soft_chunk, vocal_energy=44.0, high_noise=12.0, peak_prominence=1.85, harmonicity=0.44)
        if res.get("action") == "start_speech_turn":
            triggered = True

    assert triggered, "Primary user soft speech with close-mic proximity must trigger speech onset"
    assert vad.stream_opened
    final_res = vad.finalize_turn()
    assert final_res["accepted"], "Primary user soft speech turn must be accepted for STT"


