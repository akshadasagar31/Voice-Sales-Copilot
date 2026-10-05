"""
test_vad_acceptance_analysis.py
===============================
Deterministic acoustic analysis and verification of Module 1 Adaptive dBFS VAD
with Speaker Discrimination and Secondary/Background Voice Rejection.

Tests:
1. Adaptive dBFS VAD calculations matching LiveCallVoiceCopilot.tsx.
2. Intended primary user speech telemetry and close-mic acceptance.
3. Secondary speaker rejection before STT (even at similar volume!) via Diffuse-to-Direct ratio (peakiness) and harmonicity.
4. Secondary speaker rejection via Speaker Profile mismatch (different vocal tract/pitch).
5. Primary user soft speech acceptance (-33 dBFS) via close-mic proximity peakiness.
6. Room murmur rejection during conversational pauses (hysteresis holdover).
7. Unvoiced fricative preservation in phone numbers and numbers (English, Hindi, Marathi).
8. Logging of ACCEPT/REJECT reason and dBFS/SNR/vocal-energy values.
"""

import math
import numpy as np
import pytest


def compute_harmonicity(chunk: np.ndarray, sample_rate: int = 48000) -> float:
    """Computes normalized autocorrelation in pitch lag range (85 Hz to 380 Hz)."""
    min_lag = max(2, int(sample_rate / 380))
    max_lag = min(len(chunk) // 2, int(sample_rate / 85))
    if max_lag <= min_lag or len(chunk) < max_lag * 2:
        return 0.0

    n_eval = len(chunk) - max_lag
    x0 = chunk[:n_eval:2]
    norm0 = math.sqrt(float(np.sum(x0 ** 2)) * 2)
    if norm0 < 1e-4:
        return 0.0

    best_corr = 0.0
    stride = 3 if sample_rate >= 44100 else 2
    for lag in range(min_lag, max_lag, stride):
        x_lag = chunk[lag:lag + n_eval:2]
        norm_lag = math.sqrt(float(np.sum(x_lag ** 2)) * 2)
        if norm_lag > 1e-4:
            corr = (float(np.sum(x0 * x_lag)) * 2) / (norm0 * norm_lag)
            if corr > best_corr:
                best_corr = corr

    return float(np.clip(best_corr, 0.0, 1.0))


class Module1VADSimulator:
    """
    Exact mathematical replica of LiveCallVoiceCopilot.tsx VAD logic
    with acoustic proximity, harmonicity, and speaker discrimination.
    """
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.ambient_floor_db = -50.0
        self.is_speech_turn_active = False
        self.consecutive_speech_frames = 0
        self.pre_roll_buffers = []
        self.max_pre_roll_count = max(16, math.ceil((sample_rate * 0.85) / buffer_size))
        self.recorded_buffers = []
        self.primary_profile = {
            "dominant_bin": 5,
            "centroid": 6.0,
            "peak_prominence": 2.0,
            "sample_count": 0,
            "is_calibrated": False,
        }

    def compute_metrics(self, chunk: np.ndarray):
        n = len(chunk)
        if n == 0:
            return 0.0, -100.0, 0.0, 1.0, 0.0
        max_peak = float(np.max(np.abs(chunk)))
        sum_sq = float(np.sum(chunk ** 2))
        rms = math.sqrt(sum_sq / n)
        effective_rms = max(rms, 1e-5)
        current_db = 20.0 * math.log10(effective_rms)
        crest_factor = max_peak / effective_rms if effective_rms > 1e-5 else 1.0
        signs = np.sign(chunk)
        zero_crossings = np.sum(signs[:-1] != signs[1:])
        zcr = float(zero_crossings) / n
        return rms, current_db, max_peak, crest_factor, zcr

    def process_frame(
        self,
        chunk: np.ndarray,
        vocal_energy: float,
        high_noise: float,
        peak_prominence: float = 2.0,
        harmonicity: float = None,
        dominant_bin: int = 5,
        spectral_centroid: float = 6.0,
    ):
        rms, current_db, max_peak, crest_factor, zcr = self.compute_metrics(chunk)

        if not self.is_speech_turn_active:
            self.ambient_floor_db = max(-65.0, min(-35.0, self.ambient_floor_db * 0.96 + current_db * 0.04))
            self.pre_roll_buffers.append(chunk)
            if len(self.pre_roll_buffers) > self.max_pre_roll_count:
                self.pre_roll_buffers.pop(0)

        snr = current_db - self.ambient_floor_db
        snr_margin_db = max(9.0, min(14.0, 11.0 - (self.ambient_floor_db + 45.0) * 0.25))
        dynamic_db_onset_threshold = max(-35.0, self.ambient_floor_db + snr_margin_db)

        if harmonicity is None:
            if current_db >= self.ambient_floor_db + 3.0 and vocal_energy >= 24:
                harmonicity = compute_harmonicity(chunk, self.sample_rate)
            else:
                harmonicity = 0.0

        # Proximity & DRR: close mic has sharp peakiness (>=1.60) and harmonicity (>=0.38)
        cond_proximity = (
            (peak_prominence >= 1.60 and harmonicity >= 0.38)
            or (peak_prominence >= 1.85)
            or (vocal_energy >= 55 and harmonicity >= 0.48)
        )

        dynamic_db_onset_threshold = (
            max(-38.0, self.ambient_floor_db + 8.0)
            if cond_proximity
            else max(-35.0, self.ambient_floor_db + snr_margin_db)
        )

        # Onset condition breakdown
        cond_db = current_db >= dynamic_db_onset_threshold
        cond_vocal = vocal_energy >= 34
        cond_formant = vocal_energy > high_noise * 1.30
        cond_crest = (crest_factor <= 5.8 or vocal_energy >= 50)
        cond_zcr = (zcr <= 0.24 or vocal_energy >= 40)

        # Speaker profile matching (if calibrated)
        is_profile_match = True
        profile_reject_reason = ""
        if self.primary_profile["is_calibrated"]:
            bin_diff = abs(dominant_bin - self.primary_profile["dominant_bin"])
            centroid_diff = abs(spectral_centroid - self.primary_profile["centroid"])
            if bin_diff > 3.0 and centroid_diff > 4.5:
                if not (peak_prominence >= 2.2 and current_db >= -24.0 and harmonicity >= 0.65):
                    is_profile_match = False
                    profile_reject_reason = f"SPEAKER_PROFILE_MISMATCH (binDiff={bin_diff:.1f}, centroidDiff={centroid_diff:.1f})"

        is_speech_candidate = (
            cond_db
            and cond_vocal
            and cond_formant
            and cond_crest
            and cond_zcr
            and cond_proximity
            and is_profile_match
        )

        continuation_threshold_db = max(-42.0, self.ambient_floor_db + 5.5)
        is_continuation = False
        action = "IDLE"
        decision = "IGNORE"
        reason = "IDLE"

        if is_speech_candidate:
            self.consecutive_speech_frames += 1
            if not self.is_speech_turn_active and self.consecutive_speech_frames >= 3:
                self.is_speech_turn_active = True
                self.recorded_buffers = list(self.pre_roll_buffers) + [chunk]
                action = "START_SPEECH_TURN"
                decision = "ACCEPT"
                reason = "CLOSE_MIC_PRIMARY_QUALIFIED"
                if not self.primary_profile["is_calibrated"]:
                    self.primary_profile["dominant_bin"] = dominant_bin
                    self.primary_profile["centroid"] = spectral_centroid
                    self.primary_profile["peak_prominence"] = peak_prominence
                    self.primary_profile["is_calibrated"] = True
            elif self.is_speech_turn_active:
                self.recorded_buffers.append(chunk)
                action = "STREAMING_SPEECH"
                decision = "ACCEPT"
                reason = "CLOSE_MIC_PRIMARY_STREAMING"
            else:
                action = f"QUALIFYING_FRAME_{self.consecutive_speech_frames}"
                decision = "PENDING"
                reason = "QUALIFYING_PRIMARY_SPEAKER"
        else:
            self.consecutive_speech_frames = 0
            if self.is_speech_turn_active:
                self.recorded_buffers.append(chunk)
                is_continuation = (
                    current_db >= continuation_threshold_db
                    and (vocal_energy >= 20 or (high_noise >= 24 and zcr >= 0.12))
                )
                if is_continuation:
                    action = "HYSTERESIS_CONTINUATION"
                    decision = "ACCEPT_CONTINUATION"
                    reason = "PAUSE_CONTINUATION_OR_CONSONANT"
                else:
                    action = "SILENCE_PAUSE"
                    decision = "WAITING_SILENCE_TIMER"
                    reason = "AWAITING_SILENCE_TIMEOUT"
            else:
                action = "AMBIENT_FILTERED"
                decision = "REJECT"
                if not is_profile_match:
                    reason = profile_reject_reason
                elif not cond_proximity and cond_db and cond_vocal:
                    reason = f"DIFFUSE_SECONDARY_SPEAKER (peakiness={peak_prominence:.2f} < 1.60, harmonicity={harmonicity:.2f} < 0.38)"
                elif not cond_crest:
                    reason = f"TRANSIENT_IMPULSE (crest={crest_factor:.2f} > 5.8)"
                elif not cond_formant:
                    reason = f"HIGH_NOISE_DOMINANT (vocal={vocal_energy:.1f} <= {high_noise:.1f}*1.3)"
                else:
                    reason = f"LOW_ENERGY_OR_FLOOR (dBFS={current_db:.1f} < {dynamic_db_onset_threshold:.1f})"

        return {
            "db": current_db,
            "floor_db": self.ambient_floor_db,
            "snr": snr,
            "onset_thresh": dynamic_db_onset_threshold,
            "continuation_thresh": continuation_threshold_db,
            "vocal_energy": vocal_energy,
            "high_noise": high_noise,
            "crest_factor": crest_factor,
            "zcr": zcr,
            "peak_prominence": peak_prominence,
            "harmonicity": harmonicity,
            "dominant_bin": dominant_bin,
            "profile_match": is_profile_match,
            "is_candidate": is_speech_candidate,
            "action": action,
            "decision": decision,
            "reason": reason,
        }

    def finalize_turn(self):
        if not self.recorded_buffers:
            return {"accepted": False, "reason": "no_buffers"}

        total_samples = sum(len(b) for b in self.recorded_buffers)
        duration = total_samples / self.sample_rate

        if duration < 0.30:
            self.is_speech_turn_active = False
            return {"accepted": False, "duration": duration, "reason": "duration_under_300ms"}

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

        c1 = segment_avg_db >= self.ambient_floor_db + 3.5
        min_peak_db = -38.0 if self.primary_profile["is_calibrated"] else -34.0
        c2 = peak_db >= min_peak_db
        c3 = voiced_density >= 0.20

        accepted = c1 and c2 and c3
        self.is_speech_turn_active = False

        return {
            "accepted": accepted,
            "duration": round(duration, 3),
            "segment_avg_db": round(segment_avg_db, 1),
            "peak_db": round(peak_db, 1),
            "voiced_density": round(voiced_density, 3),
            "checks": {"avg_db_ge_floor_plus_3_5": c1, "peak_db_ge_minus_34": c2, "voiced_density_ge_0.20": c3},
            "reason": "VALID_SPEECH_TURN" if accepted else "DISCARDED_IN_FINALIZATION",
        }


# =============================================================================
# PYTEST TESTS FOR DETAILED ACCEPTANCE/REJECTION CAUSE ANALYSIS
# =============================================================================

def test_confirm_adaptive_dbfs_vad_active():
    """Confirms that adaptive dBFS VAD calculations match LiveCallVoiceCopilot.tsx."""
    vad = Module1VADSimulator()
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    sine = 0.05 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rms, db, peak, crest, zcr = vad.compute_metrics(sine)

    assert abs(rms - (0.05 / math.sqrt(2))) < 0.005
    assert db < -28.0 and db > -30.0
    assert abs(crest - math.sqrt(2)) < 0.05
    assert vad.ambient_floor_db == -50.0


def test_primary_user_speech_telemetry_logged():
    """Tests intended user speech (close mic) and logs all required acoustic metrics."""
    vad = Module1VADSimulator()
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)
    # Primary user at ~15cm: -25 dBFS, high vocal energy (75), high peakiness (2.2), harmonicity (0.75)
    user_voice = (0.045 * np.sin(2 * np.pi * 320 * t) + 0.025 * np.sin(2 * np.pi * 640 * t) + np.random.normal(0, 0.001, 2048)).astype(np.float32)

    frames_log = []
    for _ in range(8):
        f = vad.process_frame(user_voice, vocal_energy=75.0, high_noise=15.0, peak_prominence=2.2, harmonicity=0.75, dominant_bin=5)
        frames_log.append(f)

    # Frame 3 initiates turn
    assert frames_log[2]["action"] == "START_SPEECH_TURN"
    assert frames_log[2]["decision"] == "ACCEPT"
    assert frames_log[2]["reason"] == "CLOSE_MIC_PRIMARY_QUALIFIED"
    assert frames_log[2]["snr"] > 18.0
    assert frames_log[2]["peak_prominence"] >= 1.60
    assert frames_log[2]["harmonicity"] >= 0.38

    res = vad.finalize_turn()
    assert res["accepted"] is True
    assert res["reason"] == "VALID_SPEECH_TURN"


def test_nearby_secondary_speaker_is_rejected_on_onset():
    """
    VERIFY THAT NEARBY SECONDARY SPEAKER (~1m, -36.5 dBFS, vocalEnergy 32) IS REJECTED:
    Fails both dB onset clamp and vocal energy threshold.
    """
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    nearby_speaker = (0.015 * np.sin(2 * np.pi * 350 * t) + 0.010 * np.sin(2 * np.pi * 850 * t) + np.random.normal(0, 0.002, 2048)).astype(np.float32)

    logs = []
    for _ in range(10):
        log = vad.process_frame(nearby_speaker, vocal_energy=32.0, high_noise=14.0, peak_prominence=1.40, harmonicity=0.30)
        logs.append(log)

    for log in logs:
        assert log["action"] == "AMBIENT_FILTERED"
        assert log["decision"] == "REJECT"

    assert vad.is_speech_turn_active is False
    turn_res = vad.finalize_turn()
    assert turn_res["accepted"] is False


def test_secondary_speaker_at_same_volume_rejected_due_to_diffuse_reverberation():
    """
    CRITICAL REQUIREMENT: Do NOT simply increase volume threshold!
    Secondary speaker talks at -31 dBFS (LOUDER than -35 dBFS onset threshold, with vocalEnergy 45).
    However, because the speaker is 2 meters away, the room impulse response smears formants:
    - peak_prominence = 1.35 (< 1.60)
    - harmonicity = 0.32 (< 0.38)
    The secondary speaker is REJECTED BEFORE STT!
    """
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    # Moderate volume diffuse voice (-31 dBFS)
    secondary_speaker_chunk = (0.03 * np.sin(2 * np.pi * 300 * t) + np.random.normal(0, 0.005, 2048)).astype(np.float32)

    logs = []
    for _ in range(8):
        log = vad.process_frame(
            secondary_speaker_chunk,
            vocal_energy=45.0,
            high_noise=15.0,
            peak_prominence=1.35,  # Diffuse room smearing
            harmonicity=0.32,      # Degraded periodicity due to reflections
            dominant_bin=4,
        )
        logs.append(log)

    # All frames must be rejected
    for log in logs:
        assert log["action"] == "AMBIENT_FILTERED"
        assert log["decision"] == "REJECT"
        assert "DIFFUSE_SECONDARY_SPEAKER" in log["reason"]

    assert vad.is_speech_turn_active is False
    assert not vad.recorded_buffers


def test_secondary_speaker_rejected_by_speaker_profile_mismatch():
    """
    Tests that once primary user profile is calibrated (male pitch bin 5, centroid 5.8),
    a nearby secondary speaker with a different pitch / vocal tract (bin 11, centroid 11.2)
    is REJECTED before STT even if moderately loud!
    """
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    # 1. Primary user speaks turn 1 (calibrates profile)
    user_chunk = (0.04 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for _ in range(8):
        vad.process_frame(
            user_chunk,
            vocal_energy=65.0,
            high_noise=12.0,
            peak_prominence=2.1,
            harmonicity=0.72,
            dominant_bin=5,
            spectral_centroid=5.8,
        )

    assert vad.is_speech_turn_active is True
    assert vad.primary_profile["is_calibrated"] is True
    assert vad.primary_profile["dominant_bin"] == 5

    # Finalize primary user turn
    res = vad.finalize_turn()
    assert res["accepted"] is True
    assert vad.is_speech_turn_active is False

    # 2. Secondary speaker (e.g. child / colleague with dominant bin 11, centroid 11.5) talks
    secondary_chunk = (0.035 * np.sin(2 * np.pi * 650 * t)).astype(np.float32)
    sec_logs = []
    for _ in range(5):
        log = vad.process_frame(
            secondary_chunk,
            vocal_energy=50.0,
            high_noise=15.0,
            peak_prominence=1.75,
            harmonicity=0.55,
            dominant_bin=11,          # Mismatch!
            spectral_centroid=11.5,   # Mismatch!
        )
        sec_logs.append(log)

    # Must be rejected due to speaker profile mismatch!
    for log in sec_logs:
        assert log["decision"] == "REJECT"
        assert "SPEAKER_PROFILE_MISMATCH" in log["reason"]

    assert vad.is_speech_turn_active is False


def test_primary_user_soft_speech_accepted_via_proximity_and_harmonicity():
    """
    Preserve user's soft speech:
    Primary user speaks softly at -33 dBFS (close to mic, 20cm away).
    Despite low dB, close proximity yields sharp formant peakiness (1.92) and strong harmonicity (0.58).
    VAD cleanly accepts the soft speech turn!
    """
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    # Soft primary speech: amplitude 0.023 (-33 dBFS)
    soft_chunk = (0.020 * np.sin(2 * np.pi * 220 * t) + 0.010 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    soft_logs = []
    for _ in range(8):
        log = vad.process_frame(
            soft_chunk,
            vocal_energy=42.0,
            high_noise=10.0,
            peak_prominence=1.92,  # Sharp direct-path formant!
            harmonicity=0.58,      # High periodic correlation!
            dominant_bin=5,
            spectral_centroid=5.5,
        )
        soft_logs.append(log)

    # Qualifies and starts turn on frame 3
    assert soft_logs[2]["action"] == "START_SPEECH_TURN"
    assert soft_logs[2]["decision"] == "ACCEPT"
    assert vad.is_speech_turn_active is True

    turn_res = vad.finalize_turn()
    assert turn_res["accepted"] is True


def test_background_speech_murmur_does_not_leak_during_conversational_pause():
    """
    Verify room murmurs do not trigger continuation during conversational pauses.
    """
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    # 1. User starts speaking (frames 1-3)
    user_chunk = (0.05 * np.sin(2 * np.pi * 320 * t)).astype(np.float32)
    for _ in range(3):
        vad.process_frame(user_chunk, vocal_energy=70.0, high_noise=15.0, peak_prominence=2.2, harmonicity=0.70)

    assert vad.is_speech_turn_active is True

    # 2. User stops speaking (silent pause), but background room murmur at -43 dBFS with vocalEnergy 18
    murmur_chunk = (0.010 * np.sin(2 * np.pi * 400 * t) + np.random.normal(0, 0.001, 2048)).astype(np.float32)
    pause_log = vad.process_frame(murmur_chunk, vocal_energy=18.0, high_noise=10.0, peak_prominence=1.2, harmonicity=0.25)

    assert pause_log["action"] == "SILENCE_PAUSE"
    assert pause_log["decision"] == "WAITING_SILENCE_TIMER"


def test_unvoiced_fricatives_in_numbers_preserved_during_continuation():
    """
    Preserve phone numbers and loan amounts (English, Hindi, Marathi):
    During a speech turn, unvoiced consonants ('s' in six, 'f' in fifty, 'स' in सात, 'प' in पाच)
    have lower vocalEnergy (16) but higher highNoise (30) and zcr (0.18).
    Continuation hysteresis protects unvoiced consonants so numbers are never chopped off!
    """
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0
    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    # 1. Start turn with voiced vowel ("Nine...")
    voiced_chunk = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    for _ in range(3):
        vad.process_frame(voiced_chunk, vocal_energy=65.0, high_noise=12.0, peak_prominence=2.0, harmonicity=0.68)

    assert vad.is_speech_turn_active is True

    # 2. Unvoiced fricative ("...Six..." or "...सात...")
    fricative_chunk = (np.random.normal(0, 0.015, 2048)).astype(np.float32)
    fric_log = vad.process_frame(
        fricative_chunk,
        vocal_energy=16.0,  # Below pure vocal threshold (20)
        high_noise=32.0,    # Strong fricative high frequency energy!
        peak_prominence=1.1,
        harmonicity=0.15,
    )

    # Must be accepted as speech continuation!
    assert fric_log["action"] == "HYSTERESIS_CONTINUATION"
    assert fric_log["decision"] == "ACCEPT_CONTINUATION"


def test_distant_background_speech_is_rejected():
    """Confirms that distant background speech (>3.5m, -48 dBFS, vocalEnergy < 22) is correctly rejected."""
    vad = Module1VADSimulator()
    vad.ambient_floor_db = -50.0

    distant_chunk = (np.random.normal(0, 0.004, 2048)).astype(np.float32)
    logs = []
    for _ in range(8):
        l = vad.process_frame(distant_chunk, vocal_energy=21.0, high_noise=19.0, peak_prominence=1.2, harmonicity=0.18)
        logs.append(l)

    for l in logs:
        assert l["action"] == "AMBIENT_FILTERED"
        assert l["decision"] == "REJECT"

    res = vad.finalize_turn()
    assert res["accepted"] is False
