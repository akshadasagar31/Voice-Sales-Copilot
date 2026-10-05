import math
import numpy as np

class DiagnosticAdaptiveDbVAD:
    """
    Exact mirror of frontend/app/components/LiveCallVoiceCopilot.tsx VAD logic
    with detailed per-frame telemetry logging.
    """
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.ambient_floor_db = -50.0
        self.speaker_bleed_db = -45.0
        self.is_speech_turn_active = False
        self.consecutive_speech_frames = 0
        self.pre_roll_buffers = []
        self.max_pre_roll_count = max(16, math.ceil((sample_rate * 0.85) / buffer_size))
        self.recorded_buffers = []
        self.silence_timer_running = False
        self.logs = []

    def compute_features(self, chunk: np.ndarray):
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

    def process_chunk(self, chunk: np.ndarray, vocal_energy: float, high_noise: float, label: str = ""):
        rms, current_db, max_peak, crest_factor, zcr = self.compute_features(chunk)

        if not self.is_speech_turn_active:
            self.ambient_floor_db = max(-65.0, min(-35.0, self.ambient_floor_db * 0.96 + current_db * 0.04))
            self.pre_roll_buffers.append(chunk)
            if len(self.pre_roll_buffers) > self.max_pre_roll_count:
                self.pre_roll_buffers.pop(0)

        # SNR calculation
        current_snr = current_db - self.ambient_floor_db
        snr_margin_db = max(9.0, min(14.0, 11.0 - (self.ambient_floor_db + 45.0) * 0.25))
        dynamic_db_onset_threshold = max(-42.0, self.ambient_floor_db + snr_margin_db)

        # Criteria breakdown
        c1_db = current_db >= dynamic_db_onset_threshold
        c2_vocal = vocal_energy >= 28
        c3_formant = vocal_energy > high_noise * 1.30
        c4_crest = (crest_factor <= 5.8 or vocal_energy >= 50)
        c5_zcr = (zcr <= 0.22 or vocal_energy >= 40)
        is_speech_candidate = c1_db and c2_vocal and c3_formant and c4_crest and c5_zcr

        action = "ambient"
        decision = "IGNORE"

        if is_speech_candidate:
            self.consecutive_speech_frames += 1
            if not self.is_speech_turn_active and self.consecutive_speech_frames >= 3:
                self.is_speech_turn_active = True
                self.recorded_buffers = list(self.pre_roll_buffers) + [chunk]
                action = "START_TURN (onset confirmed)"
                decision = "ACCEPT"
            elif self.is_speech_turn_active:
                self.recorded_buffers.append(chunk)
                action = "STREAMING_SPEECH"
                decision = "ACCEPT"
            else:
                action = f"QUALIFYING (frame {self.consecutive_speech_frames}/3)"
                decision = "PENDING"
        else:
            self.consecutive_speech_frames = 0
            if self.is_speech_turn_active:
                self.recorded_buffers.append(chunk)
                continuation_threshold_db = max(-48.0, self.ambient_floor_db + 4.5)
                is_continuation = current_db >= continuation_threshold_db and vocal_energy >= 16
                if is_continuation:
                    action = "HYSTERESIS_CONTINUATION"
                    decision = "ACCEPT (holdover)"
                else:
                    action = "SILENCE_PAUSE"
                    decision = "PAUSE_WAIT"
            else:
                action = "AMBIENT_REJECT"
                decision = "REJECT"

        log_entry = {
            "label": label,
            "db": round(current_db, 1),
            "floor_db": round(self.ambient_floor_db, 1),
            "snr_db": round(current_snr, 1),
            "onset_thresh": round(dynamic_db_onset_threshold, 1),
            "vocal_energy": round(vocal_energy, 1),
            "high_noise": round(high_noise, 1),
            "crest_factor": round(crest_factor, 2),
            "zcr": round(zcr, 3),
            "c_breakdown": f"dB:{c1_db}|Voc:{c2_vocal}|Fmt:{c3_formant}|Cr:{c4_crest}|ZCR:{c5_zcr}",
            "action": action,
            "decision": decision,
        }
        self.logs.append(log_entry)
        return log_entry

    def finalize_turn(self):
        if not self.recorded_buffers:
            return {"accepted": False, "reason": "empty"}
        total_samples = sum(len(b) for b in self.recorded_buffers)
        duration = total_samples / self.sample_rate
        if duration < 0.30:
            self.is_speech_turn_active = False
            return {"accepted": False, "reason": f"duration_under_300ms ({duration*1000:.0f}ms)"}

        total_energy = 0.0
        peak_rms = 0.0
        voiced_count = 0
        continuation_threshold_rms = 10.0 ** (max(-48.0, self.ambient_floor_db + 4.5) / 20.0)

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

        c1 = segment_avg_db >= self.ambient_floor_db + 3.0
        c2 = peak_db >= -40.0
        c3 = voiced_density >= 0.20
        accepted = c1 and c2 and c3

        reason = "PASSED" if accepted else f"FAILED (c1_avgDb:{c1}, c2_peakDb:{c2}, c3_density:{c3})"
        self.is_speech_turn_active = False
        return {
            "accepted": accepted,
            "duration": round(duration, 2),
            "segment_avg_db": round(segment_avg_db, 1),
            "peak_db": round(peak_db, 1),
            "voiced_density": round(voiced_density, 2),
            "reason": reason,
        }

def run_diagnostics():
    print("=" * 115)
    print("                      MODULE 1 VAD ACOUSTIC TELEMETRY & DECISION LOG")
    print("=" * 115)

    scenarios = [
        {
            "name": "1. Intended Primary User (Close Mic ~15cm)",
            "rms_mult": 0.055,  # ~ -26.0 dBFS
            "vocal_energy": 72.0,
            "high_noise": 16.0,
            "frames": 6,
            "description": "Close proximity intended speaker (-26 dBFS, vocal energy 72, SNR +24dB)",
        },
        {
            "name": "2. Nearby Secondary Speaker / Colleague (~1.0m)",
            "rms_mult": 0.025,  # ~ -33.2 dBFS
            "vocal_energy": 45.0,
            "high_noise": 14.0,
            "frames": 6,
            "description": "Coworker 1 meter away talking at normal conversation volume (-33 dBFS, vocal energy 45, SNR +15dB)",
        },
        {
            "name": "3. Distant TV / Diffuse Murmur (>3.5m)",
            "rms_mult": 0.004,  # ~ -47.9 dBFS
            "vocal_energy": 21.0,
            "high_noise": 18.0,
            "frames": 6,
            "description": "Distant TV down hallway (-48 dBFS, vocal energy 21, SNR +2dB)",
        },
        {
            "name": "4. Conversational Pause Murmur Leakage",
            "rms_mult": 0.010,  # ~ -42.9 dBFS
            "vocal_energy": 22.0,
            "high_noise": 10.0,
            "frames": 6,
            "is_pause": True,
            "description": "User pauses speaking, background room voice murmurs at -43 dBFS during hysteresis window",
        },
        {
            "name": "5. Steady Ambient Fan Noise / HVAC",
            "rms_mult": 0.008,  # ~ -41.9 dBFS
            "vocal_energy": 14.0,
            "high_noise": 13.0,
            "frames": 6,
            "is_fan": True,
            "description": "Steady air conditioner / fan rumble (-42 dBFS non-vocal)",
        },
        {
            "name": "6. Mechanical Keyboard Typing Clicks",
            "rms_mult": 0.018,
            "vocal_energy": 22.0,
            "high_noise": 48.0,
            "frames": 6,
            "is_click": True,
            "description": "High crest factor impulsive typing clicks (> 6.0)",
        },
    ]

    t = np.linspace(0, 2048 / 48000, 2048, endpoint=False)

    for sc in scenarios:
        print(f"\n>>> SCENARIO: {sc['name']}")
        print(f"    Description: {sc['description']}")
        print(f"    {'FRAME':<10} | {'dBFS':<7} | {'SNR':<7} | {'Vocal':<6} | {'Crest':<6} | {'ZCR':<6} | {'Thresh':<7} | {'DECISION':<18} | {'ACTION':<24}")
        print("    " + "-" * 105)

        vad = DiagnosticAdaptiveDbVAD()
        vad.ambient_floor_db = -50.0

        if sc.get("is_pause"):
            # Start turn first
            user_chunk = (0.04 * np.sin(2 * np.pi * 320 * t)).astype(np.float32)
            for _ in range(3):
                vad.process_chunk(user_chunk, 70.0, 15.0)

        for i in range(sc["frames"]):
            if sc.get("is_click"):
                chunk = np.zeros(2048, dtype=np.float32)
                chunk[50:70] = np.random.uniform(0.08, 0.14, 20)
                chunk[70:120] = np.random.uniform(-0.06, 0.06, 50) * np.exp(-np.linspace(0, 3, 50))
            elif sc.get("is_fan"):
                chunk = (sc["rms_mult"] * np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.002, 2048)).astype(np.float32)
            else:
                chunk = (
                    sc["rms_mult"] * 0.6 * np.sin(2 * np.pi * 320 * t)
                    + sc["rms_mult"] * 0.4 * np.sin(2 * np.pi * 780 * t)
                    + np.random.normal(0, 0.002, 2048)
                ).astype(np.float32)

            log = vad.process_chunk(chunk, sc["vocal_energy"], sc["high_noise"], label=f"Frame_{i+1}")
            thresh_val = log['onset_thresh'] if not vad.is_speech_turn_active or i < 3 else round(max(-48.0, log['floor_db'] + 4.5), 1)

            print(
                f"    {log['label']:<10} | "
                f"{log['db']:>6.1f} | "
                f"{'+'+str(log['snr_db']):>6} | "
                f"{log['vocal_energy']:>5.1f} | "
                f"{log['crest_factor']:>5.2f} | "
                f"{log['zcr']:>5.3f} | "
                f"{thresh_val:>6.1f} | "
                f"{log['decision']:<18} | "
                f"{log['action']:<24}"
            )

        final_res = vad.finalize_turn()
        status_str = "ACCEPTED" if final_res["accepted"] else f"REJECTED ({final_res['reason']})"
        print("    " + "-" * 105)
        print(f"    -> TURN FINALIZATION: {status_str} | Duration: {final_res.get('duration', 0)}s | Peak: {final_res.get('peak_db', 'N/A')} dBFS | Voiced Density: {final_res.get('voiced_density', 'N/A')}")

if __name__ == "__main__":
    run_diagnostics()


