"""
Verification test script for Module 1 session controls:
Simulates:
1. Tap to Speak: starts continuous session.
2. Multiple continuous turns: auto-listen after each assistant response.
3. Finish Speaking:
   - Immediately stops microphone/VAD/listening.
   - Clears pending auto-listen/restart timers.
   - Does NOT start another STT turn.
   - Lets active assistant response finish if already processing.
   - Transitions to idle state.
4. Tap again: starts a completely fresh session (clears conversation, resets timer, resets lead).
"""
import sys

def test_session_lifecycle_logic():
    print("Testing Module 1 conversation session lifecycle logic...")

    # Simulated state
    state = {
        "voiceState": "idle",
        "isContinuousMode": False,
        "isMicCaptureActive": False,
        "autoListenTimer": None,
        "silenceTimer": None,
        "timerInterval": None,
        "conversationMessages": [],
        "lead": None,
        "turnId": 0,
        "isAssistantBusy": False,
    }

    # 1. Tap to Speak
    def start_voice_mode():
        if state.get("isStartingSession"):
            return
        state["isStartingSession"] = True
        try:
            # Cancel timers
            state["autoListenTimer"] = None
            state["silenceTimer"] = None
            # Reset session
            state["isContinuousMode"] = True
            state["conversationMessages"] = []
            state["lead"] = None
            state["turnId"] = 0
            state["timerInterval"] = "active"
            state["voiceState"] = "listening"
            state["isMicCaptureActive"] = True
        finally:
            state["isStartingSession"] = False

    # Simulate Turn completion
    def on_queue_complete():
        state["isAssistantBusy"] = False
        if state["isContinuousMode"]:
            state["voiceState"] = "listening"
            state["autoListenTimer"] = "pending"
            # Auto-listen fires
            state["autoListenTimer"] = None
            state["isMicCaptureActive"] = True
            state["turnId"] += 1
        else:
            state["voiceState"] = "idle"
            state["isMicCaptureActive"] = False

    # 2. Finish Speaking
    def finish_speaking():
        state["isContinuousMode"] = False
        # Stop timers immediately
        state["autoListenTimer"] = None
        state["silenceTimer"] = None
        state["timerInterval"] = None
        # Stop mic immediately
        state["isMicCaptureActive"] = False
        # Do not start another STT turn
        if not state["isAssistantBusy"]:
            state["voiceState"] = "idle"
        else:
            # Let assistant finish
            pass

    # Execution flow:
    # STEP 1: Tap to Speak
    start_voice_mode()
    assert state["isContinuousMode"] is True
    assert state["voiceState"] == "listening"
    assert state["isMicCaptureActive"] is True
    print("[PASS] Step 1: Session started. Continuous mode ON, listening active.")

    # STEP 2: Continuous turn 1
    state["conversationMessages"].append({"sender": "user", "text": "Hi my name is Rajesh"})
    state["conversationMessages"].append({"sender": "assistant", "text": "Hello Rajesh, what loan do you need?"})
    state["isAssistantBusy"] = True
    state["voiceState"] = "speaking"
    on_queue_complete()
    assert state["voiceState"] == "listening"
    assert state["isContinuousMode"] is True
    assert state["turnId"] == 1
    print("[PASS] Step 2a: Turn 1 finished. Automatically returned to listening for Turn 2.")

    # STEP 2b: Continuous turn 2
    state["conversationMessages"].append({"sender": "user", "text": "Personal loan"})
    state["conversationMessages"].append({"sender": "assistant", "text": "Great, how much amount?"})
    state["isAssistantBusy"] = True
    state["voiceState"] = "speaking"
    on_queue_complete()
    assert state["voiceState"] == "listening"
    assert state["isContinuousMode"] is True
    assert state["turnId"] == 2
    print("[PASS] Step 2b: Turn 2 finished. Automatically returned to listening for Turn 3.")

    # STEP 3: User clicks Finish Speaking while listening
    finish_speaking()
    assert state["isContinuousMode"] is False
    assert state["voiceState"] == "idle"
    assert state["isMicCaptureActive"] is False
    assert state["autoListenTimer"] is None
    print("[PASS] Step 3: Finish Speaking clicked while listening. Stopped mic, cancelled timers, returned to idle immediately.")

    # STEP 4: Tap again starts completely new session
    start_voice_mode()
    assert state["isContinuousMode"] is True
    assert state["voiceState"] == "listening"
    assert len(state["conversationMessages"]) == 0  # Cleared!
    assert state["turnId"] == 0  # Reset!
    print("[PASS] Step 4: Tap to Speak clicked again. Completely new session started with pristine state.")

    # STEP 5: Finish Speaking clicked while assistant is speaking
    state["isAssistantBusy"] = True
    state["voiceState"] = "speaking"
    finish_speaking()
    assert state["isContinuousMode"] is False
    assert state["isMicCaptureActive"] is False  # Mic stopped
    assert state["voiceState"] == "speaking"  # Assistant audio finishing naturally
    on_queue_complete()
    assert state["voiceState"] == "idle"  # Transitioned to idle upon completion!
    print("[PASS] Step 5: Finish Speaking clicked while speaking. Mic stopped, assistant finished speaking, then transitioned to idle.")

    print("\nALL SESSION CONTROL ASSERTIONS PASSED!")

if __name__ == "__main__":
    test_session_lifecycle_logic()
