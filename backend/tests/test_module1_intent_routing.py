# ============================================================================
# MODULE 1 COMPREHENSIVE TESTS: INTENT ROUTING
# (backend/tests/test_module1_intent_routing.py)
# ============================================================================
# Verifies:
# 1. Clear loan/lead intent -> starts lead flow.
# 2. Generic/unrelated questions -> active LLM response, NO lead flow.
# 3. Loan-related questions without application intent -> generic response + invitation to apply, NO lead flow.
# 4. If lead flow is active, answer generic/loan questions without losing lead data, then resume pending field.
# 5. User introduction without loan intent -> greets user, NO lead flow.
# 6. Intent detected from the current user turn only.
# 7. Dynamic multilingual support (English, Hindi, Marathi).
# ============================================================================

import json
import pytest
from unittest.mock import MagicMock

from main import generate_module1_ws_response
from services.lead_extractor import (
    Lead,
    classify_turn_intent,
    is_loan_intent,
    is_loan_informational_question,
    is_general_question,
    INTENT_LOAN_APPLICATION,
    INTENT_LOAN_INFO_QUESTION,
    INTENT_GENERIC_QUESTION,
    INTENT_GREETING_OR_INTRO,
    INTENT_LEAD_DATA,
)


# ----------------------------------------------------------------------------
# 1. Semantic Intent Classification Tests
# ----------------------------------------------------------------------------
def test_classify_turn_intent_clear_loan_application():
    """Verify that explicit loan application intent is accurately detected."""
    # English
    assert classify_turn_intent("I want to apply for a personal loan") == INTENT_LOAN_APPLICATION
    assert classify_turn_intent("I need a loan of 5 lakhs") == INTENT_LOAN_APPLICATION
    assert classify_turn_intent("Looking for 10 lakh home loan") == INTENT_LOAN_APPLICATION

    # Hindi
    assert classify_turn_intent("मुझे पर्सनल लोन चाहिए") == INTENT_LOAN_APPLICATION
    assert classify_turn_intent("लोन के लिए अप्लाई करना है") == INTENT_LOAN_APPLICATION

    # Marathi
    assert classify_turn_intent("मला पाच लाख कर्ज हवे आहे") == INTENT_LOAN_APPLICATION
    assert classify_turn_intent("कर्जासाठी अर्ज करायचा आहे") == INTENT_LOAN_APPLICATION


def test_classify_turn_intent_loan_informational_queries():
    """Verify that loan questions without application intent are classified as loan_info_question."""
    # English
    assert classify_turn_intent("What are your interest rates?") == INTENT_LOAN_INFO_QUESTION
    assert classify_turn_intent("What documents are required for a personal loan?") == INTENT_LOAN_INFO_QUESTION
    assert classify_turn_intent("How does the loan approval process work?") == INTENT_LOAN_INFO_QUESTION
    assert classify_turn_intent("What is the minimum CIBIL score for home loan?") == INTENT_LOAN_INFO_QUESTION
    assert classify_turn_intent("I want to know about your personal loan interest rate") == INTENT_LOAN_INFO_QUESTION

    # Hindi
    assert classify_turn_intent("पर्सनल लोन की ब्याज दर क्या है?") == INTENT_LOAN_INFO_QUESTION
    assert classify_turn_intent("लोन के लिए क्या नियम हैं?") == INTENT_LOAN_INFO_QUESTION

    # Marathi
    assert classify_turn_intent("कर्जाचे व्याजदर किती आहेत?") == INTENT_LOAN_INFO_QUESTION
    assert classify_turn_intent("कागदपत्रे काय लागतील?") == INTENT_LOAN_INFO_QUESTION


def test_classify_turn_intent_generic_unrelated_queries():
    """Verify that trivia, weather, and general questions are classified as generic_question."""
    # English
    assert classify_turn_intent("What is the capital of France?") == INTENT_GENERIC_QUESTION
    assert classify_turn_intent("What is the weather today?") == INTENT_GENERIC_QUESTION
    assert classify_turn_intent("Who are you?") == INTENT_GENERIC_QUESTION
    assert classify_turn_intent("Tell me a joke") == INTENT_GENERIC_QUESTION

    # Hindi
    assert classify_turn_intent("भारत की राजधानी क्या है?") == INTENT_GENERIC_QUESTION
    assert classify_turn_intent("आप कौन हैं?") == INTENT_GENERIC_QUESTION

    # Marathi
    assert classify_turn_intent("महाराष्ट्राची राजधानी काय आहे?") == INTENT_GENERIC_QUESTION
    assert classify_turn_intent("तुम्ही कोण आहात?") == INTENT_GENERIC_QUESTION


def test_classify_turn_intent_greetings_and_intros_without_loan_intent():
    """Verify that greetings and name self-introductions without loan intent are greeting_or_intro."""
    assert classify_turn_intent("Hello") == INTENT_GREETING_OR_INTRO
    assert classify_turn_intent("Good morning") == INTENT_GREETING_OR_INTRO
    assert classify_turn_intent("Hi, my name is Rahul") == INTENT_GREETING_OR_INTRO
    assert classify_turn_intent("This is Sarah") == INTENT_GREETING_OR_INTRO
    assert classify_turn_intent("मेरा नाम अमित है") == INTENT_GREETING_OR_INTRO
    assert classify_turn_intent("माझे नाव सचिन आहे") == INTENT_GREETING_OR_INTRO


# ----------------------------------------------------------------------------
# 2. Fresh Turn Routing (No Active Lead)
# ----------------------------------------------------------------------------
def test_fresh_turn_loan_question_does_not_start_lead_flow():
    """
    Loan-related question without application intent must:
    - Return an active LLM generic response + polite invitation to apply.
    - NOT start a lead flow (lead is empty, next_missing_parameter is None).
    """
    resp = generate_module1_ws_response(
        current="What are your interest rates for a personal loan?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    # Must NOT start lead flow
    assert resp["next_missing_parameter"] is None
    assert resp["lead"] == {}
    assert resp["is_new_lead"] is False
    assert resp["is_assistant_query"] is True
    # Immediate sentence must answer the question
    sentence = resp["immediate_sentence1"].lower()
    assert "interest" in sentence or "rate" in sentence or "loan" in sentence or "10.5" in sentence


def test_fresh_turn_generic_question_does_not_start_lead_flow():
    """
    Generic/unrelated question must:
    - Return an active LLM generic response.
    - NOT start a lead flow (lead is empty, next_missing_parameter is None).
    """
    resp = generate_module1_ws_response(
        current="What is the capital of Australia?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    # Must NOT start lead flow
    assert resp["next_missing_parameter"] is None
    assert resp["lead"] == {}
    assert resp["is_new_lead"] is False
    assert resp["is_assistant_query"] is True
    assert bool(resp["immediate_sentence1"]) is True


def test_fresh_turn_self_intro_without_loan_intent_does_not_start_lead_flow():
    """
    User introducing name without loan intent must:
    - Greet user politely by name.
    - NOT start a lead flow (lead is empty, next_missing_parameter is None).
    """
    resp = generate_module1_ws_response(
        current="Hello, my name is Rahul",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    # Must NOT start lead flow
    assert resp["next_missing_parameter"] is None
    assert resp["lead"] == {}
    assert resp["is_new_lead"] is False
    # Greeting / welcome acknowledged
    sentence = resp["immediate_sentence1"].lower()
    assert "rahul" in sentence or "hello" in sentence or "help" in sentence


def test_fresh_turn_clear_loan_intent_starts_lead_flow():
    """
    Clear loan application intent on a fresh turn must start lead flow.
    """
    resp = generate_module1_ws_response(
        current="I want to apply for a personal loan of 5 lakhs",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    # Must start lead flow!
    assert resp["is_new_lead"] is True
    assert resp["next_missing_parameter"] is not None
    assert resp["lead"]["loan_type"] == "Personal Loan"
    assert resp["lead"]["loan_amount"] == 500000.0


# ----------------------------------------------------------------------------
# 3. Active Lead Flow: Questions Answered Without Losing Data, Then Resume
# ----------------------------------------------------------------------------
def test_active_lead_flow_loan_question_preserves_lead_and_resumes():
    """
    When lead flow is active (e.g. pending company) and user asks a loan question:
    - Answer the question.
    - Preserve all previously collected lead data intact.
    - Keep next_missing_parameter on the pending field ('company').
    - Prompt transitions back to the pending field.
    """
    existing_lead = {
        "name": "Rajesh Kumar",
        "phone": "9876543210",
    }
    resp = generate_module1_ws_response(
        current="What are your interest rates?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )
    # 1. Lead data must be 100% preserved
    assert resp["lead"]["name"] == "Rajesh Kumar"
    assert resp["lead"]["phone"] == "9876543210"
    # 2. Next missing parameter remains 'company'
    assert resp["next_missing_parameter"] == "company"
    assert resp["is_unclear"] is False
    # 3. Immediate sentence answers question and resumes company
    sentence = resp["immediate_sentence1"].lower()
    assert ("company" in sentence or "employer" in sentence or "work" in sentence)


def test_active_lead_flow_generic_question_preserves_lead_and_resumes():
    """
    When lead flow is active and user asks an unrelated generic question:
    - Answer the question.
    - Preserve all previously collected lead data intact.
    - Keep next_missing_parameter on the pending field ('phone').
    - Prompt transitions back to the pending field.
    """
    existing_lead = {
        "name": "Pooja Sharma",
    }
    resp = generate_module1_ws_response(
        current="What is the capital of France?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )
    # 1. Lead data preserved
    assert resp["lead"]["name"] == "Pooja Sharma"
    # 2. Next missing parameter remains 'phone'
    assert resp["next_missing_parameter"] == "phone"
    assert resp["is_unclear"] is False
    # 3. Immediate sentence prompts to resume phone
    sentence = resp["immediate_sentence1"].lower()
    assert ("phone" in sentence or "number" in sentence or "contact" in sentence)


def test_subsequent_turn_after_question_satisfies_pending_field():
    """
    Verify complete multi-turn flow:
    Turn 1: User provides name ("Amit Patel") -> lead created, pending phone.
    Turn 2: User asks "What documents are required?" -> question answered, phone resumed.
    Turn 3: User gives phone "9876543210" -> phone captured, advances to company!
    """
    # Turn 1
    resp1 = generate_module1_ws_response(
        current="I want a personal loan, my name is Amit Patel",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    assert resp1["lead"]["name"] == "Amit Patel"
    assert resp1["next_missing_parameter"] == "phone"

    # Turn 2: Question
    resp2 = generate_module1_ws_response(
        current="What documents are needed for this?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(resp1["lead"]),
    )
    assert resp2["lead"]["name"] == "Amit Patel"
    assert resp2["next_missing_parameter"] == "phone"

    # Turn 3: Answer to pending field
    resp3 = generate_module1_ws_response(
        current="My phone number is 9876543210",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(resp2["lead"]),
    )
    assert resp3["lead"]["name"] == "Amit Patel"
    assert resp3["lead"]["phone"] == "9876543210"
    assert resp3["next_missing_parameter"] == "company"


# ----------------------------------------------------------------------------
# 4. Turn Isolation: Current Turn Only
# ----------------------------------------------------------------------------
def test_intent_detected_from_current_turn_only():
    """
    Intent must NOT be inherited from previous turns.
    Even if the previous turn had a question or loan intent,
    the current turn is evaluated strictly on its own transcript.
    """
    # Previous turn had a loan question; current turn is generic query
    resp1 = generate_module1_ws_response(
        current="What is personal loan interest rate?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    assert resp1["lead"] == {}
    assert resp1["next_missing_parameter"] is None

    # Next turn from same user is trivia
    resp2 = generate_module1_ws_response(
        current="Tell me a joke",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    assert resp2["lead"] == {}
    assert resp2["next_missing_parameter"] is None

    # Next turn expresses loan intent
    resp3 = generate_module1_ws_response(
        current="I want to apply for 5 lakh loan",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str="",
    )
    assert resp3["is_new_lead"] is True
    assert resp3["lead"]["loan_amount"] == 500000.0
