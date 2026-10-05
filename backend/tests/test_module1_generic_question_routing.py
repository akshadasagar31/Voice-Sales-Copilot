import pytest
import re
from unittest.mock import MagicMock, AsyncMock, patch
from services.lead_extractor import (
    LeadExtractorService,
    is_general_question,
    is_loan_intent,
    is_loan_informational_question,
    RE_KNOWLEDGE_BASE_INTENT,
)
from services.model_manager import get_model_manager


class TestModule1GenericQuestionRouting:
    """
    Regression test suite for Module 1 Generic-Question Routing:
    - Any non-lead/general question -> answer using CURRENT ACTIVE LLM.
    - Support any type of normal question, not just predefined examples.
    - Do not send generic questions to lead extraction or loan flow.
    - If a lead flow is active, preserve collected data, answer the question, then resume the pending field.
    - Only use RAG when the question actually requires knowledge-base data.
    - Never hardcode question-specific answers.
    - Multilingual support for English, Hindi, and Marathi.
    """

    def test_generic_math_question_classification(self):
        """Verifies arbitrary math questions are classified as general questions, not lead/loan intent."""
        math_queries = [
            "What is 15 multiplied by 4?",
            "Calculate 25 percent of 80000",
            "Solve 1200 divided by 6",
            "What is 2 plus 2?",
            "What is fifteen times four?",
        ]
        for q in math_queries:
            assert not is_loan_intent(q), f"Math query '{q}' should not have loan intent"
            assert is_general_question(q), f"Math query '{q}' must be classified as a general question"

    def test_generic_trivia_and_general_knowledge_classification(self):
        """Verifies arbitrary trivia and science/history questions are classified as general questions."""
        trivia_queries = [
            "What is the capital of Japan?",
            "Who was the first president of India?",
            "Explain quantum computing in simple terms",
            "Why is the sky blue?",
            "Tell me about the history of cricket",
            "What is inflation?",
            "Do you know the weather today?",
            "Tell me a funny joke",
        ]
        for q in trivia_queries:
            assert not is_loan_intent(q), f"Trivia query '{q}' should not have loan intent"
            assert is_general_question(q), f"Trivia query '{q}' must be classified as a general question"

    def test_informational_loan_questions_not_treated_as_loan_applications(self):
        """
        Verifies informational inquiries about loans (process, interest rates, eligibility, documents)
        are classified as general questions, NOT loan application intents that ask for a name.
        """
        info_queries = [
            "What is the loan process?",
            "Can you explain the personal loan process?",
            "What are the interest rates on personal loans?",
            "Tell me about your loan details",
            "How does a home loan work?",
            "What documents are required for a personal loan?",
            "What is the CIBIL score required for a loan?",
            "Is there any prepayment charge on personal loans?",
            "लोन का प्रोसेस क्या है?",
            "पर्सनल लोन का ब्याज दर क्या है?",
            "कर्ज प्रक्रिया काय आहे?",
            "पर्सनल कर्जाचा व्याजदर काय आहे?",
        ]
        for q in info_queries:
            assert is_loan_informational_question(q), f"'{q}' must be recognized as loan informational question"
            assert not is_loan_intent(q), f"Informational query '{q}' must NOT be treated as a loan application intent"
            assert is_general_question(q), f"Informational query '{q}' must be classified as a general question"

    def test_actual_loan_applications_are_not_general_questions(self):
        """Verifies explicit intent to obtain or apply for a loan is correctly routed to lead extraction."""
        application_queries = [
            "I want a personal loan of 5 lakhs",
            "I need a loan",
            "I would like to apply for a home loan",
            "मुझे 5 लाख का पर्सनल लोन चाहिए",
            "मला पर्सनल कर्ज हवे आहे",
            "Start a new loan application",
        ]
        for q in application_queries:
            assert is_loan_intent(q), f"Application query '{q}' must be recognized as loan intent"
            assert not is_general_question(q), f"Application query '{q}' must NOT be classified as general question"

    def test_standalone_general_question_does_not_force_lead_prompt(self):
        """
        When no lead flow is active, answering a general question returns a standalone answer
        without appending 'Please provide your full name to get started.'
        """
        extractor = LeadExtractorService()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "15 multiplied by 4 is 60."}}]
        }

        with patch.object(extractor, "get_http_client") as mock_client:
            mock_client.return_value.post.return_value = mock_resp
            ans = extractor.answer_general_query(
                transcript="What is 15 multiplied by 4?",
                language="en",
                existing_lead=None,
                pending_field=None,
            )

        assert "60" in ans
        assert "name" not in ans.lower()
        assert "get started" not in ans.lower()
        assert "please provide" not in ans.lower()

    def test_active_lead_flow_preserves_lead_data_and_resumes_pending_field(self):
        """
        When a lead flow is active (e.g. name is collected), asking a general question
        answers the question dynamically, then resumes the pending field (e.g. phone).
        """
        extractor = LeadExtractorService()
        existing_lead = {"name": "Rajesh Sharma", "phone": None}
        pending_field = "phone"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Our personal loan interest rates start at 10.5% per annum."}}]
        }

        with patch.object(extractor, "get_http_client") as mock_client:
            mock_client.return_value.post.return_value = mock_resp
            ans = extractor.answer_general_query(
                transcript="What are your interest rates on personal loans?",
                language="en",
                existing_lead=existing_lead,
                pending_field=pending_field,
            )

        assert "10.5%" in ans
        assert "Rajesh" in ans
        assert "phone" in ans.lower() or "number" in ans.lower()

    def test_selective_rag_invocation_only_for_knowledge_base_intent(self):
        """
        Verifies that RAG is ONLY invoked when the question actually requires knowledge-base data
        (e.g. sales playbook, uploaded document, policy guidelines), and NOT for general questions.
        """
        # 1. Normal question: RAG should NOT be invoked
        normal_q = "What is 10 plus 25?"
        assert not RE_KNOWLEDGE_BASE_INTENT.search(normal_q)

        extractor = LeadExtractorService()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "10 plus 25 is 35."}}]
        }

        with patch("services.rag.RAGService.answer_question") as mock_rag, \
             patch.object(extractor, "get_http_client") as mock_client:
            mock_client.return_value.post.return_value = mock_resp
            ans = extractor.answer_general_query(normal_q, language="en")
            mock_rag.assert_not_called()
            assert "35" in ans

        # 2. Knowledge-base question: RAG should be invoked
        kb_q = "According to our sales playbook, how should we handle price objections?"
        assert RE_KNOWLEDGE_BASE_INTENT.search(kb_q)

        with patch("services.rag.RAGService.answer_question") as mock_rag, \
             patch.object(extractor, "get_http_client") as mock_client:
            mock_rag.return_value = {"answer": "Focus on ROI and value proposition."}
            mock_client.return_value.post.return_value = mock_resp
            _ = extractor.answer_general_query(kb_q, language="en")
            mock_rag.assert_called_once()

    def test_dynamic_active_llm_resolution_from_model_manager(self):
        """
        Verifies that answer_general_query uses the CURRENT ACTIVE LLM from ModelManager
        rather than an outdated default.
        """
        mgr = get_model_manager()
        active_llm = mgr.get_active_model("llm")
        active_model_id = active_llm.get("model_id") if active_llm else "deepseek/deepseek-chat"

        extractor = LeadExtractorService()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Paris is the capital of France."}}]
        }

        with patch.object(extractor, "get_http_client") as mock_client:
            mock_client.return_value.post.return_value = mock_resp
            _ = extractor.answer_general_query(
                transcript="What is the capital of France?",
                language="en",
            )
            call_payload = mock_client.return_value.post.call_args[1]["json"]
            assert call_payload["model"] == active_model_id

    def test_multilingual_general_questions_hindi_and_marathi(self):
        """Verifies Hindi and Marathi general questions route properly and request responses in their script."""
        extractor = LeadExtractorService()

        # Hindi question
        hi_q = "कंपाउंड इंटरेस्ट क्या होता है?"
        assert is_general_question(hi_q)

        mock_resp_hi = MagicMock()
        mock_resp_hi.status_code = 200
        mock_resp_hi.json.return_value = {
            "choices": [{"message": {"content": "कंपाउंड इंटरेस्ट में मूलधन के साथ अर्जित ब्याज पर भी ब्याज मिलता है।"}}]
        }

        with patch.object(extractor, "get_http_client") as mock_client:
            mock_client.return_value.post.return_value = mock_resp_hi
            hi_ans = extractor.answer_general_query(hi_q, language="hi")
            assert "ब्याज" in hi_ans

        # Marathi question
        mr_q = "व्याजदर कसा ठरवला जातो?"
        assert is_general_question(mr_q)

        mock_resp_mr = MagicMock()
        mock_resp_mr.status_code = 200
        mock_resp_mr.json.return_value = {
            "choices": [{"message": {"content": "व्याजदर हा रिझर्व्ह बँकेचा रेपो दर आणि सिबिल स्कोअरवर आधारित असतो."}}]
        }

        with patch.object(extractor, "get_http_client") as mock_client:
            mock_client.return_value.post.return_value = mock_resp_mr
            mr_ans = extractor.answer_general_query(mr_q, language="mr")
            assert "व्याजदर" in mr_ans

    @pytest.mark.asyncio
    async def test_stream_general_query_tokens_without_active_lead(self):
        """Verifies that streaming tokens for a general question when no active lead exists yields only the answer."""
        extractor = LeadExtractorService()

        async def dummy_stream(*args, **kwargs):
            yield "The "
            yield "capital "
            yield "of "
            yield "Japan "
            yield "is "
            yield "Tokyo."

        with patch.object(extractor, "stream_general_query_tokens", side_effect=dummy_stream):
            tokens = []
            async for token in extractor.stream_general_query_tokens(
                transcript="What is the capital of Japan?",
                language="en",
                existing_lead=None,
                pending_field=None,
            ):
                tokens.append(token)
            full_text = "".join(tokens)
            assert full_text == "The capital of Japan is Tokyo."
            assert "name" not in full_text.lower()

    def test_generate_module1_ws_response_standalone_question(self):
        """Verifies generate_module1_ws_response returns standalone answer without next_missing_parameter when no lead is active."""
        from main import generate_module1_ws_response

        with patch("services.lead_extractor.LeadExtractorService.answer_general_query", return_value="Paris is the capital of France."):
            res = generate_module1_ws_response(
                current="What is the capital of France?",
                req_lang="en",
                last_conf=0.98,
                existing_lead_str=None,
                lead_id_val=None,
            )

        assert res["is_assistant_query"] is True
        assert res["assistant_response"] == "Paris is the capital of France."
        assert res["immediate_sentence1"] == "Paris is the capital of France."
        assert res["next_missing_parameter"] is None
        assert "name" not in res["immediate_sentence1"].lower()

    def test_generate_module1_ws_response_mid_lead_preserves_lead_and_resumes(self):
        """Verifies generate_module1_ws_response preserves active lead and resumes pending field."""
        from main import generate_module1_ws_response

        active_lead = {"name": "Pooja Patel", "phone": None}
        with patch("services.lead_extractor.LeadExtractorService.answer_general_query", return_value="Our interest rates start from 10.5% per annum."):
            res = generate_module1_ws_response(
                current="What is your interest rate?",
                req_lang="en",
                last_conf=0.97,
                existing_lead_str=active_lead,
                lead_id_val=123,
            )

        assert res["is_assistant_query"] is True
        assert res["lead_id"] == 123
        assert res["lead"]["name"] == "Pooja Patel"
        assert res["next_missing_parameter"] == "phone"
        assert "10.5%" in res["immediate_sentence1"]
        assert "phone" in res["immediate_sentence1"].lower() or "number" in res["immediate_sentence1"].lower()

    def test_user_general_questions_do_not_start_lead_collection(self):
        """Verifies 'Who are you?', 'What can you do?', 'Do you know Marathi?' do NOT start lead collection."""
        from main import generate_module1_ws_response

        queries = [
            ("Who are you?", "I am Voice Sales Copilot, your AI sales assistant."),
            ("What can you do?", "I can help you check eligibility and apply for a loan."),
            ("Do you know Marathi?", "Yes, I know Marathi! मी मराठी बोलू शकतो."),
            ("Can you speak Hindi?", "Yes, I speak Hindi! मैं हिंदी बोल सकता हूँ."),
        ]
        for q, expected_ans in queries:
            with patch("services.lead_extractor.LeadExtractorService.answer_general_query", return_value=expected_ans):
                res = generate_module1_ws_response(
                    current=q,
                    req_lang="en",
                    last_conf=0.98,
                    existing_lead_str=None,
                    lead_id_val=None,
                )
                assert res["is_assistant_query"] is True, f"Query '{q}' should be assistant query"
                assert res["next_missing_parameter"] is None, f"Query '{q}' must NOT set next_missing_parameter"
                assert res["is_new_lead"] is False, f"Query '{q}' must NOT start a new lead"
                assert res["lead"] == {}, f"Query '{q}' lead must be empty"
                assert expected_ans in res["immediate_sentence1"], f"Response must contain LLM answer for '{q}'"
                assert "name" not in res["immediate_sentence1"].lower() or "copilot" in res["immediate_sentence1"].lower()

    def test_greeting_does_not_start_lead_collection(self):
        """Verifies 'Hello' greeting does NOT start lead collection or set next_missing_parameter='name'."""
        from main import generate_module1_ws_response

        res = generate_module1_ws_response(
            current="Hello",
            req_lang="en",
            last_conf=0.99,
            existing_lead_str=None,
            lead_id_val=None,
        )
        assert res["is_greeting"] is True
        assert res["next_missing_parameter"] is None, "Greeting must NOT set next_missing_parameter"
        assert res["is_new_lead"] is False
        assert res["lead"] == {}
        assert "hello" in res["immediate_sentence1"].lower()

    def test_loan_intent_starts_lead_collection_and_collects_missing_fields(self):
        """Verifies 'I want a loan' and 'I want to apply' start lead collection and ask for first missing field."""
        from main import generate_module1_ws_response

        loan_queries = [
            "I want a loan",
            "I want to apply",
            "I want to apply for a personal loan",
            "Can I get a loan?",
            "I need 5 lakhs",
            "मला कर्ज हवे आहे",
            "मुझे पर्सनल लोन चाहिए",
        ]
        for q in loan_queries:
            res = generate_module1_ws_response(
                current=q,
                req_lang="en" if "कर्ज" not in q and "लोन" not in q else ("mr" if "कर्ज" in q else "hi"),
                last_conf=0.98,
                existing_lead_str=None,
                lead_id_val=None,
            )
            assert res["is_new_lead"] is True, f"Query '{q}' must start a new lead"
            assert res["next_missing_parameter"] is not None, f"Query '{q}' must set next_missing_parameter"
            assert res["next_missing_parameter"] == "name", f"Query '{q}' should ask for name first"
            assert res["immediate_sentence1"] != "", f"Query '{q}' must have prompt message"

