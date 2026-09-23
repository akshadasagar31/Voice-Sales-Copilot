import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from services.language import (
    translate_indic_query_to_english,
    detect_language,
    LANG_HI,
    LANG_MR,
)
from services.retriever import VectorRetriever
from services.rag import RAGService


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: Cross-Lingual Indic Query Translation
# ---------------------------------------------------------------------------

def test_translate_marathi_docs_query():
    q = "वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?"
    translated = translate_indic_query_to_english(q, language=LANG_MR)
    assert "personal loan" in translated
    assert "documents" in translated or "checklist" in translated
    assert "required" in translated


def test_translate_hindi_docs_query():
    q = "पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?"
    translated = translate_indic_query_to_english(q, language=LANG_HI)
    assert "personal loan" in translated
    assert "documents" in translated or "checklist" in translated
    assert "required" in translated


def test_translate_marathi_cibil_query():
    q = "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?"
    translated = translate_indic_query_to_english(q, language=LANG_MR)
    assert "HDFC Bank" in translated
    assert "personal loan" in translated
    assert "CIBIL" in translated
    assert "minimum" in translated


def test_translate_hindi_interest_query():
    q = "पर्सनल लोन पर ब्याज दर क्या है?"
    translated = translate_indic_query_to_english(q, language=LANG_HI)
    assert "personal loan" in translated
    assert "interest" in translated or "rate" in translated or "ROI" in translated


def test_translate_english_query_unchanged():
    q = "What is the minimum CIBIL score for an HDFC personal loan?"
    translated = translate_indic_query_to_english(q, language="en")
    assert translated == q


# ---------------------------------------------------------------------------
# Integration Tests: Cross-Lingual Retrieval in VectorRetriever
# ---------------------------------------------------------------------------

def test_retriever_uses_cross_lingual_alignment_for_indic_queries():
    mock_pinecone = MagicMock()
    mock_pinecone.namespace = "sales_playbooks"
    mock_pinecone.search_records.return_value = [
        {
            "id": "doc_1",
            "score": 0.85,
            "text": "Documents Commonly Requested: Identity proof, address proof, income documents, bank statements.",
            "source": "Personal_Loan_General_Information.pdf",
            "page": 1,
        }
    ]

    retriever = VectorRetriever(pinecone_service=mock_pinecone)
    res = retriever.retrieve(
        query="पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?",
        top_k=3,
        language="hi",
    )

    assert len(res["results"]) == 1
    assert "Personal_Loan_General_Information.pdf" in res["results"][0]["source"]
    # Verify search_records was called with English-aligned query
    call_args = mock_pinecone.search_records.call_args_list[0][1]
    query_text = call_args["query_text"]
    assert "personal loan" in query_text
    assert "documents" in query_text or "checklist" in query_text


# ---------------------------------------------------------------------------
# End-to-End RAG Tests: English PDF -> Marathi/Hindi Answer Grounding
# ---------------------------------------------------------------------------

def test_rag_answers_marathi_query_from_english_pdf_context():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "chunk_1",
                "score": 0.90,
                "text": "Documents Commonly Requested: Identity proof, address proof, income documents, and bank statements.",
                "source": "Personal_Loan_General_Information.pdf",
                "page": 1,
            }
        ]
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "वैयक्तिक कर्जासाठी आवश्यक कागदपत्रांमध्ये ओळख पुरावा, पत्ता पुरावा, उत्पन्न दस्तऐवज आणि बँक स्टेटमेंट समाविष्ट आहेत."
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    rag = RAGService(
        retriever=mock_retriever,
        api_key="test_api_key",
        http_client=mock_client,
    )

    ans = rag.answer_question(
        question="वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?",
        language="mr",
    )

    assert ans["language"] == "mr"
    assert ans["fallback_used"] is False
    assert "कागदपत्रांमध्ये" in ans["answer"] or "पुरावा" in ans["answer"]
    assert len(ans["context_used"]) == 1
    assert ans["sources"] == ["Personal_Loan_General_Information.pdf"]


def test_rag_answers_hindi_query_from_english_pdf_context():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "chunk_1",
                "score": 0.92,
                "text": "Rate Card dated 07-03-25: CIBIL >730 gets rate-card pricing. CIBIL <=730 adds 25 bps to rack rate.",
                "source": "HDFC_Bank_Master_Policy_CIBIL_Updated.pdf",
                "page": 2,
            }
        ]
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "एचडीएफसी बैंक में सिबिल स्कोर 730 से अधिक होने पर स्टैंडर्ड रेट कार्ड प्राइसिंग लागू होती है, और 730 या उससे कम होने पर 25 बेसिस पॉइंट्स अतिरिक्त जोड़े जाते हैं।"
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    rag = RAGService(
        retriever=mock_retriever,
        api_key="test_api_key",
        http_client=mock_client,
    )

    ans = rag.answer_question(
        question="एचडीएफसी बैंक में पर्सनल लोन के लिए सिबिल स्कोर का क्या नियम है?",
        language="hi",
    )

    assert ans["language"] == "hi"
    assert ans["fallback_used"] is False
    assert "सिबिल स्कोर" in ans["answer"]
    assert len(ans["context_used"]) == 1
    assert ans["sources"] == ["HDFC_Bank_Master_Policy_CIBIL_Updated.pdf"]
